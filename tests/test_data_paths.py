import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from miniwam.config import load_training_config
from scripts.data_paths import data_path, data_root, resolve_data_path, resolve_feature_roots
from scripts.experiments.prepare_libero_sigreg import SUITES, prepare
from scripts.experiments.run_libero_sigreg import Queue


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def nas(tmp_path, monkeypatch):
    root = tmp_path / "绿联 NAS 数据"
    root.mkdir()
    monkeypatch.setenv("MINIWAM_DATA_ROOT", str(root))
    return root


def test_legacy_and_explicit_paths_keep_their_meaning(tmp_path, monkeypatch):
    monkeypatch.delenv("MINIWAM_DATA_ROOT", raising=False)
    assert data_root() == Path("datasets")
    assert resolve_data_path("datasets/features") == Path("datasets/features")
    monkeypatch.setenv("MINIWAM_DATA_ROOT", str(tmp_path / "nas"))
    assert resolve_data_path(tmp_path / "explicit") == tmp_path / "explicit"
    assert resolve_data_path("custom/features") == Path("custom/features")
    assert resolve_data_path("datasets/features") == tmp_path / "nas/features"


@pytest.mark.parametrize("value", ["", "relative/nas", "/nas/with,comma"])
def test_invalid_root_fails_instead_of_falling_back(value, monkeypatch):
    monkeypatch.setenv("MINIWAM_DATA_ROOT", value)
    with pytest.raises(ValueError, match="MINIWAM_DATA_ROOT"):
        data_path("LIBERO-fastwam")


@pytest.mark.parametrize("benchmark", ["libero", "robotwin"])
@pytest.mark.parametrize("stage", [1, 2])
def test_stage_configs_resolve_all_feature_roots_on_nas(nas, benchmark, stage):
    cfg = load_training_config(ROOT / "configs/base.yaml", ROOT / f"configs/{benchmark}/stage{stage}.yaml")
    roots = cfg["data"]["features_root"].split(",")
    assert all(Path(path).is_relative_to(nas) for path in roots)
    assert all("dino_features_" in path for path in roots)
    assert len(roots) == (4 if benchmark == "libero" else 1)


def test_multi_suite_roots_preserve_spaces_and_absolute_overrides(nas, tmp_path):
    explicit = tmp_path / "explicit cache"
    result = resolve_feature_roots(f"datasets/features/one, {explicit},datasets/features/two")
    assert result.split(",") == [str(nas / "features/one"), str(explicit), str(nas / "features/two")]


def make_raw(root):
    for suite in SUITES:
        meta = root / suite / "meta"
        meta.mkdir(parents=True)
        records = [{"episode_index": i, "tasks": ["instruction"], "length": i + 2} for i in range(4)]
        (meta / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))


def test_protocol_and_preprocessing_use_the_same_nas_paths(nas, tmp_path, monkeypatch):
    raw = data_path("LIBERO-fastwam")
    features = data_path("dino_features_libero_sigreg_split42")
    make_raw(raw)
    output = tmp_path / "experiment"
    report = prepare(raw, output / "protocol", features)
    assert report["data_root"] == str(raw)
    shared = yaml.safe_load((output / "protocol/shared.yaml").read_text())
    assert shared["data"]["features_root"].split(",") == [str(features / suite) for suite in SUITES]
    for suite in SUITES:
        for index in range(4):
            episode = features / suite / f"episode_{index:06d}"
            episode.mkdir(parents=True)
            (episode / "metadata.json").write_text(json.dumps({"num_raw_frames": index + 2}))
            for name in ("dino_features.pt", "text_embed.pt", "actions.npy", "proprio.npy"):
                (episode / name).touch()

    # CWD contains no datasets: any surviving hardcoded input path will fail.
    monkeypatch.chdir(tmp_path)
    queue = Queue(output)
    queue.verify_assets = lambda wait: None
    commands = []
    queue.command = lambda name, command, **kwargs: commands.append(command)
    queue.parallel = lambda tasks: commands.extend(command for name, command, env in tasks)
    queue.preprocess()
    for command in commands:
        if "--data_root" in command:
            assert command[command.index("--data_root") + 1] == str(raw)
            assert Path(command[command.index("--output_dir") + 1]).is_relative_to(features)
        else:
            assert command[command.index("--features_root") + 1] == str(features)
    assert json.loads((output / "status.json").read_text())["state"] == "cache_ready"

    # A different raw location is rejected even with identical data and cache.
    other_raw = nas / "another-raw-copy"
    make_raw(other_raw)
    with pytest.raises(ValueError, match="locked experiment artifact"):
        prepare(other_raw, output / "protocol", features)


def test_standalone_prepare_uses_root_from_environment(nas, tmp_path):
    make_raw(nas / "LIBERO-fastwam")
    output = tmp_path / "protocol"
    subprocess.run([sys.executable, str(ROOT / "scripts/experiments/prepare_libero_sigreg.py"),
                    "--output_dir", str(output)], cwd=tmp_path, check=True, capture_output=True)
    report = json.loads((output / "protocol.json").read_text())
    assert report["data_root"] == str(nas / "LIBERO-fastwam")
    assert report["features_root"] == str(nas / "dino_features_libero_sigreg_split42")


def test_path_cli_runs_without_model_or_site_packages(nas, tmp_path):
    result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/data_paths.py")],
                            cwd=tmp_path, check=True, capture_output=True, text=True)
    assert result.stdout.strip() == str(nas)


def test_libero_shell_wrapper_routes_every_suite_and_text_cache_to_nas(nas, tmp_path, monkeypatch):
    wrapper = tmp_path / "record python"
    log = tmp_path / "commands.jsonl"
    wrapper.write_text("#!/usr/bin/env python3\n" + """import json, os, subprocess, sys
if sys.argv[1] == 'scripts/data_paths.py':
    subprocess.run([sys.executable, *sys.argv[1:]], check=True)
else:
    with open(os.environ['COMMAND_LOG'], 'a') as output:
        output.write(json.dumps(sys.argv[1:]) + '\\n')
""")
    wrapper.chmod(0o755)
    monkeypatch.setenv("PYTHON", str(wrapper))
    monkeypatch.setenv("COMMAND_LOG", str(log))
    monkeypatch.setenv("VISUAL_ENCODER", "dinov3")
    monkeypatch.delenv("FEATURES_ROOT", raising=False)
    monkeypatch.delenv("LIBERO_DATA_ROOT", raising=False)
    subprocess.run(["bash", str(ROOT / "scripts/data/extract_dataset_feature_libero.sh")], check=True)
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(commands) == 5
    for command in commands[:4]:
        assert command[command.index("--data_root") + 1] == str(nas / "LIBERO-fastwam")
        assert Path(command[command.index("--output_dir") + 1]).is_relative_to(
            nas / "dino_features_libero_aggregate_minmax")
    assert commands[-1][-1] == str(nas / "dino_features_libero_aggregate_minmax")


def test_robotwin_download_and_archive_cache_are_both_on_nas(nas, tmp_path, monkeypatch):
    checkout = tmp_path / "RoboTwin"
    download = checkout / "scripts/download_xpolicylab_data.sh"
    download.parent.mkdir(parents=True)
    download.write_text("#!/bin/bash\n" + """python3 - <<'PY'
import json, os
print(json.dumps({name: os.environ[name] for name in ('ROBOTWIN_DATA_ROOT', 'HF_ARCHIVE_CACHE')}))
PY
""")
    download.chmod(0o755)
    monkeypatch.setenv("ROBOTWIN_ROOT", str(checkout))
    for name in ("ROBOTWIN_DATA_ROOT", "XPOLICYLAB_DATA_ROOT", "HF_ARCHIVE_CACHE"):
        monkeypatch.delenv(name, raising=False)
    result = subprocess.run(["bash", str(ROOT / "scripts/download_robotwin_training_data.sh")],
                            check=True, capture_output=True, text=True)
    paths = json.loads(result.stdout)
    assert paths["ROBOTWIN_DATA_ROOT"] == str(nas / "RoboTwin")
    assert paths["HF_ARCHIVE_CACHE"] == str(nas / "RoboTwin/download_cache")
