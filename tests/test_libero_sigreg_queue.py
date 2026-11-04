import json

import pytest
import yaml

from miniwam.config import load_training_config
from scripts.experiments.prepare_libero_sigreg import SUITES, prepare
from scripts.experiments.run_libero_sigreg import Queue
from scripts.train import resolve_per_device_batch_size


def test_locked_task_stratified_split_and_two_stage_batch(tmp_path):
    raw = tmp_path / "raw"
    for suite in SUITES:
        meta = raw / suite / "meta"
        meta.mkdir(parents=True)
        records = [{"episode_index": i, "tasks": [f"task_{i // 4}"], "length": i + 2} for i in range(8)]
        (meta / "episodes.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    protocol = tmp_path / "protocol"
    report = prepare(raw, protocol, tmp_path / "features")
    assert prepare(raw, protocol, tmp_path / "features") == report
    train = json.loads((protocol / "train_episodes.json").read_text())
    heldout = json.loads((protocol / "heldout_episodes.json").read_text())
    for suite in SUITES:
        assert not set(train[suite]) & set(heldout[suite])
        assert sorted(train[suite] + heldout[suite]) == list(range(8))
        for lower in (0, 4):
            assert sum(lower <= i < lower + 4 for i in heldout[suite]) == 2
    for stage in (1, 2):
        cfg = load_training_config("configs/base.yaml", [f"configs/libero/stage{stage}.yaml",
            "configs/ablations/objective/id_sigreg.yaml", str(protocol / "shared.yaml")])
        assert resolve_per_device_batch_size(cfg["training"], 2) == 32
        assert cfg["training"]["gradient_accumulation_steps"] == 4
        assert cfg["data"]["episode_manifest_sha256"] == report["train_manifest_sha256"]
        if stage == 2:
            assert cfg["checkpoint"]["policy_init"] == "fresh"
    selection = protocol / "selected_sigreg.yaml"
    selection.write_text(yaml.safe_dump({"stage1": {"loss": {"lambda_sigreg": 0.003}}}))
    for stage in (1, 2):
        cfg = load_training_config("configs/base.yaml", [f"configs/libero/stage{stage}.yaml",
            "configs/ablations/objective/id_sigreg.yaml", str(selection)])
        assert cfg["loss"]["lambda_sigreg"] == (0.003 if stage == 1 else 0.0)
    with pytest.raises(ValueError, match="Refusing to replace"):
        prepare(raw, protocol, tmp_path / "features", seed=43)


@pytest.mark.parametrize("project", [None, "migration-test"])
def test_queue_wandb_is_optional_and_names_each_training_phase(tmp_path, project):
    queue = Queue.__new__(Queue)
    queue.output = tmp_path
    queue.protocol = tmp_path / "protocol"
    queue.wandb_project = project
    commands = []

    def finish_training(name, command, *, env):
        commands.append(command)
        checkpoint = tmp_path / "runs" / name / "latest"
        checkpoint.mkdir(parents=True)
        (checkpoint / "metadata.json").write_text(json.dumps({"global_step": 100}))
        (checkpoint.parent / "metrics.jsonl").write_text(json.dumps({
            "step": 100, "weighted_kl_loss": 0.0,
            "weighted_feature_reconstruction_loss": 0.0,
        }) + "\n")

    queue.command = finish_training
    for stage in (1, 2):
        label = f"smoke_C_stage{stage}"
        queue.train(label, "C", stage, 100)
        command = commands[-1]
        if project is None:
            assert "--wandb" not in command
            assert "--wandb_project" not in command
            assert "--wandb_run_name" not in command
        else:
            assert "--wandb" in command
            assert command[command.index("--wandb_project") + 1] == project
            assert command[command.index("--wandb_run_name") + 1] == label
