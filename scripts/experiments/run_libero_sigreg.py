#!/usr/bin/env python
"""Logged, restartable LIBERO A/B/C/D experiment queue (seed 42).

Run in the miniwam environment from the repository root. A successful command
gets a completion marker; failed commands stop the queue. Restarting skips only
commands with matching arguments and successful completion markers. Interrupted
training restarts that run from its seed rather than resuming a partial RNG state.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.prepare_libero_sigreg import SUITES, prepare, write_locked
from scripts.data_paths import data_path, resolve_data_path

ARMS = {"A": "id_recon_kl", "B": "id_recon_no_kl", "C": "id_sigreg", "D": "id_only"}
UMT5_SHA256 = "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Queue:
    def __init__(self, output, *, wandb_project=None, data_root=None, features_root=None):
        self.output = output
        self.wandb_project = wandb_project
        self.protocol = output / "protocol"
        self.data = resolve_data_path(data_root if data_root is not None else data_path("LIBERO-fastwam"))
        self.features = resolve_data_path(features_root if features_root is not None
                                          else data_path("dino_features_libero_sigreg_split42"))
        self.logs = output / "logs"
        self.markers = output / "completed"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.markers.mkdir(parents=True, exist_ok=True)
        self.children = []
        self.env = {
            **os.environ, "HF_HUB_CACHE": str(ROOT / "pretrained/hf-cache"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "4", "NUMBA_CACHE_DIR": "/tmp/miniwam_numba_cache",
            "MPLCONFIGDIR": "/tmp/miniwam_matplotlib", "TOKENIZERS_PARALLELISM": "false",
            "LIBERO_CONFIG_PATH": str(ROOT / ".libero"), "PYTHONUNBUFFERED": "1",
        }
        source_files = [*ROOT.glob("miniwam/**/*.py"), ROOT / "scripts/train.py", ROOT / "scripts/data_paths.py",
                        *ROOT.glob("scripts/data/*.py"), *ROOT.glob("scripts/analysis/*.py"),
                        *ROOT.glob("scripts/experiments/*.py"), *ROOT.glob("configs/**/*.yaml")]
        snapshot = {
            "data_root": str(self.data), "features_root": str(self.features),
            "started_at": now(), "python": sys.version,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "huggingface_hub", "numpy", "protobuf", "av", "pandas")},
            "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(set(source_files))},
        }
        snapshots = self.output / "software_snapshots"
        snapshots.mkdir(exist_ok=True)
        (snapshots / f"{time.time_ns()}.json").write_text(json.dumps(snapshot, indent=2) + "\n")
        self.status("starting")

    def status(self, state, **details):
        record = {"state": state, "updated_at": now(), "pid": os.getpid(), **details}
        path = self.output / "status.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(record, indent=2) + "\n")
        temp.replace(path)
        print(json.dumps(record), flush=True)

    def command(self, name, command, *, env=None):
        return self.parallel([(name, command, env)])[0]

    def parallel(self, commands):
        running, results = [], []
        try:
            for name, command, extra_env in commands:
                fingerprint = hashlib.sha256(json.dumps({"command": command, "env": extra_env}, sort_keys=True).encode()).hexdigest()
                marker = self.markers / f"{name}.json"
                if marker.exists():
                    if json.loads(marker.read_text())["command_sha256"] != fingerprint:
                        raise ValueError(f"Changed command for completed phase {name}")
                    results.append(name)
                    continue
                log_path = self.logs / f"{name}.log"
                # A failed training retry must not append old, incompatible metrics.
                if "scripts/train.py" in command and "--output_dir" in command:
                    run_dir = Path(command[command.index("--output_dir") + 1])
                    metrics = run_dir / "metrics.jsonl"
                    if metrics.exists():
                        metrics.rename(run_dir / f"metrics.interrupted_{time.time_ns()}.jsonl")
                log = log_path.open("a", buffering=1)
                log.write(f"\n{now()} command={json.dumps(command)}\n")
                process = subprocess.Popen(command, cwd=ROOT, env={**self.env, **(extra_env or {})},
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                self.children.append(process)
                running.append((name, process, log, log_path, marker, fingerprint))
            if running:
                self.status("running", commands=[{"name": name, "pid": process.pid, "log": str(log_path)}
                            for name, process, _, log_path, _, _ in running])
            while running:
                for item in running[:]:
                    name, process, log, log_path, marker, fingerprint = item
                    code = process.poll()
                    if code is None:
                        continue
                    log.close()
                    self.children.remove(process)
                    running.remove(item)
                    if code:
                        raise RuntimeError(f"{name} exited {code}; see {log_path}")
                    marker.write_text(json.dumps({"name": name, "completed_at": now(), "command_sha256": fingerprint}, indent=2) + "\n")
                    results.append(name)
                if running:
                    time.sleep(2)
            return results
        finally:
            for _, process, log, _, _, _ in running:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                process.wait()
                log.close()
                if process in self.children:
                    self.children.remove(process)

    def verify_assets(self, wait_seconds=0):
        path = Path("pretrained/language/models_t5_umt5-xxl-enc-bf16.pth")
        deadline = time.monotonic() + wait_seconds
        if not path.is_file() and wait_seconds:
            self.status("waiting_for_text_weights", path=str(path), timeout_seconds=wait_seconds)
        while not path.is_file() and time.monotonic() < deadline:
            time.sleep(10)
        if not path.is_file():
            raise FileNotFoundError(f"UMT5 download must finish before extraction: {path}")
        manifest_path = self.protocol / "text_assets.json"
        if not manifest_path.exists():
            self.status("verifying_text_weights", path=str(path))
            digest = sha256_file(path)
            if digest != UMT5_SHA256:
                raise ValueError("Downloaded UMT5 SHA256 does not match the pinned source.")
            revision = (ROOT / "pretrained/hf-cache/models--google--umt5-xxl/refs/main").read_text().strip()
            manifest_path.write_text(json.dumps({
                "text_encoder": "Wan-AI/Wan2.1-T2V-14B", "revision": "b62153b54fce61f5908de23f6d8715e147de0daf",
                "filename": path.name, "sha256": digest, "bytes": path.stat().st_size,
                "tokenizer": "google/umt5-xxl", "tokenizer_revision": revision,
            }, indent=2) + "\n")
        self.command("environment", [sys.executable, "scripts/verify_miniwam_environment.py",
                                     "--require-cuda", "--check-libero", "--check-assets"])

    def preprocess(self, wait_assets=0):
        self.verify_assets(wait_assets)
        base = [sys.executable, "scripts/data/extract_dataset_feature_libero.py",
                "--data_root", str(self.data), "--visual_encoder", "dinov3",
                "--stats_episode_manifest", str(self.protocol / "train_episodes.json"),
                "--batch_size", "32", "--dtype", "bf16", "--device", "cuda:0"]
        # Validate the real model/decoder path before committing to all episodes.
        self.command("extract_preflight", [*base, "--data_mix", SUITES[0], "--max_episodes", "1",
                                          "--output_dir", str(self.features / SUITES[0])], env={"CUDA_VISIBLE_DEVICES": "0"})
        for suite in SUITES:
            self.parallel([
                (f"extract_{suite}_shard{rank}", [*base, "--data_mix", suite, "--num_shards", "2",
                 "--shard_id", str(rank), "--output_dir", str(self.features / suite)], {"CUDA_VISIBLE_DEVICES": str(rank)})
                for rank in range(2)
            ])
        self.command("text_cache", [sys.executable, "scripts/data/build_libero_text_feature_cache.py",
                                    "--features_root", str(self.features)])
        # Do not let a partially extracted dataset silently become the experiment.
        for suite in SUITES:
            records = [json.loads(line) for line in (self.data / suite / "meta/episodes.jsonl").read_text().splitlines()]
            for episode in records:
                directory = self.features / suite / f"episode_{episode['episode_index']:06d}"
                for filename in ("metadata.json", "dino_features.pt", "text_embed.pt", "actions.npy", "proprio.npy"):
                    if not (directory / filename).is_file():
                        raise FileNotFoundError(directory / filename)
                if json.loads((directory / "metadata.json").read_text())["num_raw_frames"] != episode["length"]:
                    raise ValueError(f"Incomplete frame extraction: {directory}")
        self.status("cache_ready", episodes=1712, frames=277713)

    def train(self, label, arm, stage, steps, *, stage1=None, extra_overlay=None):
        directory = self.output / "runs" / label
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                   "scripts/train.py", "--fsdp", "--config", f"configs/libero/stage{stage}.yaml",
                   f"configs/ablations/objective/{ARMS[arm]}.yaml", str(self.protocol / "shared.yaml")]
        if extra_overlay:
            command.append(str(extra_overlay))
        save_interval = 5420 if stage == 2 and steps == 54200 else steps
        command.extend(["--max_steps", str(steps), "--save_interval", str(save_interval),
                        "--log_interval", "10", "--seed", "42", "--output_dir", str(directory)])
        if stage1:
            command.extend(["--stage1_checkpoint", str(stage1)])
        if self.wandb_project is not None:
            command.extend(["--wandb", "--wandb_project", self.wandb_project,
                            "--wandb_run_name", label])
        self.command(label, command, env={"CUDA_VISIBLE_DEVICES": "0,1"})
        checkpoint = directory / "latest"
        if json.loads((checkpoint / "metadata.json").read_text())["global_step"] != steps:
            raise ValueError(f"Incomplete checkpoint: {checkpoint}")
        records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        if not records or records[-1]["step"] != steps:
            raise ValueError(f"Missing final metrics: {directory}")
        for record in records:
            if any(isinstance(value, float) and not math.isfinite(value) for value in record.values()):
                raise ValueError(f"Nonfinite training metrics: {directory}")
            if stage == 1:
                for key in ((["weighted_kl_loss"] if arm != "A" else []) +
                            (["weighted_feature_reconstruction_loss"] if arm in "CD" else []) +
                            (["weighted_sigreg_loss"] if arm != "C" else [])):
                    if record[key] != 0:
                        raise ValueError(f"Ablation objective mismatch: {directory} {key}")
        return checkpoint

    def diagnose(self, label, checkpoint=None, *, calibrate=False):
        output = self.output / "diagnostics" / f"{label}.json"
        command = [sys.executable, "scripts/analysis/analyze_stage1_representations.py",
                   "--device", "cuda:0", "--autocast_bf16", "--output", str(output)]
        if calibrate:
            command.extend(["--config", "configs/libero/stage1.yaml", "configs/ablations/objective/id_sigreg.yaml",
                            str(self.protocol / "shared.yaml"), "--max_samples", "256", "--batch_size", "64", "--calibrate"])
        else:
            command.extend(["--checkpoint", str(checkpoint), "--episode_manifest", str(self.protocol / "heldout_episodes.json"),
                            "--split_label", "heldout", "--max_samples", "256", "--batch_size", "16",
                            "--id_eval_seeds", "101", "102", "103"])
        self.command(label, command, env={"CUDA_VISIBLE_DEVICES": "0"})
        return json.loads(output.read_text())

    def smoke(self):
        for arm in ARMS:
            checkpoint = self.train(f"smoke_{arm}_stage1", arm, 1, 100)
            self.diagnose(f"smoke_{arm}_heldout", checkpoint)
            self.train(f"smoke_{arm}_stage2", arm, 2, 100, stage1=checkpoint)
        self.status("smoke_complete", note="Real cached data; separate from formal training.")

    def calibrate(self):
        report = self.diagnose("initial_gradient_calibration", calibrate=True)
        weights = report["calibration_first_microbatch"]["candidate_weights"]
        if len(weights) != 3 or any(not math.isfinite(weight) or weight <= 0 for weight in weights):
            raise ValueError("Invalid gradient calibration; inspect diagnostics before continuing.")
        candidates = []
        for index, weight in enumerate(weights):
            overlay = self.protocol / f"candidate_{index}.yaml"
            write_locked(overlay, yaml.safe_dump({"stage1": {"loss": {"lambda_sigreg": weight}}}))
            checkpoint = self.train(f"pilot_C_{index}_stage1", "C", 1, 500, extra_overlay=overlay)
            diagnostics = self.diagnose(f"pilot_C_{index}_heldout", checkpoint)
            finite = all(not value for slot in diagnostics["slots"].values()
                         for key, value in slot.items() if key.startswith("nonfinite_"))
            noncollapsed = bool(diagnostics["slots"]) and all(
                slot.get("covariance_trace", 0) > 1e-8 and slot.get("effective_rank", 0) > 1.01
                for slot in diagnostics["slots"].values())
            score = diagnostics["id_evaluation"]["mean_id_loss"]
            candidates.append({"index": index, "lambda_sigreg": weight, "heldout_id_loss": score,
                               "eligible": finite and noncollapsed and math.isfinite(score)})
        eligible = [candidate for candidate in candidates if candidate["eligible"]]
        if not eligible:
            raise ValueError("All SIGReg pilots failed finite/noncollapse checks; formal queue stopped.")
        winner = min(eligible, key=lambda candidate: (candidate["heldout_id_loss"], candidate["index"]))
        locked = self.protocol / "selected_sigreg.yaml"
        write_locked(locked, yaml.safe_dump({"stage1": {"loss": {"lambda_sigreg": winner["lambda_sigreg"]}}}))
        write_locked(self.protocol / "lambda_selection.json", json.dumps({
            "candidates": candidates, "selected": winner,
            "selection_rule": "Minimum heldout ID loss over 256 fixed frames and noise seeds 101/102/103, after finite and noncollapse checks (covariance trace > 1e-8, effective rank > 1.01).",
            "formal_test_used": False,
        }, indent=2) + "\n")
        self.status("lambda_locked", **winner)
        return locked

    def formal(self, selected):
        for arm in ARMS:
            overlay = selected if arm == "C" else None
            checkpoint = self.train(f"formal_{arm}_seed42_stage1", arm, 1, 16260, extra_overlay=overlay)
            self.diagnose(f"formal_{arm}_heldout", checkpoint)
            self.train(f"formal_{arm}_seed42_stage2", arm, 2, 54200, stage1=checkpoint, extra_overlay=overlay)
        self.status("training_complete", pending="LIBERO and LIBERO-Plus rollout success-rate evaluation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/libero_sigreg_seed42"))
    parser.add_argument("--data_root", type=Path, default=data_path("LIBERO-fastwam"),
                        help="LIBERO raw datasets (default: MINIWAM_DATA_ROOT/LIBERO-fastwam)")
    parser.add_argument("--features_root", type=Path, default=data_path("dino_features_libero_sigreg_split42"),
                        help="Feature cache (default: MINIWAM_DATA_ROOT/dino_features_libero_sigreg_split42)")
    parser.add_argument("--through", choices=["preprocess", "smoke", "calibration", "training"], default="training")
    parser.add_argument("--wait_assets", type=int, default=0, help="Seconds to wait for an already running UMT5 download")
    parser.add_argument("--wandb", action="store_true", help="Log each training phase to a separate W&B run")
    parser.add_argument("--wandb_project", default="miniwam-sigreg", help="W&B project used with --wandb")
    args = parser.parse_args()
    os.chdir(ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(args.data_root, args.output_dir / "protocol", args.features_root)
        queue = Queue(args.output_dir, wandb_project=args.wandb_project if args.wandb else None,
                      data_root=args.data_root, features_root=args.features_root)
        def stop(signum, frame):
            raise KeyboardInterrupt(f"Received {signal.Signals(signum).name}")
        signal.signal(signal.SIGTERM, stop)
        (args.output_dir / "queue.pid").write_text(str(os.getpid()) + "\n")
        try:
            queue.preprocess(args.wait_assets)
            if args.through == "preprocess":
                return
            queue.smoke()
            if args.through == "smoke":
                return
            selected = queue.calibrate()
            if args.through == "calibration":
                return
            queue.formal(selected)
        except BaseException as exc:
            queue.status("failed", error=str(exc), error_type=type(exc).__name__)
            raise


if __name__ == "__main__":
    main()
