#!/usr/bin/env python
"""Lock task-stratified episode splits and shared settings before ablation runs."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data_paths import data_path, resolve_data_path


SUITES = (
    "libero_10_no_noops_lerobot", "libero_spatial_no_noops_lerobot",
    "libero_object_no_noops_lerobot", "libero_goal_no_noops_lerobot",
)


def write_locked(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != text:
        raise ValueError(f"Refusing to replace a different locked experiment artifact: {path}")
    path.write_text(text)


def prepare(data_root, output_dir, features_root, seed=42, heldout_per_task=2):
    data_root = resolve_data_path(data_root)
    features_root = resolve_data_path(features_root)
    train, heldout, summary = {}, {}, {}
    rng = random.Random(seed)
    for suite in SUITES:
        episodes = [json.loads(line) for line in (data_root / suite / "meta/episodes.jsonl").read_text().splitlines()]
        by_task = defaultdict(list)
        for episode in episodes:
            if len(episode["tasks"]) != 1:
                raise ValueError(f"Expected one exact instruction per episode: {episode}")
            by_task[episode["tasks"][0]].append(episode)
        train[suite], heldout[suite], counts = [], [], {}
        for task, records in sorted(by_task.items()):
            records.sort(key=lambda record: record["episode_index"])
            if len(records) <= heldout_per_task:
                raise ValueError(f"Too few demonstrations for task {task!r}")
            rng.shuffle(records)
            selected = {record["episode_index"] for record in records[:heldout_per_task]}
            train[suite].extend(record["episode_index"] for record in records if record["episode_index"] not in selected)
            heldout[suite].extend(selected)
            counts[task] = {"train_episodes": len(records) - heldout_per_task, "heldout_episodes": heldout_per_task}
        train[suite].sort()
        heldout[suite].sort()
        summary[suite] = {
            "train_episodes": len(train[suite]), "heldout_episodes": len(heldout[suite]),
            "train_frames": sum(record["length"] for record in episodes if record["episode_index"] in train[suite]),
            "heldout_frames": sum(record["length"] for record in episodes if record["episode_index"] in heldout[suite]),
            "tasks": counts,
        }
    for name, records in (("train", train), ("heldout", heldout)):
        write_locked(output_dir / f"{name}_episodes.json", json.dumps(records, indent=2, sort_keys=True) + "\n")
    train_path = output_dir / "train_episodes.json"
    digest = hashlib.sha256(train_path.read_bytes()).hexdigest()
    config = {
        "data": {
            "features_root": ",".join(str(features_root / suite) for suite in SUITES),
            "episode_manifest": str(train_path), "episode_manifest_sha256": digest,
            "num_workers": 4,
        },
        "training": {"per_device_batch_size": 32, "gradient_accumulation_steps": 4, "seed": seed},
        "logging": {"use_wandb": False},
    }
    write_locked(output_dir / "shared.yaml", yaml.safe_dump(config, sort_keys=False))
    report = {
        "data_root": str(data_root), "features_root": str(features_root),
        "seed": seed, "heldout_per_task": heldout_per_task, "suites": summary,
        "train_manifest_sha256": digest,
        "heldout_manifest_sha256": hashlib.sha256((output_dir / "heldout_episodes.json").read_bytes()).hexdigest(),
        "world_size": 2, "per_device_batch_size": 32, "gradient_accumulation_steps": 4,
        "optimizer_batch_size": 256, "sigreg_statistical_batch_size": 64,
        "normalization": "aggregate_minmax_training_episodes_only",
        "main_seed": seed, "stage1_updates": 16260, "stage2_updates": 54200,
        "stage2_policy_init": "fresh", "stage1_sampling": "enabled_in_all_four_arms",
        "comparison": "B_to_C_is_whole_recipe; D_to_C_is_SIGReg_contribution",
        "pilot_selection": "Finite training and posterior statistics required; covariance trace > 1e-8 and effective rank > 1.01 in each slot; select lowest heldout ID loss over 256 fixed frames and noise seeds 101/102/103 among the three calibrated weights. Formal rollout tests do not select lambda.",
        "protocol_difference_from_original": "Two demonstrations per task excluded from both training stages and normalization; fresh policy initialization in all Stage 2 runs.",
    }
    write_locked(output_dir / "protocol.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=data_path("LIBERO-fastwam"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/libero_sigreg_seed42/protocol"))
    parser.add_argument("--features_root", type=Path, default=data_path("dino_features_libero_sigreg_split42"))
    args = parser.parse_args()
    result = prepare(args.data_root, args.output_dir, args.features_root)
    print(json.dumps({"protocol": str(args.output_dir), "suites": {
        name: {key: value for key, value in counts.items() if key != "tasks"}
        for name, counts in result["suites"].items()
    }}, indent=2))
