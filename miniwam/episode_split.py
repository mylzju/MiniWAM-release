"""Explicit episode selection shared by extraction, training and diagnostics."""

import hashlib
import json
from pathlib import Path


def load_episode_manifest(path, *, expected_sha256=None):
    """Read {dataset_name: [episode_index, ...]}, rejecting ambiguous selections."""
    if path is None:
        return None, None
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Episode manifest SHA256 mismatch: {path}")
    manifest = json.loads(payload)
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Episode manifest must be a nonempty dataset-to-indices mapping.")
    selected = {}
    for name, indices in manifest.items():
        if not isinstance(name, str) or not name or not isinstance(indices, list) or not indices:
            raise ValueError(f"Invalid episode selection for {name!r}.")
        if any(type(index) is not int or index < 0 for index in indices):
            raise ValueError(f"Episode indices must be nonnegative integers: {name!r}")
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate episode indices in {name!r}.")
        selected[name] = set(indices)
    return selected, digest


def select_episode_records(episodes, dataset_name, manifest):
    """Select raw LeRobot records, requiring every requested episode to exist."""
    if manifest is None:
        return episodes
    if dataset_name not in manifest:
        raise ValueError(f"Dataset {dataset_name!r} is missing from the episode manifest.")
    wanted = manifest[dataset_name]
    found = {episode["episode_index"] for episode in episodes}
    missing = wanted - found
    if missing:
        raise ValueError(f"Missing requested episodes for {dataset_name!r}: {sorted(missing)}")
    return [episode for episode in episodes if episode["episode_index"] in wanted]
