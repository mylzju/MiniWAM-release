import hashlib
import json

import pytest

from miniwam.dataset import MiniWAMPreextractedDataset
from miniwam.episode_split import load_episode_manifest, select_episode_records
from test_miniwam_multiview import make_runtime_protocol, write_episode


def test_train_and_heldout_feature_frames_are_disjoint(tmp_path):
    root = tmp_path / "features"
    for index in range(3):
        write_episode(root, index, main_slots=1, wrist_slots=0)
    manifests = []
    for label, indices in (("train", [0, 2]), ("heldout", [1])):
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps({"test": indices}))
        dataset = MiniWAMPreextractedDataset(
            str(root), make_runtime_protocol(), episode_manifest=str(path),
            episode_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        assert len(dataset) == len(indices) * 3
        assert [int(episode.name.removeprefix("episode_")) for episode in dataset.episode_dirs] == indices
        manifests.append(set(dataset.episode_dirs))
    assert not manifests[0] & manifests[1]
    path.write_text(json.dumps({"test": [4]}))
    with pytest.raises(FileNotFoundError, match="Missing selected"):
        MiniWAMPreextractedDataset(str(root), make_runtime_protocol(), episode_manifest=str(path))


def test_manifest_requires_exact_unique_indices_and_digest(tmp_path):
    path = tmp_path / "split.json"
    path.write_text('{"test": [0, 0]}')
    with pytest.raises(ValueError, match="Duplicate"):
        load_episode_manifest(path)
    path.write_text('{"test": [true]}')
    with pytest.raises(ValueError, match="integers"):
        load_episode_manifest(path)
    path.write_text('{"test": [0]}')
    with pytest.raises(ValueError, match="SHA256"):
        load_episode_manifest(path, expected_sha256="changed")
    selected, _ = load_episode_manifest(path)
    assert select_episode_records([{"episode_index": 0}, {"episode_index": 1}], "test", selected) == [{"episode_index": 0}]
    with pytest.raises(ValueError, match="Missing requested"):
        select_episode_records([], "test", selected)
