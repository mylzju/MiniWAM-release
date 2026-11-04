import json
import os
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from scripts.analysis.analyze_stage1_representations import calibrate_gradients, evaluate_id, summarize_slot
from test_miniwam_ablation_axes import batch_for, build_tiny, tiny_cfg


def test_diagnostics_use_cross_transition_variance_and_report_bad_posteriors():
    mu = torch.randn(20, 4, 6)
    summary = summarize_slot(mu)
    assert 0 < summary["effective_rank"] <= summary["rank_upper_bound"] + 1e-4
    assert summary["rank_upper_bound"] == 19
    collapsed = summarize_slot(mu[:1].repeat(20, 1, 1))
    assert collapsed["effective_rank"] == pytest.approx(0, abs=1e-5)
    assert collapsed["sigreg_loss"] > summary["sigreg_loss"]
    bad = mu.clone()
    bad[0, 0, 0] = float("nan")
    lv = torch.zeros(20, 24)
    lv[0, 0] = 1000
    report = summarize_slot(bad, lv)
    assert report["nonfinite_mu_rows"] == 1
    assert report["nonfinite_posterior_std_elements"] == 1
    json.dumps(report, allow_nan=False)


def test_calibration_restores_rng_weights_and_progress():
    cfg = tiny_cfg()
    cfg["model"]["latent_encoder"]["use_kl"] = True
    cfg["model"]["loss"]["lambda_sigreg"] = 0.01
    cfg["model"]["sigreg"] = {"num_slices": 8}
    model, _ = build_tiny(cfg)
    batch = batch_for(cfg)
    global_rng, projection_rng = torch.get_rng_state(), model.sigreg.get_rng_state()
    before = {name: p.clone() for name, p in model.state_dict().items()}
    report = calibrate_gradients(model, batch)
    assert report["id_gradient_norm"] > 0 and report["sigreg_gradient_norm"] > 0
    assert report["suggested_lambda_sigreg"] > 0 and len(report["candidate_weights"]) == 3
    assert model._global_step == 0 and model.lambda_sigreg == 0.01
    assert torch.equal(global_rng, torch.get_rng_state())
    assert torch.equal(projection_rng, model.sigreg.get_rng_state())
    assert all(p.grad is None for p in model.parameters())
    for name, p in model.state_dict().items():
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)


def test_id_evaluation_pairs_noise_and_restores_training_settings():
    cfg = tiny_cfg()
    cfg["model"]["latent_encoder"]["use_kl"] = True
    cfg["model"]["loss"]["lambda_sigreg"] = 0.01
    model, _ = build_tiny(cfg)
    batch = batch_for(cfg)
    rng = torch.get_rng_state()
    projection_rng = model.sigreg.get_rng_state()
    result = evaluate_id(model, [batch], [101, 102])
    assert result == evaluate_id(model, [batch], [101, 102])
    assert len(result["per_noise_seed"]) == 2
    assert model._global_step == 0 and model.lambda_sigreg == 0.01
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(projection_rng, model.sigreg.get_rng_state())


def test_analysis_cli_on_cached_features(tmp_path):
    features = tmp_path / "features"
    features.mkdir()
    stats = {"test": {"stats_id": "test_stats", "action": {
        "normalization": "minmax_to_minus1_plus1", "min": [-1.] * 7, "max": [1.] * 7,
    }, "absolute_action_mask": [False] * 6 + [True]}}
    with (features / "dataset_statistics.pkl").open("wb") as handle:
        pickle.dump(stats, handle)
    episode = features / "episode_000000"
    episode.mkdir()
    g = torch.Generator().manual_seed(5)
    torch.save({
        "dino_main": torch.randn(3, 1, 196, 768, generator=g).bfloat16(),
        "dino_wrist": torch.randn(3, 1, 196, 768, generator=g).bfloat16(),
        "main_present": torch.ones(3, 1, dtype=torch.bool),
        "wrist_present": torch.ones(3, 1, dtype=torch.bool),
    }, episode / "dino_features.pt")
    torch.save({"context": torch.randn(4, 16), "mask": torch.ones(4, dtype=torch.bool)}, episode / "text_embed.pt")
    np.save(episode / "actions.npy", np.zeros((3, 7), dtype=np.float32))
    (episode / "metadata.json").write_text(json.dumps({
        "dataset_name": "test", "num_raw_frames": 3, "num_main_slots": 1,
        "num_wrist_slots": 1, "stats_id": "test_stats", "dinov3": {
            "cache_layout": "per_frame_v1", "architecture": "vitb16", "pretraining": "lvd1689m",
            "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m", "revision": "synthetic-revision",
            "resolved_revision": "synthetic-revision", "input_size": 224, "feature_type": "patch_tokens",
            "feature_normalization": "final_layernorm", "l2_normalize": False, "include_cls": False,
            "include_registers": False, "feature_layers": "final", "compute_dtype": "float32",
            "boundary_dtype": "bfloat16", "storage_dtype": "bfloat16",
        },
        "visual_window_contract": {"storage": "per_frame_v1", "future_horizon_source": "runtime_protocol",
                                   "endpoint_padding": "repeat_last_frame"},
    }))
    cfg = tiny_cfg("dinov3")
    cfg["data"]["features_root"] = str(features)
    cfg["model"]["loss"]["lambda_sigreg"] = 0.01
    cfg["model"]["sigreg"] = {"num_slices": 8}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    output = tmp_path / "report.json"
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([
        sys.executable, str(root / "scripts/analysis/analyze_stage1_representations.py"),
        "--config", str(config_path), "--base_config", str(config_path),
        "--max_samples", "3", "--batch_size", "3", "--calibrate", "--output", str(output),
    ], capture_output=True, text=True, timeout=60, env={**os.environ, "OMP_NUM_THREADS": "1"})
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert set(report["slots"]) == {"main_0", "wrist_0"}
    assert report["protocol"]["split_label"] == "training_diagnostic"
    assert report["calibration_first_microbatch"]["suggested_lambda_sigreg"] > 0
    assert len(json.loads(output.with_suffix(".indices.json").read_text())) == 3
    assert output.with_suffix(".csv").is_file()
