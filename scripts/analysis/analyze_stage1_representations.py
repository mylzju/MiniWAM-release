#!/usr/bin/env python
"""Fixed-sample Stage-1 mu diagnostics and ID/SIGReg gradient calibration.

Use a checkpoint or a config stack (for calibration before training). Samples
are explicitly recorded; the default is a training diagnostic, not validation.
No optimizer step is taken. Run --help for the input/output options.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from miniwam.checkpointing import (
    build_cpu_model_with_seed, build_model_protocol, load_checkpoint, validate_model_protocol,
)
from miniwam.config import load_training_config
from scripts.data_paths import resolve_feature_roots
from miniwam.dataset import MiniWAMCollator, MiniWAMPreextractedDataset
from miniwam.factory import build_miniwam_model
from miniwam.model import _flatten_role_slot_features
from miniwam.models.sigreg import REFERENCE_URL, SIGReg
from miniwam.protocol import resolve_miniwam_protocol


def _quantiles(x):
    x = x.float()
    return [float(v) for v in torch.quantile(x, x.new_tensor([0., .1, .5, .9, 1.]))]


def summarize_slot(mu: torch.Tensor, logvar: torch.Tensor | None = None, *, seed=42) -> dict:
    """mu is [valid transitions, tokens, channels], never tokens as samples."""
    n, tokens, channels = mu.shape
    flat = mu.float().reshape(n, tokens * channels)
    finite = torch.isfinite(flat).all(dim=1)
    result = {
        "valid_count": n, "num_tokens": tokens, "channels": channels,
        "flattened_dim": tokens * channels,
        "nonfinite_mu_elements": int((~torch.isfinite(flat)).sum()),
        "nonfinite_mu_rows": int((~finite).sum()),
    }
    clean = flat[finite]
    if len(clean):
        result.update(mu_mean=float(clean.mean()), mu_rms=float(clean.square().mean().sqrt()))
    if len(clean) >= 2:
        # FP64 centering keeps a repeated FP32 vector exactly collapsed rather
        # than assigning a spurious rank to FP32 mean-rounding residuals.
        centered = clean.double() - clean.double().mean(0)
        variance = centered.square().sum(0) / (len(clean) - 1)
        # A Gram matrix avoids a potentially enormous [S*D,S*D] covariance.
        eigenvalues = torch.linalg.eigvalsh(centered @ centered.T / (len(clean) - 1)).clamp_min(0)
        total = eigenvalues.sum()
        probabilities = eigenvalues / total.clamp_min(torch.finfo(torch.float64).tiny)
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
        result.update(
            std_quantiles_0_10_50_90_100=_quantiles(variance.sqrt()),
            token_variance_quantiles_0_10_50_90_100=_quantiles(variance.reshape(tokens, channels).mean(-1)),
            covariance_trace=float(total),
            covariance_top16_eigenvalues=[float(v) for v in eigenvalues.flip(0)[:16]],
            effective_rank=float(entropy.exp()) if total > 0 else 0.0,
            rank_upper_bound=min(len(clean) - 1, flat.shape[1]),
            mean_pairwise_squared_distance=float(2 * variance.sum()),
            sigreg_loss=float(SIGReg(seed=seed)(clean, synchronize=False)),
        )
    if logvar is not None:
        logvar = logvar.float()
        result["nonfinite_logvar_elements"] = int((~torch.isfinite(logvar)).sum())
        std = (0.5 * logvar).exp()
        result["nonfinite_posterior_std_elements"] = int((~torch.isfinite(std)).sum())
        for name, values in (("logvar", logvar), ("posterior_std", std)):
            values = values[torch.isfinite(values)]
            if values.numel():
                result[f"{name}_quantiles_0_10_50_90_100"] = _quantiles(values)
    return result


def _encoder_gradient_norm(loss, parameters):
    if not loss.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return math.sqrt(sum(float(g.float().square().sum()) for g in gradients if g is not None))


def calibrate_gradients(model, batch: dict, *, seed=42, target_ratio=0.1) -> dict:
    """Measure raw ID/SIGReg gradients on shared encoder weights, with no step.

    This diagnostic intentionally runs two forwards so the ID and regularizer
    gradients can be measured separately without retaining a training graph.
    It restores all loss settings, counters, module RNG and global RNG state.
    """
    if model.stage != 1 or model.stage1_objective != "inverse_dynamics":
        raise ValueError("Gradient calibration requires inverse_dynamics Stage 1.")
    names = ("lambda_action", "lambda_latent", "lambda_rec", "kl_beta", "lambda_sigreg", "_global_step")
    old = {name: getattr(model, name) for name in names}
    original_sigreg = model.sigreg
    rng = original_sigreg.get_rng_state() if original_sigreg is not None else None
    parameters = [p for encoder in (model.latent_encoder, model.wrist_latent_encoder)
                  for name, p in encoder.named_parameters() if p.requires_grad and not name.startswith("logvar_head.")]
    devices = [model.device.index] if model.device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            model.lambda_action, model.lambda_latent, model.lambda_rec = 1.0, 0.0, 0.0
            model.kl_beta, model.lambda_sigreg = 0.0, 0.0
            loss, metrics = model(**batch)
            id_norm = _encoder_gradient_norm(loss, parameters)
            model.lambda_sigreg = 1.0
            model.sigreg = SIGReg(**{**model.sigreg_config, "seed": seed})
            keys = ("z_t_main", "z_fwd_main", "main_present", "z_t_wrist", "z_fwd_wrist", "wrist_present")
            mv = model._prepare_multiview_inputs(**{key: batch[key] for key in keys})
            sigreg_norm = _encoder_gradient_norm(mv["sigreg_loss"], parameters)
            suggested = target_ratio * id_norm / sigreg_norm if id_norm > 0 and sigreg_norm > 0 else None
            return {
                "id_loss": metrics["action_loss"], "sigreg_loss": float(mv["sigreg_loss"].detach()),
                "id_gradient_norm": id_norm, "sigreg_gradient_norm": sigreg_norm,
                "target_weighted_gradient_ratio": target_ratio,
                "slot_counts": mv["sigreg_stats"],
                "suggested_lambda_sigreg": suggested,
                "candidate_weights": [suggested / 10, suggested, suggested * 10] if suggested else [],
                "note": "Search scale only; choose a final weight using development data.",
            }
    finally:
        for name, value in old.items():
            setattr(model, name, value)
        model.sigreg = original_sigreg
        if original_sigreg is not None:
            original_sigreg.set_rng_state(rng)


def analyze(model, loader, *, seed=42, calibrate=False, autocast_bf16=False):
    collected, logvars = {}, {}
    calibration = None
    for batch in loader:
        batch = {k: v.to(device=model.device, dtype=model.dtype) if v.is_floating_point()
                 else v.to(model.device) for k, v in batch.items()}
        if calibrate and calibration is None:
            with torch.autocast(model.device.type, dtype=torch.bfloat16, enabled=autocast_bf16):
                calibration = calibrate_gradients(model, batch, seed=seed)
        with torch.no_grad(), torch.autocast(model.device.type, dtype=torch.bfloat16, enabled=autocast_bf16):
            for role, encoder in (("main", model.latent_encoder), ("wrist", model.wrist_latent_encoder)):
                current, future, present = (batch[f"z_t_{role}"], batch[f"z_fwd_{role}"], batch[f"{role}_present"])
                if current.shape[1] == 0 or not bool(present.any()):
                    continue
                flat = _flatten_role_slot_features(current, future, present, model.visual_encoder_type)
                posterior = encoder.posterior_parameters(flat.concatenated)
                mu = posterior.mu_tokens.reshape(flat.batch_size, flat.num_slots, -1, encoder.latent_dim)
                lv = posterior.logvar_grid
                if lv is not None:
                    lv = lv.reshape(flat.batch_size, flat.num_slots, -1)
                for slot in range(flat.num_slots):
                    key = f"{role}_{slot}"
                    keep = present[:, slot].bool()
                    collected.setdefault(key, []).append(mu[:, slot][keep].float().cpu())
                    if lv is not None:
                        logvars.setdefault(key, []).append(lv[:, slot][keep].float().cpu())
    summaries = {key: summarize_slot(torch.cat(chunks), torch.cat(logvars[key]) if key in logvars else None,
                                     seed=seed) for key, chunks in collected.items() if sum(len(c) for c in chunks)}
    return {"slots": summaries, "calibration_first_microbatch": calibration}


def evaluate_id(model, loader, seeds, *, autocast_bf16=False):
    """Paired ID evaluation: fixed diffusion/latent noise for each seed and batch."""
    names = ("lambda_action", "lambda_latent", "lambda_rec", "kl_beta", "lambda_sigreg", "_global_step")
    original = {name: getattr(model, name) for name in names}
    results = []
    try:
        model.lambda_action = 1.0
        model.lambda_latent = model.lambda_rec = model.kl_beta = model.lambda_sigreg = 0.0
        devices = [model.device.index] if model.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            for seed in seeds:
                total, count = 0.0, 0
                for batch_index, batch in enumerate(loader):
                    torch.manual_seed(int(seed) * 1_000_000 + batch_index)
                    batch = {key: value.to(device=model.device, dtype=model.dtype) if value.is_floating_point()
                             else value.to(model.device) for key, value in batch.items()}
                    with torch.autocast(model.device.type, dtype=torch.bfloat16, enabled=autocast_bf16):
                        _, metrics = model(**batch)
                    size = len(batch["actions"])
                    total += float(metrics["action_loss"]) * size
                    count += size
                results.append({"noise_seed": seed, "mean_id_loss": total / count, "num_samples": count})
    finally:
        for name, value in original.items():
            setattr(model, name, value)
    return {"mean_id_loss": sum(item["mean_id_loss"] for item in results) / len(results),
            "per_noise_seed": results, "sampling": "same stochastic Stage-1 bottleneck as training"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="Stage-1 checkpoint folder")
    source.add_argument("--config", nargs="+", help="Stage-1 config followed by overlays; analyze a seeded initialization")
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--features_root", help="Override feature roots; must match checkpoint's features/statistics")
    parser.add_argument("--sample_indices", help="JSON array of fixed dataset frame indices, e.g. a prior report's .indices.json")
    parser.add_argument("--episode_manifest", help="Override the episode split, e.g. the held-out development manifest")
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16, help="Actual statistical microbatch for gradient calibration")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--autocast_bf16", action="store_true", help="FP32 weights with BF16 autocast, matching training")
    parser.add_argument("--id_eval_seeds", nargs="+", type=int, help="Evaluate ID with these paired diffusion and latent noise seeds")
    parser.add_argument("--split_label", choices=["training_diagnostic", "heldout"], default="training_diagnostic")
    parser.add_argument("--calibrate", action="store_true", help="Also measure ID/SIGReg gradients on the first microbatch")
    parser.add_argument("--output", type=Path, required=True, help="JSON output; CSV and indices JSON are written alongside it")
    args = parser.parse_args()
    if args.max_samples < 2 or args.batch_size < 2:
        parser.error("max_samples and batch_size must be >= 2")
    checkpoint = load_checkpoint(args.checkpoint) if args.checkpoint else None
    cfg = checkpoint.config if checkpoint else load_training_config(args.base_config, args.config)
    cfg, runtime = resolve_miniwam_protocol(cfg)
    if cfg["stage"] != 1 or cfg["model"]["architecture"] != "miniwam":
        parser.error("Representation analysis requires a Stage-1 MiniWAM checkpoint/config")
    # Training applies these top-level weights after model construction.
    cfg["model"]["loss"].update(cfg.get("loss", {}))
    model = build_cpu_model_with_seed(lambda: build_miniwam_model(cfg["model"], runtime), args.seed)
    if checkpoint:
        validate_model_protocol(checkpoint.model_protocol, build_model_protocol(model, runtime), "analysis")
        model.load_state_dict(checkpoint.state_dict, strict=True)
    model.to(device=args.device, dtype=getattr(torch, args.dtype)).eval()
    features_root = resolve_feature_roots(args.features_root or cfg["data"].get("features_root"))
    dataset = MiniWAMPreextractedDataset(
        features_root, runtime, action_dim=cfg["data"].get("action_dim", 7),
        proprio_dim=cfg["model"]["context_encoder"].get("proprio_dim"),
        visual_encoder_config=cfg["model"]["visual_encoder"],
        episode_manifest=args.episode_manifest or cfg["data"].get("episode_manifest"),
        episode_manifest_sha256=(None if args.episode_manifest else cfg["data"].get("episode_manifest_sha256")),
    )
    if checkpoint:
        for name, statistics in checkpoint.statistics.items():
            if name not in dataset.statistics or dataset.statistics[name].get("stats_id") != statistics.get("stats_id"):
                raise ValueError(f"Analysis feature statistics do not match checkpoint: {name}")
    if args.sample_indices:
        indices = json.loads(Path(args.sample_indices).read_text())
        if not isinstance(indices, list) or len(indices) < 2 or any(type(i) is not int or not 0 <= i < len(dataset) for i in indices):
            parser.error("sample_indices must contain at least two in-range integer frame indices")
        if len(set(indices)) != len(indices):
            parser.error("sample_indices must not repeat frames")
    else:
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[:args.max_samples].tolist()
    if len(indices) < 2:
        parser.error("Analysis needs at least two dataset samples")
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, collate_fn=MiniWAMCollator())
    report = analyze(model, loader, seed=args.seed, calibrate=args.calibrate, autocast_bf16=args.autocast_bf16)
    if args.id_eval_seeds:
        report["id_evaluation"] = evaluate_id(model, loader, args.id_eval_seeds, autocast_bf16=args.autocast_bf16)
    digest = hashlib.sha256(json.dumps(indices).encode()).hexdigest()
    report["protocol"] = {
        "checkpoint": str(Path(args.checkpoint).resolve()) if checkpoint else None,
        "config": cfg, "features_root": features_root,
        "episode_dirs": [str(p.resolve()) for p in dataset.episode_dirs],
        "episode_manifest_sha256": dataset.episode_manifest_sha256,
        "indices_sha256": digest, "num_samples": len(indices),
        "batch_size": args.batch_size, "seed": args.seed, "split_label": args.split_label,
        "autocast_bf16": args.autocast_bf16, "id_eval_seeds": args.id_eval_seeds,
        "sigreg_statistics": "full_selected_sample_per_slot_for_diagnostics; first_microbatch_for_calibration",
        "sigreg_reference": REFERENCE_URL,
        "diagnostic_sigreg_config": SIGReg(seed=args.seed).config,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".indices.json").write_text(json.dumps(indices) + "\n")
    rows = [{"slot": key, **{k: v for k, v in summary.items() if not isinstance(v, list)}}
            for key, summary in report["slots"].items()]
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(args.output), "calibration": report["calibration_first_microbatch"]}, indent=2))


if __name__ == "__main__":
    main()
