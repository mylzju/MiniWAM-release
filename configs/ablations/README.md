# Ablations

Every ablation in the paper is the default LIBERO recipe plus one or more
overlays from this folder. Pass the same overlays to both stages, except the
Stage-2-only overlays [`no_cotrain.yaml`](no_cotrain.yaml) and [`fresh_policy.yaml`](fresh_policy.yaml), which go on
Stage 2 alone:

```bash
python scripts/train.py --config configs/libero/stage1.yaml <overlays>
python scripts/train.py --config configs/libero/stage2.yaml <overlays>
```

A stacked run is named after its files, e.g.
`outputs/libero_stage1__grid2x2_d64__recon_only`, and Stage 2 loads the
matching Stage-1 run automatically. Every ablation arm starts Stage 2 from a
seeded policy; only the PRISM encoder comes from the arm's own Stage 1.
Evaluate on LIBERO and on the fixed LIBERO-Plus-2K subset
([evaluation](../../docs/evaluation.md#libero-plus-2k)).

| Overlay | Effect |
| --- | --- |
| [`capacity/grid2x2_d64.yaml`](capacity/grid2x2_d64.yaml) | pool the PRISM grid to 2×2 locations, 64 channels |
| [`capacity/grid1x1_d16.yaml`](capacity/grid1x1_d16.yaml) | pool to 1×1, 16 channels |
| [`objective/recon_only.yaml`](objective/recon_only.yaml) | Stage 1 trains the encoder by feature reconstruction only |
| [`objective/inverse_only.yaml`](objective/inverse_only.yaml) | Stage 1 uses inverse dynamics only (no decoder) |
| [`no_cotrain.yaml`](no_cotrain.yaml) | Stage 2 sets the world-token loss weight to 0 |
| [`fresh_policy.yaml`](fresh_policy.yaml) | Stage 2 of the default recipe starts from a seeded policy (the reference arm) |

[`libero/`](libero/) and [`robotwin/`](robotwin/) hold full stage configs rather than overlays.

## Table 1: world-modeling targets

Evaluated on LIBERO and LIBERO-Plus-2K.

| Features | Target | Stage 1 | Stage 2 |
| --- | --- | --- | --- |
| DINOv3 | PRISM | [`libero/stage1.yaml`](../libero/stage1.yaml) | [`libero/stage2.yaml`](../libero/stage2.yaml) + [`ablations/fresh_policy.yaml`](fresh_policy.yaml) |
| DINOv3 | native co-train | — | [`ablations/libero/native_dinov3.yaml`](libero/native_dinov3.yaml) |
| DINOv3 | w/o co-train | [`libero/stage1.yaml`](../libero/stage1.yaml) | [`libero/stage2.yaml`](../libero/stage2.yaml) + [`ablations/no_cotrain.yaml`](no_cotrain.yaml) |
| Wan2.1-VAE | PRISM | [`ablations/libero/wan_vae_stage1.yaml`](libero/wan_vae_stage1.yaml) | [`ablations/libero/wan_vae_stage2.yaml`](libero/wan_vae_stage2.yaml) |
| Wan2.1-VAE | native co-train | — | [`ablations/libero/native_wan_vae.yaml`](libero/native_wan_vae.yaml) |
| Wan2.1-VAE | w/o co-train | [`ablations/libero/wan_vae_stage1.yaml`](libero/wan_vae_stage1.yaml) | [`ablations/libero/wan_vae_stage2.yaml`](libero/wan_vae_stage2.yaml) + [`ablations/no_cotrain.yaml`](no_cotrain.yaml) |

Paths are relative to `configs/`. A cell with two files is one run: pass both
to `--config`, stage config first, e.g.
`--config configs/libero/stage2.yaml configs/ablations/no_cotrain.yaml`.

The native co-training controls have no PRISM encoder: one stage jointly
denoises the actions and all native future features (16 DINOv3 patch grids, or
4 Wan-VAE latent frames).

## Table 4: PRISM objective and capacity

Evaluated on LIBERO-Plus-2K. Each cell is the default LIBERO recipe plus the
overlays below, passed to both stages; [`fresh_policy.yaml`](fresh_policy.yaml) goes on Stage 2 only.

| Cell | Overlays |
| --- | --- |
| 4×4 Both | [`fresh_policy.yaml`](fresh_policy.yaml) (Stage 2 only) |
| 4×4 Recon. / Inv. dyn. | [`objective/recon_only.yaml`](objective/recon_only.yaml) / [`objective/inverse_only.yaml`](objective/inverse_only.yaml) |
| 2×2 Both | [`capacity/grid2x2_d64.yaml`](capacity/grid2x2_d64.yaml) |
| 2×2 Recon. / Inv. dyn. | [`capacity/grid2x2_d64.yaml`](capacity/grid2x2_d64.yaml) + [`objective/recon_only.yaml`](objective/recon_only.yaml) / [`objective/inverse_only.yaml`](objective/inverse_only.yaml) |
| 1×1 Both | [`capacity/grid1x1_d16.yaml`](capacity/grid1x1_d16.yaml) |
| 1×1 Recon. / Inv. dyn. | [`capacity/grid1x1_d16.yaml`](capacity/grid1x1_d16.yaml) + [`objective/recon_only.yaml`](objective/recon_only.yaml) / [`objective/inverse_only.yaml`](objective/inverse_only.yaml) |

## RoboTwin controls

[`ablations/robotwin/`](robotwin/) holds RoboTwin versions of the Wan-VAE and
native-feature controls (horizon 16). [`robotwin/stage2.yaml`](../robotwin/stage2.yaml) +
[`ablations/no_cotrain.yaml`](no_cotrain.yaml) gives the RoboTwin model without co-training.

## Stage-1 KL / reconstruction / SIGReg

These four overlays isolate the loss changes while retaining the same
stochastic bottleneck (`use_kl: true`) in Stage 1. All start Stage 2 with
`policy_init: fresh`, transferring only their own frozen encoders; Stage 2
uses deterministic `mu` targets. Apply the same overlay to both stages.

| Arm | Overlay | Stage-1 loss |
| --- | --- | --- |
| A | [`objective/id_recon_kl.yaml`](objective/id_recon_kl.yaml) | ID + reconstruction + KL |
| B | [`objective/id_recon_no_kl.yaml`](objective/id_recon_no_kl.yaml) | ID + reconstruction |
| C | [`objective/id_sigreg.yaml`](objective/id_sigreg.yaml) | ID + SIGReg(mu) |
| D | [`objective/id_only.yaml`](objective/id_only.yaml) | ID only |

A→B removes only KL; B→D removes only reconstruction; D→C adds only SIGReg.
B→C changes two losses and compares complete training recipes.
`inverse_only.yaml` above retains the original KL weight and is different
from D. Setting `kl_beta: 0` retains sampling; `use_kl: false` also removes
the posterior variance head and sampling and is a separate ablation.

C's `lambda_sigreg: 0.01` is a **pilot value requiring calibration**. Its
FP32 Epps–Pulley statistic uses 256 unit directions and 17 integration points
on [0,3], with symmetric integration and the valid-sample-count multiplier.
Each sample is a full flattened `mu` grid; each main/wrist camera slot is
regularized separately, excluding missing cameras. Slot losses are averaged
with valid-count weights. SIGReg is disabled by default and in Stage 2.

Across GPUs, the empirical characteristic function uses the **global
microbatch**, with differentiable reductions. Gradient accumulation does not
increase this statistical batch. Keep the GPU count, per-device batch and
accumulation fixed for all arms. The projection RNG is independent of action
noise and saved in training checkpoints; SIGReg resume also checks the layout
and regularizer settings. Old checkpoints remain usable with SIGReg disabled.

Use explicit paths for multiple training seeds; `--seed` alone does not name
a new output directory:

```bash
python scripts/train.py --config configs/libero/stage1.yaml \
  configs/ablations/objective/id_sigreg.yaml --seed 42 \
  --output_dir outputs/libero_C_seed42_stage1

python scripts/train.py --config configs/libero/stage2.yaml \
  configs/ablations/objective/id_sigreg.yaml --seed 42 \
  --stage1_checkpoint outputs/libero_C_seed42_stage1/latest \
  --output_dir outputs/libero_C_seed42_stage2
```

Before training, inspect a seeded initialization and calibrate the search
scale using actual cached data (set the same microbatch as training):

```bash
python scripts/analysis/analyze_stage1_representations.py \
  --config configs/libero/stage1.yaml configs/ablations/objective/id_sigreg.yaml \
  --features_root /path/to/features --batch_size 16 --max_samples 256 \
  --device cuda:0 --calibrate --output outputs/sigreg_calibration.json
```

This takes no optimizer steps. It reports raw ID and SIGReg gradient norms
on shared encoder weights, plus a candidate λ grid around a weighted gradient
ratio of 0.1. That ratio sets a search scale; choose the final weight using
development data, then freeze it before the main comparison. For distributed
training, use a calibration batch equal to the global microbatch, on one GPU
if it fits; a local microbatch measures a different regularizer.

For a trained encoder, replace `--config ...` with `--checkpoint <stage1>`.
The tool writes JSON, CSV and `.indices.json`; pass the same indices through
`--sample_indices` for each arm with the same ordered feature roots. Output
includes mu scale, per-token cross-sample variance, covariance spectrum,
effective rank and its finite-sample upper bound, posterior variance and
nonfinite counts. The default split label is `training_diagnostic`; a held-out
label does not create or enforce a data split. Use a separately prepared
held-out feature root/indices when reporting validation.

See the [experiment plan](../../docs/sigreg_ablation_plan.md) for the fixed
Stage-2 protocol, evaluation and optional deterministic/E controls.
