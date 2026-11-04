"""Numerical, distributed and experiment contracts for the SIGReg ablation."""

from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from miniwam.checkpointing import (
    Checkpoint, build_model_protocol, capture_sigreg_state, initialize_stage2,
    restore_sigreg_state, validate_model_protocol,
)
from miniwam.config import load_training_config
from miniwam.models.sigreg import SIGReg
from miniwam.protocol import resolve_miniwam_protocol
from test_miniwam_ablation_axes import batch_for, build_tiny, tiny_cfg

torch.set_num_threads(1)


def _reference(x, axes, points=17, maximum=3.0):
    axes = axes / axes.norm(dim=0)
    knots = torch.linspace(0, maximum, points)
    weights = torch.ones(points) * (2 * maximum / (points - 1))
    weights[0] *= 0.5
    weights[-1] *= 0.5
    window = (-knots.square() / 2).exp()
    phase = (x @ axes)[..., None] * knots
    error = (phase.cos().mean(0) - window).square() + phase.sin().mean(0).square()
    return ((error * weights * window).sum(-1) * len(x)).mean()


def test_reference_mask_fp32_gradient_and_independent_rng():
    torch.manual_seed(18)
    x = torch.randn(19, 12, requires_grad=True)
    axes = torch.randn(12, 23)
    mask = torch.arange(19) % 3 != 0
    x.data[~mask] = float("nan")
    loss_fn = SIGReg(slice_chunk_size=7)
    before = torch.get_rng_state()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, count = loss_fn.loss_and_count(x, mask, directions=axes)
    assert loss.dtype == torch.float32 and count == mask.sum()
    expected = _reference(x[mask], axes)
    torch.testing.assert_close(loss, expected)
    grad = torch.autograd.grad(loss, x)[0]
    ref_grad = torch.autograd.grad(expected, x)[0]
    torch.testing.assert_close(grad, ref_grad, atol=3e-7, rtol=1e-5)
    assert torch.isfinite(grad).all() and (grad[~mask] == 0).all()
    loss_fn(x, mask)
    assert torch.equal(before, torch.get_rng_state())
    state = loss_fn.get_rng_state()
    next_loss = loss_fn(x, mask)
    loss_fn.set_rng_state(state)
    torch.testing.assert_close(loss_fn(x, mask), next_loss, rtol=0, atol=0)
    assert not loss_fn.state_dict()


def test_detects_distribution_errors_and_spatial_only_diversity():
    g = torch.Generator().manual_seed(42)
    normal = torch.randn(2048, 16, generator=g)
    axes = torch.randn(16, 128, generator=g)
    loss_fn = SIGReg()
    normal_loss = loss_fn(normal, directions=axes)
    for x in (torch.zeros_like(normal), normal + 3, normal * 4,
              normal[:1].expand_as(normal)):
        loss = loss_fn(x, directions=axes)
        assert torch.isfinite(loss) and loss > normal_loss * 10
    # A fixed grid has diverse tokens, but its flattened grid is identical
    # across transitions and must still be penalized.
    assert loss_fn(normal[:1].repeat(32, 1), directions=axes) > 1
    for n in (0, 1):
        x = torch.empty(n, 16, requires_grad=True)
        loss, count = loss_fn.loss_and_count(x)
        assert loss == 0 and count == n
        loss.backward()


@pytest.mark.parametrize("visual_type", ["wan_vae", "dinov3"])
def test_training_reuses_backbone_and_only_regularizes_mu(visual_type):
    cfg = tiny_cfg(visual_type)
    cfg["model"]["latent_encoder"]["use_kl"] = True
    cfg["model"]["loss"].update(kl_beta=0.0, lambda_sigreg=0.01)
    cfg["model"]["sigreg"] = {"num_slices": 12, "slice_chunk_size": 5}
    model, _ = build_tiny(cfg)
    batch = batch_for(cfg, batch=3)
    batch["wrist_present"][0] = False
    calls = []
    handle = model.latent_encoder.backbone.register_forward_hook(lambda *args: calls.append(1))
    mv_keys = {"z_t_main", "z_fwd_main", "main_present", "z_t_wrist", "z_fwd_wrist", "wrist_present"}
    mv = model._prepare_multiview_inputs(**{k: v for k, v in batch.items() if k in mv_keys})
    mv["sigreg_loss"].backward()
    assert len(calls) == 1
    handle.remove()
    assert model.latent_encoder.mu_head.weight.grad.abs().sum() > 0
    assert model.latent_encoder.logvar_head.weight.grad is None
    model.zero_grad(set_to_none=True)
    total, metrics = model(**batch)
    total.backward()
    assert metrics["weighted_kl_loss"] == 0
    assert metrics["weighted_sigreg_loss"] == pytest.approx(metrics["sigreg_loss"] * 0.01)
    assert model.latent_encoder.logvar_head.weight.grad.abs().sum() > 0
    torch.testing.assert_close(total, total.new_tensor(metrics["action_loss"] + metrics["weighted_sigreg_loss"]))


@pytest.mark.parametrize("benchmark", ["libero", "robotwin"])
def test_four_arm_configs_and_automatic_stage_paths(benchmark):
    root = Path(__file__).resolve().parents[1]
    arms = ["id_recon_kl", "id_recon_no_kl", "id_sigreg", "id_only"]
    for i, arm in enumerate(arms):
        overlay = root / f"configs/ablations/objective/{arm}.yaml"
        configs = []
        for stage in (1, 2):
            cfg = load_training_config(root / "configs/base.yaml", [
                root / f"configs/{benchmark}/stage{stage}.yaml", overlay,
            ])
            cfg, _ = resolve_miniwam_protocol(cfg)
            assert cfg["model"]["latent_encoder"]["use_kl"] is True
            assert cfg["loss"]["lambda_sigreg"] == (0.01 if i == 2 and stage == 1 else 0)
            configs.append(cfg)
        first, second = configs
        assert first["model"]["reconstruction"]["decoder_enabled"] == (i < 2)
        assert first["loss"]["kl_beta"] == (1e-6 if i == 0 else 0)
        assert second["checkpoint"]["policy_init"] == "fresh"
        assert second["checkpoint"]["stage1_path"] == first["training"]["output_dir"] + "/latest"


def test_sampling_unchanged_at_beta_zero_and_fresh_transfer_is_fair():
    source = tiny_cfg(reconstruction=True)
    source["model"]["latent_encoder"]["use_kl"] = True
    models, runtimes = [], []
    for i in range(4):
        cfg = deepcopy(source)
        cfg["model"]["loss"].update(kl_beta=1e-6 if i == 0 else 0,
                                       lambda_sigreg=0.01 if i == 2 else 0)
        cfg["model"]["reconstruction"].update(decoder_enabled=i < 2, lambda_rec=1.0 if i < 2 else 0)
        torch.manual_seed(42)
        model, runtime = build_tiny(cfg)
        models.append(model)
        runtimes.append(runtime)
    common = set.intersection(*(set(m.state_dict()) for m in models))
    for key in common:
        for model in models[1:]:
            torch.testing.assert_close(model.state_dict()[key], models[0].state_dict()[key], atol=0, rtol=0)
    z = torch.randn(2, 16, 5, 8, 8)
    b = models[1].latent_encoder
    assert not torch.equal(b(z, sample=True)[0], b(z, sample=True)[0])
    torch.testing.assert_close(b(z, sample=False)[0], b.posterior_parameters(z).mu_tokens)
    transferred = []
    for arm, (model, runtime) in enumerate(zip(models, runtimes)):
        cfg2 = tiny_cfg(stage=2)
        cfg2["model"]["latent_encoder"]["use_kl"] = True
        torch.manual_seed(77)
        second, runtime2 = build_tiny(cfg2)
        checkpoint = Checkpoint(Path("unused"), model.state_dict(), {"stage": 1}, {},
                                {"model_protocol": build_model_protocol(model, runtime)})
        initialize_stage2(second, checkpoint, "fresh", runtime2)
        assert not any(p.requires_grad for p in second.latent_encoder.parameters())
        transferred.append(second)
    for key, value in transferred[0].state_dict().items():
        if not key.startswith(("latent_encoder.", "wrist_latent_encoder.")):
            for model in transferred[1:]:
                torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)


def test_resume_protocol_rng_and_legacy_checkpoint():
    cfg = tiny_cfg()
    cfg["model"]["loss"]["lambda_sigreg"] = 0.02
    model, runtime = build_tiny(cfg)
    layout = dict(world_size=2, per_device_batch_size=8, gradient_accumulation_steps=16)
    state = capture_sigreg_state(model, **layout)
    x = torch.randn(8, 16)
    expected = model.sigreg(x)
    restore_sigreg_state(model, state, **layout)
    torch.testing.assert_close(model.sigreg(x), expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="microbatch"):
        restore_sigreg_state(model, state, **{**layout, "world_size": 1})
    protocol = build_model_protocol(model, runtime)
    changed = deepcopy(protocol)
    changed["sigreg"]["lambda_sigreg"] *= 2
    with pytest.raises(ValueError, match="sigreg"):
        validate_model_protocol(protocol, changed, "resume")
    disabled, runtime = build_tiny(tiny_cfg())
    current = build_model_protocol(disabled, runtime)
    legacy = {k: v for k, v in current.items() if k != "sigreg"}
    validate_model_protocol(legacy, current, "legacy")
    restore_sigreg_state(disabled, {}, **layout)


def _distributed_worker(rank, init_file, result_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=40))
    try:
        g = torch.Generator().manual_seed(8)
        source = torch.randn(7, 5, generator=g)
        weights = torch.randn(5, 4, generator=g).requires_grad_()
        local = source[:3] if rank == 0 else source[3:]
        # Slot 0 has uneven counts; slot 1 has no samples on rank 0.
        loss_fn = SIGReg(num_slices=13, slice_chunk_size=5)
        losses, counts = [], []
        for slot in range(2):
            samples = local @ weights if slot == 0 or rank == 1 else torch.empty(0, 4)
            loss, count = loss_fn.loss_and_count(samples)
            losses.append(loss)
            counts.append(count)
        loss = sum(l * n for l, n in zip(losses, counts)) / sum(counts)
        loss.backward()
        # This is the parameter-gradient averaging used by DDP/FSDP.
        dist.all_reduce(weights.grad)
        weights.grad /= 2
        torch.save({"loss": loss.detach(), "grad": weights.grad}, Path(result_dir) / f"{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_distributed_loss_and_ddp_parameter_gradients_match_global_batch(tmp_path):
    mp.spawn(_distributed_worker, args=(str(tmp_path / "init"), str(tmp_path)), nprocs=2)
    g = torch.Generator().manual_seed(8)
    source = torch.randn(7, 5, generator=g)
    weights = torch.randn(5, 4, generator=g).requires_grad_()
    loss_fn = SIGReg(num_slices=13, slice_chunk_size=5)
    expected = (loss_fn(source @ weights) * 7 + loss_fn(source[3:] @ weights) * 4) / 11
    expected.backward()
    for rank in range(2):
        actual = torch.load(tmp_path / f"{rank}.pt", weights_only=True)
        torch.testing.assert_close(actual["loss"], expected)
        torch.testing.assert_close(actual["grad"], weights.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf")])
def test_invalid_weights_rejected(weight):
    cfg = tiny_cfg()
    cfg["loss"] = {"lambda_sigreg": weight}
    with pytest.raises(ValueError, match="lambda_sigreg"):
        resolve_miniwam_protocol(cfg)


def test_stage2_and_unsupported_statistical_protocol_rejected():
    cfg = tiny_cfg(stage=2)
    cfg["loss"] = {"lambda_sigreg": 0.1}
    with pytest.raises(ValueError, match="Stage 1"):
        resolve_miniwam_protocol(cfg)
    with pytest.raises(ValueError, match="sample_unit"):
        SIGReg(sample_unit="tokens_as_samples")
