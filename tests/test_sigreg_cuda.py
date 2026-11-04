"""GPU coverage for FP32 statistics under BF16 training and FSDP collectives."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from miniwam.models.sigreg import SIGReg
from test_miniwam_ablation_axes import batch_for, build_tiny, tiny_cfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_cuda_fp32_statistics_inside_bf16_autocast():
    x = torch.randn(32, 19, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    axes = torch.randn(19, 31, device="cuda")
    objective = SIGReg(slice_chunk_size=11).cuda().bfloat16()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = objective(x, directions=axes)
    expected = SIGReg(slice_chunk_size=11)(x.float(), directions=axes)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    assert x.grad.dtype == torch.bfloat16 and torch.isfinite(x.grad).all()


def test_cuda_four_arm_100_update_smoke():
    torch.set_num_threads(1)
    for arm in range(4):
        torch.manual_seed(42)
        cfg = tiny_cfg("dinov3", reconstruction=arm < 2)
        cfg["model"]["latent_encoder"]["use_kl"] = True
        cfg["model"]["loss"].update(kl_beta=1e-6 if arm == 0 else 0, lambda_sigreg=.01 if arm == 2 else 0)
        cfg["model"]["sigreg"] = {"num_slices": 32, "slice_chunk_size": 16}
        model, _ = build_tiny(cfg)
        model.cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        batch = {k: v.detach().cuda() for k, v in batch_for(cfg, batch=4).items()}
        # Include a partially absent camera in the numerical stability smoke.
        batch["wrist_present"][0] = False
        for step in range(100):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, stats = model(**batch)
            assert torch.isfinite(loss), (arm, step, stats)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            assert torch.isfinite(norm), (arm, step)
            assert model.latent_encoder.logvar_head.weight.grad is not None
            optimizer.step()
        assert model._global_step == 100
        assert (stats["weighted_sigreg_loss"] > 0) == (arm == 2)


def _fsdp_worker(rank, init_file):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=45))
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision

        torch.manual_seed(42)
        cfg = tiny_cfg("dinov3")
        cfg["model"]["latent_encoder"]["use_kl"] = True
        cfg["model"]["loss"].update(kl_beta=0., lambda_sigreg=.01)
        cfg["model"]["sigreg"] = {"num_slices": 12, "slice_chunk_size": 5}
        raw, _ = build_tiny(cfg)
        raw.cuda(rank)
        model = FSDP(raw, device_id=rank, use_orig_params=True,
                     mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16))
        batch = {k: v.detach().cuda(rank) for k, v in batch_for(cfg, batch=3).items()}
        if rank == 0:
            batch["wrist_present"][:] = False
        else:
            batch["main_present"][0] = False
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, stats = model(**batch)
            assert torch.isfinite(loss)
            assert stats["sigreg_count_slot_0"] == 5
            assert stats["sigreg_count_slot_2"] == 3
            loss.backward()
            assert torch.isfinite(model.clip_grad_norm_(1.0))
            optimizer.step()
        # No rank-dependent projection sequence after different local presence.
        states = [None, None]
        dist.all_gather_object(states, raw.sigreg.get_rng_state().tolist())
        assert states[0] == states[1]
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
def test_two_gpu_fsdp_with_a_missing_camera_role(tmp_path):
    mp.spawn(_fsdp_worker, args=(str(tmp_path / "nccl_init"),), nprocs=2)
