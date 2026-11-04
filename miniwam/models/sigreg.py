"""Sketched Gaussian regularization of deterministic transition representations.

Independent implementation of the Epps--Pulley objective in LeJEPA
(Balestriero and LeCun, arXiv:2511.08544). Numerical convention: symmetric
trapezoidal integration on [-3,3], multiplied by the number of valid samples.
The upstream numerical reference is MINIMAL.md at REFERENCE_COMMIT; no
upstream source is bundled. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.nn.functional import all_reduce

REFERENCE_COMMIT = "c293d291ca87cd4fddee9d3fffe4e914c7272052"
REFERENCE_URL = f"https://github.com/galilai-group/lejepa/blob/{REFERENCE_COMMIT}/MINIMAL.md"
SIGREG_DEFAULTS = {
    "target": "mu",
    "sample_unit": "flattened_grid_per_camera_slot",
    "num_slices": 256,
    "num_points": 17,
    "t_max": 3.0,
    "distributed_statistics": "global_microbatch",
    "seed": 42,
    "slice_chunk_size": 64,
}


def resolve_sigreg_config(config: dict | None = None) -> dict:
    config = dict(config or {})
    unknown = set(config) - set(SIGREG_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown model.sigreg settings: {sorted(unknown)}")
    resolved = {**SIGREG_DEFAULTS, **config}
    for key in ("target", "sample_unit", "distributed_statistics"):
        if resolved[key] != SIGREG_DEFAULTS[key]:
            raise ValueError(f"model.sigreg.{key} must be {SIGREG_DEFAULTS[key]!r}.")
    for key, minimum in (("num_slices", 1), ("num_points", 2), ("slice_chunk_size", 1)):
        value = resolved[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"model.sigreg.{key} must be an integer >= {minimum}.")
    if not math.isfinite(float(resolved["t_max"])) or float(resolved["t_max"]) <= 0:
        raise ValueError("model.sigreg.t_max must be finite and positive.")
    if isinstance(resolved["seed"], bool) or not isinstance(resolved["seed"], int):
        raise ValueError("model.sigreg.seed must be an integer.")
    return resolved


class SIGReg(nn.Module):
    """Regularize rows of [N,D]; each row is one complete camera-slot grid.

    Projection RNG lives on CPU independently of the model/global RNG. It is
    saved as training state, not as a model buffer, so old weight files and
    BF16/FSDP buffer casts cannot change the statistic or its precision.
    """

    def __init__(self, **config):
        super().__init__()
        self.config = resolve_sigreg_config(config)
        self.generator = torch.Generator(device="cpu").manual_seed(self.config["seed"])

    def get_rng_state(self) -> torch.Tensor:
        return self.generator.get_state().clone()

    def set_rng_state(self, state: torch.Tensor) -> None:
        self.generator.set_state(state.cpu())

    def forward(self, samples, present=None, *, directions=None, synchronize=True):
        return self.loss_and_count(
            samples, present, directions=directions, synchronize=synchronize,
        )[0]

    def loss_and_count(self, samples, present=None, *, directions=None, synchronize=True):
        """Return scalar loss and global valid count (including skipped slots).

        All ranks must call this in the same slot order, including ranks with
        zero local rows. The differentiable SUM has SUM backward: combined
        with DDP/FSDP gradient averaging this matches a global batch exactly.
        """
        if samples.ndim != 2 or samples.shape[1] < 1:
            raise ValueError("SIGReg samples must have shape [N,D] with D > 0.")
        distributed = synchronize and dist.is_available() and dist.is_initialized()
        with torch.autocast(samples.device.type, enabled=False):
            x = samples.float()
            if present is not None:
                if present.shape != samples.shape[:1]:
                    raise ValueError("SIGReg presence mask must have shape [N].")
                # Select before trig/projection: absent padding may contain NaNs.
                x = x[present.to(device=x.device, dtype=torch.bool)]
            if distributed and not x.requires_grad and torch.is_grad_enabled():
                # A rank with no encoded local slots must still execute the
                # autograd collectives in backward, in the same order.
                x = x.detach().requires_grad_()
            count = x.new_tensor(x.shape[0])
            if distributed:
                dist.all_reduce(count)
            if count.item() < 2:
                return x.sum() * 0.0, count

            if directions is None:
                directions = torch.randn(
                    x.shape[1], self.config["num_slices"], generator=self.generator,
                    device="cpu", dtype=torch.float32,
                ).to(x.device)
                if distributed:
                    dist.broadcast(directions, src=0)
            else:
                directions = directions.to(device=x.device, dtype=torch.float32)
            if directions.ndim != 2 or directions.shape[0] != x.shape[1] or directions.shape[1] < 1:
                raise ValueError("SIGReg directions must have shape [D,K] with K > 0.")
            norms = torch.linalg.vector_norm(directions, dim=0, keepdim=True)
            if not bool(torch.isfinite(norms).all()) or bool((norms == 0).any()):
                raise ValueError("SIGReg directions must be finite and nonzero.")
            directions = directions / norms
            t = torch.linspace(0, self.config["t_max"], self.config["num_points"],
                               device=x.device, dtype=torch.float32)
            gaussian_cf = torch.exp(-0.5 * t.square())
            loss = x.sum() * 0.0
            for axes in directions.split(self.config["slice_chunk_size"], dim=1):
                phase = torch.nn.functional.linear(x, axes.T)[..., None] * t
                moments = torch.stack((phase.cos().sum(0), phase.sin().sum(0)))
                if distributed:
                    moments = all_reduce(moments, op=dist.ReduceOp.SUM)
                real, imaginary = moments / count
                discrepancy = (real - gaussian_cf).square() + imaginary.square()
                integrals = 2 * torch.trapezoid(discrepancy * gaussian_cf, t, dim=-1)
                loss = loss + integrals.sum() * count / directions.shape[1]
        return loss, count
