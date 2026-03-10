import torch
import torch.nn as nn
from torch import Tensor


class Flow(nn.Module):
    """Unconditional flow matching velocity field."""

    def __init__(self, dim: int = 2, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h), nn.ELU(),
            nn.Linear(h, h),       nn.ELU(),
            nn.Linear(h, h),       nn.ELU(),
            nn.Linear(h, dim),
        )

    def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
        return self.net(torch.cat([t, x_t], dim=-1))

    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        t_s = t_start.view(1, 1).expand(x_t.shape[0], 1)
        dt  = (t_end - t_start).float()
        v_s = self(t_s, x_t)
        v_m = self(t_s + dt / 2, x_t + v_s * dt / 2)
        return x_t + dt * v_m


class ConditionedFlow(nn.Module):
    """Flow matching velocity field conditioned on an ESM-2 per-residue embedding.

    The conditioning vector is the full ESM-2 hidden state for a residue,
    concatenated directly with time and the current position.

    Works for both:
    - Per-residue conditioning (per-token ESM-2 hidden state, 320-dim)
    - Pooled protein-level conditioning (mean-pooled ESM-2, 320-dim)

    Args:
        dim:       Dimensionality of the data space (2 for phi/psi angles).
        embed_dim: Dimensionality of the ESM-2 embedding (320 for ESM-2 8M).
        h:         Hidden layer width.
    """

    def __init__(self, dim: int = 2, embed_dim: int = 320, h: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1 + embed_dim, h), nn.ELU(),
            nn.Linear(h, h),                   nn.ELU(),
            nn.Linear(h, h),                   nn.ELU(),
            nn.Linear(h, h),                   nn.ELU(),
            nn.Linear(h, dim),
        )

    def forward(self, t: Tensor, x_t: Tensor, emb: Tensor) -> Tensor:
        return self.net(torch.cat([t, x_t, emb], dim=-1))

    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor, emb: Tensor) -> Tensor:
        t_s = t_start.view(1, 1).expand(x_t.shape[0], 1)
        dt  = (t_end - t_start).float()
        v_s = self(t_s, x_t, emb)
        v_m = self(t_s + dt / 2, x_t + v_s * dt / 2, emb)
        return x_t + dt * v_m
