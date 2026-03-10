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

class CoordRefiner(nn.Module):
    """Transformer encoder that refines NeRF-reconstructed backbone coordinates.

    Each residue is represented as flattened [N, CA, C, O] xyz coordinates
    (12-dim) concatenated with its ESM-2 hidden state (320-dim).  The model
    predicts a residual correction δ that is added to the noisy input, so the
    output projection is zero-initialised and training starts from an identity
    mapping.

    Args:
        coord_dim:  Flattened atoms per residue (4 atoms × 3 = 12).
        esm2_dim:   ESM-2 embedding dimension (320 for the 8M model).
        d_model:    Transformer hidden size.
        nhead:      Number of attention heads (d_model must be divisible by nhead).
        num_layers: Number of TransformerEncoderLayer blocks.
        ffn_dim:    Feed-forward inner dimension.
        dropout:    Dropout probability.
        max_len:    Maximum sequence length for learned positional embeddings.
    """

    def __init__(
        self,
        coord_dim:  int   = 12,
        esm2_dim:   int   = 320,
        d_model:    int   = 256,
        nhead:      int   = 8,
        num_layers: int   = 4,
        ffn_dim:    int   = 1024,
        dropout:    float = 0.1,
        max_len:    int   = 2048,
    ):
        super().__init__()
        self.input_proj = nn.Linear(coord_dim + esm2_dim, d_model)
        self.pos_emb    = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, ffn_dim, dropout,
            batch_first=True,
            norm_first=True,   # pre-LN for stable gradients
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers, enable_nested_tensor=False,
        )
        self.output_proj = nn.Linear(d_model, coord_dim)
        # Zero-init → starts as identity (no correction at epoch 0)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, noisy_coords: Tensor, esm2: Tensor) -> Tensor:
        """
        Args:
            noisy_coords: (L, coord_dim)  flattened backbone atoms, centred
            esm2:         (L, esm2_dim)   per-residue ESM-2 hidden states
        Returns:
            refined:      (L, coord_dim)  noisy_coords + predicted residual δ
        """
        L   = noisy_coords.shape[0]
        # Clamp indices so proteins longer than max_len don't cause an
        # IndexError in the embedding table.  Positions beyond max_len-1
        # wrap to the last learned embedding, which is a graceful fallback.
        pos = torch.arange(L, device=noisy_coords.device).clamp(
            max=self.pos_emb.num_embeddings - 1
        )
        x   = torch.cat([noisy_coords, esm2], dim=-1)        # (L, coord+esm2)
        x   = self.input_proj(x) + self.pos_emb(pos)         # (L, d_model)
        x   = self.transformer(x.unsqueeze(0)).squeeze(0)    # (L, d_model)
        return noisy_coords + self.output_proj(x)             # residual refinement


class CombinedConditionedFlow(nn.Module):
    def __init__(
        self,
        dim:          int = 2,
        esm2_dim:     int = 320,
        esm2_proj_dim:int = 64,
        h:            int = 512,
    ):
        super().__init__()
        self.esm2_proj  = nn.Sequential(nn.Linear(esm2_dim, esm2_proj_dim), nn.ELU())
        in_dim = dim + 1 + esm2_proj_dim  # 2+1+64 = 67
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ELU(),
            nn.Linear(h, h),      nn.ELU(),
            nn.Linear(h, h),      nn.ELU(),
            nn.Linear(h, h),      nn.ELU(),
            nn.Linear(h, dim),
        )

    def forward(self, t: Tensor, x_t: Tensor, esm2: Tensor) -> Tensor:
        esm2_emb = self.esm2_proj(esm2)        # (B, 64)
        x_in     = torch.cat([t, x_t, esm2_emb], dim=-1)
        return self.net(x_in)

    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor,
             esm2: Tensor) -> Tensor:
        t_s = t_start.view(1, 1).expand(x_t.shape[0], 1)
        dt  = t_end - t_s
        v_s = self(t_s, x_t, esm2)
        x_m = x_t + v_s * dt / 2
        v_m = self(t_s + dt / 2, x_m, esm2)
        return x_t + dt * v_m
