"""
pipeline.py — End-to-end inference pipelines.

generate_structure(esm2_emb, flow_model, ...)
    ESM-2 embeddings → flow matching φ/ψ → NeRF → (PDB)
    Optionally accepts sequence + esm_model + tokenizer to compute embeddings on the fly.

generate_and_refine(esm2_emb, flow_model, refiner_model, ...)
    ESM-2 embeddings → flow matching φ/ψ → NeRF → CoordRefiner → (PDB)
    Optionally accepts sequence + esm_model + tokenizer to compute embeddings on the fly.
"""

from typing import Optional

import numpy as np
import torch

from pipeline.nerf import build_backbone, write_pdb
from pipeline.geometry import center_coords


def _resolve_esm2(esm2_emb, sequence, esm_model, tokenizer, device):
    """Return (L, 320) ESM-2 tensor on *device*.

    If *esm2_emb* is provided it is used directly (moved to *device*).
    Otherwise *sequence*, *esm_model*, and *tokenizer* must all be given
    and embeddings are computed on the fly.
    """
    if esm2_emb is not None:
        emb = esm2_emb if isinstance(esm2_emb, torch.Tensor) else torch.tensor(esm2_emb)
        return emb.to(device, dtype=torch.float32)

    if sequence is None or esm_model is None or tokenizer is None:
        raise ValueError(
            'Provide either esm2_emb or all of (sequence, esm_model, tokenizer).'
        )
    L = len(sequence)
    esm_model.eval()
    with torch.no_grad():
        inputs = tokenizer(sequence, return_tensors='pt').to(device)
        hidden = esm_model(**inputs).last_hidden_state[0]   # (L+2, 320)
    return hidden[1:L + 1].to(device, dtype=torch.float32)  # (L, 320)


@torch.no_grad()
def generate_structure(
    esm2_emb,
    flow_model,
    sequence:   Optional[str]   = None,
    esm_model                   = None,
    tokenizer                   = None,
    n_steps:    int             = 50,
    omega_deg:  float           = 180.0,
    output_pdb: Optional[str]   = None,
    device:     str             = 'cpu',
):
    """Generate a backbone structure using the flow model + NeRF.

    Pipeline
    --------
    1. Resolve ESM-2 embeddings (from cache or computed on the fly)
    2. Flow matching ODE: Gaussian noise → φ/ψ per residue (L, 2)
    3. NeRF reconstruction: φ/ψ → backbone coordinates (L, 4, 3)
    4. Optionally write a PDB file.

    Parameters
    ----------
    esm2_emb   : (L, 320) array or Tensor — precomputed ESM-2 hidden states
    flow_model : trained ConditionedFlow
    sequence   : str, optional — required only if esm2_emb is None
    esm_model  : ESM-2 model, optional — required only if esm2_emb is None
    tokenizer  : ESM-2 tokenizer, optional — required only if esm2_emb is None
    n_steps    : int — ODE integration steps (higher = more accurate)
    omega_deg  : float — peptide bond dihedral (default 180° = trans)
    output_pdb : str or None — if provided, writes coordinates to this path
    device     : 'cpu' or 'cuda'

    Returns
    -------
    phi_psi_deg : np.ndarray (L, 2) — generated torsion angles in degrees
    coords      : np.ndarray (L, 4, 3) — backbone [N, CA, C, O] in Å
    """
    emb = _resolve_esm2(esm2_emb, sequence, esm_model, tokenizer, device)
    L   = emb.shape[0]

    if sequence is None:
        sequence = 'X' * L   # placeholder — only needed for NeRF/PDB writing

    flow_model.eval()

    # Flow matching ODE
    x  = torch.randn(L, 2, device=device)
    ts = torch.linspace(0., 1., n_steps + 1, device=device)
    for i in range(n_steps):
        t_s = ts[i].view(1, 1).expand(L, 1)
        dt  = (ts[i + 1] - ts[i]).float()
        v_s = flow_model(t_s, x, emb)
        v_m = flow_model(t_s + dt / 2, x + v_s * dt / 2, emb)
        x   = x + dt * v_m
    phi_psi_deg = x.cpu().numpy() * 180.0    # (L, 2), degrees

    # NeRF reconstruction
    coords = build_backbone(sequence, phi_psi_deg, omega_deg=omega_deg)

    if output_pdb is not None:
        write_pdb(output_pdb, sequence, coords,
                  remark='ESM-2 Conditioned Flow Matching + NeRF')

    return phi_psi_deg, coords


@torch.no_grad()
def generate_and_refine(
    esm2_emb,
    flow_model,
    refiner_model,
    sequence:    Optional[str]  = None,
    esm_model                   = None,
    tokenizer                   = None,
    flow_steps:  int            = 50,
    omega_deg:   float          = 180.0,
    output_pdb:  Optional[str]  = None,
    device:      str            = 'cpu',
):
    """Full pipeline: ESM-2 embeddings → flow → NeRF → CoordRefiner → (PDB).

    Pipeline
    --------
    1. Resolve ESM-2 per-residue hidden states
    2. Flow matching ODE → φ/ψ angles
    3. NeRF → raw backbone coordinates (centred at Cα centroid)
    4. CoordRefiner transformer → refined coordinates
    5. Optionally write PDB.

    Parameters
    ----------
    esm2_emb      : (L, 320) array or Tensor — precomputed ESM-2 hidden states
    flow_model    : trained ConditionedFlow
    refiner_model : trained CoordRefiner
    sequence      : str, optional — required only if esm2_emb is None
    esm_model     : ESM-2 model, optional — required only if esm2_emb is None
    tokenizer     : ESM-2 tokenizer, optional — required only if esm2_emb is None
    flow_steps    : int — ODE integration steps for the flow model
    omega_deg     : float — peptide bond dihedral (default 180°)
    output_pdb    : str or None — if provided, writes the refined structure
    device        : 'cpu' or 'cuda'

    Returns
    -------
    nerf_coords    : np.ndarray (L, 4, 3) — raw NeRF output (centred)
    refined_coords : np.ndarray (L, 4, 3) — after transformer refinement
    phi_psi_deg    : np.ndarray (L, 2)    — generated torsion angles in degrees
    """
    emb = _resolve_esm2(esm2_emb, sequence, esm_model, tokenizer, device)
    L   = emb.shape[0]

    if sequence is None:
        sequence = 'X' * L

    flow_model.eval()
    refiner_model.eval()

    # Flow matching ODE → φ/ψ
    x  = torch.randn(L, 2, device=device)
    ts = torch.linspace(0., 1., flow_steps + 1, device=device)
    for i in range(flow_steps):
        t_s = ts[i].view(1, 1).expand(L, 1)
        dt  = (ts[i + 1] - ts[i]).float()
        v_s = flow_model(t_s, x, emb)
        v_m = flow_model(t_s + dt / 2, x + v_s * dt / 2, emb)
        x   = x + dt * v_m
    phi_psi_deg = x.cpu().numpy() * 180.0     # (L, 2)

    # NeRF reconstruction (centred)
    nerf_raw        = build_backbone(sequence, phi_psi_deg, omega_deg=omega_deg)
    nerf_centred, _ = center_coords(nerf_raw)

    # Transformer refinement
    noisy_t        = torch.tensor(nerf_centred.reshape(L, 12), dtype=torch.float32)
    refined_t      = refiner_model(noisy_t, emb.cpu())
    refined_coords = refined_t.numpy().reshape(L, 4, 3)

    if output_pdb is not None:
        write_pdb(output_pdb, sequence, refined_coords,
                  remark='ESM-2 Flow + NeRF + CoordRefiner')

    return nerf_centred, refined_coords, phi_psi_deg
