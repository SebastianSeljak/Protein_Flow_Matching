"""
geometry.py — Geometric utilities for protein backbone coordinates.

Public API
----------
center_coords(coords)                          -> (centred_coords, centroid)
kabsch_rotation(P_ca, Q_ca)                    -> R (3×3 rotation matrix)
kabsch_align(noisy_coords, true_coords)        -> aligned noisy coords
ca_rmsd(pred, true)                            -> float (Å)
"""

import numpy as np


def center_coords(coords: np.ndarray):
    """Translate coordinates so the Cα centroid is at the origin.

    Parameters
    ----------
    coords : np.ndarray (L, 4, 3) — backbone atoms [N, CA, C, O]

    Returns
    -------
    centred  : np.ndarray (L, 4, 3) — centred coordinates
    centroid : np.ndarray (3,)      — original Cα centroid
    """
    centroid = coords[:, 1].mean(axis=0)   # mean of all Cα positions
    return coords - centroid[None, None, :], centroid


def kabsch_rotation(P_ca: np.ndarray, Q_ca: np.ndarray) -> np.ndarray:
    """Compute the rotation matrix R that minimises Cα RMSD between P and Q.

    Both inputs must already be centred at the origin.

    Parameters
    ----------
    P_ca : np.ndarray (L, 3) — Cα coordinates of the structure to rotate
    Q_ca : np.ndarray (L, 3) — Cα coordinates of the reference structure

    Returns
    -------
    R : np.ndarray (3, 3) — rotation matrix s.t. (P @ R.T) ≈ Q
    """
    H = P_ca.T @ Q_ca
    U, _, Vt = np.linalg.svd(H)
    # Correct for reflections (det = -1 means a reflection was found)
    d = np.linalg.det(Vt.T @ U.T)
    return Vt.T @ np.diag([1., 1., d]) @ U.T


def kabsch_align(noisy_coords: np.ndarray, true_coords: np.ndarray) -> np.ndarray:
    """Rigidly align noisy_coords to true_coords via Cα Kabsch superposition.

    Both inputs should already be centred at their own Cα centroid.

    Parameters
    ----------
    noisy_coords : np.ndarray (L, 4, 3) — coordinates to rotate
    true_coords  : np.ndarray (L, 4, 3) — reference (not modified)

    Returns
    -------
    aligned : np.ndarray (L, 4, 3) — noisy_coords after optimal rotation
    """
    R = kabsch_rotation(noisy_coords[:, 1], true_coords[:, 1])
    return (noisy_coords.reshape(-1, 3) @ R.T).reshape(noisy_coords.shape)


def ca_rmsd(pred: np.ndarray, true: np.ndarray) -> float:
    """Cα RMSD in Å after Kabsch alignment.

    Handles both (L, 4, 3) full-backbone and (L, 3) Cα-only arrays.

    Parameters
    ----------
    pred : np.ndarray (L, 4, 3) or (L, 3)
    true : np.ndarray (L, 4, 3) or (L, 3)

    Returns
    -------
    rmsd : float, in Å
    """
    pc = pred[:, 1] if pred.ndim == 3 else pred
    tc = true[:, 1] if true.ndim == 3 else true

    pc_c = pc - pc.mean(axis=0)
    tc_c = tc - tc.mean(axis=0)

    R          = kabsch_rotation(pc_c, tc_c)
    pc_aligned = pc_c @ R.T

    return float(np.sqrt(((pc_aligned - tc_c) ** 2).sum(axis=-1).mean()))
