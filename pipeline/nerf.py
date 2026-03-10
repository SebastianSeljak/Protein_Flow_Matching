"""
nerf.py — NeRF backbone reconstruction and PDB I/O.

Public API
----------
build_backbone(sequence, phi_psi_deg, omega_deg=180.0) -> np.ndarray (L, 4, 3)
write_pdb(filepath, sequence, coords, chain_id='A', remark='...')

Constants
---------
BL  — standard backbone bond lengths (Å)
BA  — standard backbone bond angles (degrees)
ONE_TO_THREE — one-letter → three-letter amino acid code mapping
"""

import math
import numpy as np

# ── Standard backbone geometry (AMBER/CHARMM values) ─────────────────────────

BL = {           # Bond lengths, Angstroms
    'n_ca': 1.458,   # N–CA
    'ca_c': 1.525,   # CA–C
    'c_n' : 1.329,   # C–N  (peptide bond)
    'c_o' : 1.229,   # C=O  (carbonyl)
}

BA = {           # Bond angles, degrees
    'n_ca_c': 111.2,   # N–CA–C
    'ca_c_n': 116.2,   # CA–C–N  (used to place next N)
    'c_n_ca': 121.7,   # C–N–CA  (used to place next CA)
    'ca_c_o': 120.8,   # CA–C=O  (used to place O)
}

ONE_TO_THREE = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
    'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
    'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
}

_BACKBONE_ATOMS = ['N', 'CA', 'C', 'O']


# ── NeRF core ─────────────────────────────────────────────────────────────────

def nerf(a, b, c, bond_length: float, bond_angle_deg: float, dihedral_deg: float):
    """Place atom D given three anchor atoms A, B, C.

    The new atom satisfies:
      |C–D|               = bond_length
      angle(B, C, D)      = bond_angle_deg
      dihedral(A, B, C, D) = dihedral_deg

    Parameters
    ----------
    a, b, c        : array-like (3,) — anchor atom coordinates
    bond_length    : float — |C–D| in Å
    bond_angle_deg : float — bond angle B–C–D in degrees
    dihedral_deg   : float — dihedral A–B–C–D in degrees

    Returns
    -------
    d : np.ndarray (3,) — coordinates of the new atom D
    """
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    theta = math.radians(bond_angle_deg)
    xi    = math.radians(dihedral_deg)

    bc_hat = c - b
    bc_hat /= np.linalg.norm(bc_hat)

    n = np.cross(b - a, bc_hat)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-8:
        # degenerate (A, B, C collinear) — choose an arbitrary perpendicular
        perp = np.array([0., 0., 1.]) if abs(bc_hat[2]) < 0.9 else np.array([1., 0., 0.])
        n = np.cross(bc_hat, perp)
        n /= np.linalg.norm(n)
    else:
        n /= n_norm

    m = np.cross(bc_hat, n)

    return c + bond_length * (
        -math.cos(theta) * bc_hat
        + math.sin(theta) * math.cos(xi) * m
        + math.sin(theta) * math.sin(xi) * n
    )


def build_backbone(sequence: str, phi_psi_deg: np.ndarray, omega_deg: float = 180.0) -> np.ndarray:
    """Reconstruct backbone 3D coordinates from φ/ψ torsion angles via NeRF.

    Atoms are placed in order N → CA → C → O for each residue.  The first
    three atoms (N₀, CA₀, C₀) are seeded in a local Cartesian frame; all
    subsequent atoms are determined by the torsion angles.

    Torsion angle mapping
    ---------------------
    N_i   : dihedral N_{i-1}–CA_{i-1}–C_{i-1}–N_i  = ψ_{i-1}
    CA_i  : dihedral CA_{i-1}–C_{i-1}–N_i–CA_i      = ω  (≈ 180° trans)
    C_i   : dihedral C_{i-1}–N_i–CA_i–C_i           = φ_i
    O_i   : dihedral N_i–CA_i–C_i–O_i               ≈ ψ_i + 180°

    Parameters
    ----------
    sequence    : str, amino acid sequence of length L
    phi_psi_deg : np.ndarray (L, 2), columns [φ, ψ] in degrees
    omega_deg   : float, peptide bond dihedral (default 180° = trans)

    Returns
    -------
    coords : np.ndarray (L, 4, 3), float64
        Backbone atom coordinates in Å; axis-1 order: [N, CA, C, O]
    """
    L = len(sequence)
    if phi_psi_deg.shape != (L, 2):
        raise ValueError(f'phi_psi_deg must be ({L}, 2), got {phi_psi_deg.shape}')

    coords = np.zeros((L, 4, 3), dtype=np.float64)

    # ── Seed residue 0: place N₀, CA₀, C₀ in a local frame ──────────────────
    ang = math.radians(BA['n_ca_c'])
    coords[0, 0] = [0., 0., 0.]
    coords[0, 1] = [BL['n_ca'], 0., 0.]
    coords[0, 2] = coords[0, 1] + BL['ca_c'] * np.array([
        math.cos(math.pi - ang), math.sin(math.pi - ang), 0.
    ])
    coords[0, 3] = nerf(
        coords[0, 0], coords[0, 1], coords[0, 2],
        BL['c_o'], BA['ca_c_o'], phi_psi_deg[0, 1] + 180.,
    )

    # ── Build residues 1 … L-1 ───────────────────────────────────────────────
    for i in range(1, L):
        phi_i    = phi_psi_deg[i, 0]
        psi_i    = phi_psi_deg[i, 1]
        psi_prev = phi_psi_deg[i - 1, 1]

        n_p, ca_p, c_p = coords[i - 1, 0], coords[i - 1, 1], coords[i - 1, 2]

        n_i  = nerf(n_p,  ca_p, c_p,  BL['c_n'],  BA['ca_c_n'], psi_prev)
        ca_i = nerf(ca_p, c_p,  n_i,  BL['n_ca'], BA['c_n_ca'], omega_deg)
        c_i  = nerf(c_p,  n_i,  ca_i, BL['ca_c'], BA['n_ca_c'], phi_i)
        o_i  = nerf(n_i,  ca_i, c_i,  BL['c_o'],  BA['ca_c_o'], psi_i + 180.)

        coords[i] = [n_i, ca_i, c_i, o_i]

    return coords


# ── PDB I/O ───────────────────────────────────────────────────────────────────

def write_pdb(
    filepath: str,
    sequence: str,
    coords: np.ndarray,
    chain_id: str = 'A',
    remark: str = 'Generated by NeRF backbone builder',
) -> None:
    """Write backbone-only (N, CA, C, O) coordinates to a PDB file.

    Parameters
    ----------
    filepath  : output path, e.g. 'output.pdb'
    sequence  : str of length L (one-letter amino acid codes)
    coords    : np.ndarray (L, 4, 3) — atom order [N, CA, C, O], in Å
    chain_id  : single-character chain identifier (default 'A')
    remark    : written to the first REMARK line
    """
    if len(sequence) != len(coords):
        raise ValueError(f'sequence length {len(sequence)} != coords length {len(coords)}')

    lines  = [f'REMARK  {remark}\n']
    serial = 1

    for res_idx, (aa, res_coords) in enumerate(zip(sequence, coords)):
        res_num  = res_idx + 1
        res_name = ONE_TO_THREE.get(aa, 'UNK')

        for atom_name, (x, y, z) in zip(_BACKBONE_ATOMS, res_coords):
            name_field = f' {atom_name:<3s}' if len(atom_name) < 4 else atom_name
            lines.append(
                f'ATOM  {serial:5d} {name_field:<4s}{res_name:>3s} {chain_id}'
                f'{res_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00'
                f'          {atom_name[0]:>2s}\n'
            )
            serial += 1

    lines.append('END\n')

    with open(filepath, 'w') as fh:
        fh.writelines(lines)

    print(f'Wrote {serial - 1} atoms ({len(sequence)} residues) → {filepath}')
