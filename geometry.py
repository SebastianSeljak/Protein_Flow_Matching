import torch
import numpy as np
import math

def torsion_to_cartesian(phi, psi, omega=None):
    """
    Differentiable reconstruction of protein backbone coordinates from torsion angles.
    
    Args:
        phi: Tensor of shape (B, L)
        psi: Tensor of shape (B, L)
        omega: Tensor of shape (B, L), optional (defaults to 180 degrees)
        
    Returns:
        coords: Tensor of shape (B, L*3, 3) representing (N, CA, C) atoms in sequence.
    """
    batch_size, seq_len = phi.shape
    device = phi.device
    
    if omega is None:
        omega = torch.full_like(phi, 180.0)
        
    # Convert to radians
    phi = phi * (np.pi / 180.0)
    psi = psi * (np.pi / 180.0)
    omega = omega * (np.pi / 180.0)
    
    # Standard bond lengths (Angstroms)
    b_n_ca = 1.46
    b_ca_c = 1.52
    b_c_n = 1.33
    
    # Standard bond angles (Radians)
    a_c_n_ca = 121.0 * (np.pi / 180.0)
    a_n_ca_c = 111.0 * (np.pi / 180.0)
    a_ca_c_n = 116.0 * (np.pi / 180.0)
    
    total_atoms = seq_len * 3
    
    # 1. Bond lengths r_i (distance between i-1 and i)
    # L[1]: N-CA (1.46), L[2]: CA-C (1.52), L[3]: C-N (1.33)
    # Repeated pattern: [1.33, 1.46, 1.52]
    lengths = torch.tensor([b_c_n, b_n_ca, b_ca_c], device=device).repeat(seq_len)
    
    # 2. Bond angles theta_i (angle between i-2, i-1, i)
    # A[2]: N-CA-C (111), A[3]: CA-C-N (116), A[4]: C-N-CA (121)
    # Repeated pattern: [a_ca_c_n, a_c_n_ca, a_n_ca_c]
    angles = torch.tensor([a_ca_c_n, a_c_n_ca, a_n_ca_c], device=device).repeat(seq_len)
    
    # 3. Torsions chi_i (dihedral between i-3, i-2, i-1, i)
    # i=3 (N2): psi1, i=4 (CA2): omega1, i=5 (C2): phi2
    # Create the full torsion array (B, 3L)
    torsions = torch.zeros(batch_size, total_atoms, device=device)
    if seq_len > 1:
        torsions[:, 3::3] = psi[:, :-1]   # psi1 at index 3, psi2 at index 6...
        torsions[:, 4::3] = omega[:, :-1] # omega1 at index 4, omega2 at index 7...
        torsions[:, 5::3] = phi[:, 1:]    # phi2 at index 5, phi3 at index 8...
    
    coords = torch.zeros(batch_size, total_atoms, 3, device=device)
    
    # Initialize first three atoms to define a coordinate system
    # Atom 0: N1 at origin
    # Atom 1: CA1 at (b_n_ca, 0, 0)
    # Atom 2: C1 in mapping plane
    coords[:, 0] = 0.0
    coords[:, 1, 0] = b_n_ca
    
    # C1 pos using bond angle a_n_ca_c
    coords[:, 2, 0] = b_n_ca - b_ca_c * math.cos(a_n_ca_c)
    coords[:, 2, 1] = b_ca_c * math.sin(a_n_ca_c)
    
    # Iterative NeRF for remaining atoms
    for i in range(3, total_atoms):
        r = lengths[i]
        theta = angles[i]
        chi = torsions[:, i]
        
        # Local coordinate of atom i relative to i-1
        p = torch.stack([
            torch.full_like(chi, r * math.cos(theta)),
            r * math.sin(theta) * torch.cos(chi),
            r * math.sin(theta) * torch.sin(chi)
        ], dim=1)
        
        # Vectors defining the frame
        v1 = coords[:, i-1] - coords[:, i-2]
        v2 = coords[:, i-2] - coords[:, i-3]
        
        n = torch.cross(v1, v2, dim=1)
        n = n / (torch.norm(n, dim=1, keepdim=True) + 1e-8)
        
        b = v1 / (torch.norm(v1, dim=1, keepdim=True) + 1e-8)
        m = torch.cross(n, b, dim=1)
        
        # Rotation matrix [m, n, b]
        rot = torch.stack([b, m, n], dim=2)
        
        coords[:, i] = torch.bmm(rot, p.unsqueeze(2)).squeeze(2) + coords[:, i-1]
        
    return coords
