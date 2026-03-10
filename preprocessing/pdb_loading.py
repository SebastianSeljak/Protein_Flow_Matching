from Bio import PDB
from Bio.SeqUtils import seq1 as three_to_one
import math
import os
import requests
import numpy as np


def get_residue_dihedrals(file_path):
    """
    Extracts phi/psi dihedral angles paired with their amino acid identity.

    Interior residues only — the first residue of each chain has no phi and
    the last has no psi, so both are excluded explicitly.

    Returns a list of dicts:
        [{'aa': 'A', 'phi': -60.1, 'psi': -45.3}, ...]
    where 'aa' is the one-letter amino acid code and angles are in degrees.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('input_structure', file_path)

    records = []
    for model in structure:
        for chain in model:
            for poly in PDB.PPBuilder().build_peptides(chain):
                residues = list(poly)
                phi_psi  = poly.get_phi_psi_list()

                for residue, (phi, psi) in zip(residues, phi_psi):
                    # Skip first residue (phi=None) and last residue (psi=None)
                    if phi is None or psi is None:
                        continue
                    try:
                        aa = three_to_one(residue.resname)
                    except KeyError:
                        continue  # skip non-standard residues
                    records.append({
                        'aa':  aa,
                        'phi': math.degrees(phi),
                        'psi': math.degrees(psi),
                    })
    return records


def get_residues_with_positions(file_path):
    """
    Returns the full amino acid sequence and per-interior-residue dihedral records
    that include each residue's index in the full sequence.

    The index is required to extract the matching per-residue hidden state from
    ESM-2: hidden_states[seq_pos + 1] (offset by 1 for the CLS token).

    First and last residues of each polypeptide are excluded because they lack
    one of the two dihedral angles.

    Returns:
        full_sequence (str): the complete concatenated protein sequence
        records (list of dicts):
            [{'seq_pos': int, 'aa': str, 'phi': float, 'psi': float}, ...]
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('input_structure', file_path)

    full_sequence = ''
    records = []

    for model in structure:
        for chain in model:
            for poly in PDB.PPBuilder().build_peptides(chain):
                offset   = len(full_sequence)
                residues = list(poly)
                phi_psi  = poly.get_phi_psi_list()

                for local_i, (residue, (phi, psi)) in enumerate(zip(residues, phi_psi)):
                    if phi is None or psi is None:
                        continue
                    try:
                        aa = three_to_one(residue.resname)
                    except KeyError:
                        continue
                    records.append({
                        'seq_pos': offset + local_i,
                        'aa':      aa,
                        'phi':     math.degrees(phi),
                        'psi':     math.degrees(psi),
                    })
                full_sequence += str(poly.get_sequence())
        break  # use first model only (relevant for NMR multi-model files)

    return full_sequence, records


def get_ramachandran_coordinates(file_path):
    """
    Extracts only the phi (x) and psi (y) angles from a PDB file.
    Returns a dictionary with lists of x and y values.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('input_structure', file_path)
    
    ramachandran_data = {"x": [], "y": []}

    for model in structure:
        for chain in model:
            polypeptides = PDB.PPBuilder().build_peptides(chain)
            for poly in polypeptides:
                phi_psi = poly.get_phi_psi_list()
                
                for i, (phi, psi) in enumerate(phi_psi):
                    # Only append if both angles are present (not None)
                    if phi is not None and psi is not None:
                        # Convert radians to degrees
                        ramachandran_data["x"].append(math.degrees(phi))
                        ramachandran_data["y"].append(math.degrees(psi))
    ramachandran_data["x"] = np.array(ramachandran_data["x"])
    ramachandran_data["y"] = np.array(ramachandran_data["y"])
    coords = np.column_stack((ramachandran_data["x"], ramachandran_data["y"])) 
    return coords

def batch_download_pdb(pdb_ids, download_dir="PDBs"):
    """
    Download PDB files for a list of IDs, skipping ones already on disk.

    Returns:
        successful (list[str]): PDB IDs that were downloaded or already present.
        failed     (list[str]): PDB IDs that could not be fetched.
    """
    os.makedirs(download_dir, exist_ok=True)
    successful, failed = [], []
    for i, pdb_id in enumerate(pdb_ids):
        path = download_pdb_file(pdb_id, download_dir=download_dir)
        if path:
            successful.append(pdb_id)
        else:
            failed.append(pdb_id)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(pdb_ids)} processed  "
                  f"({len(successful)} ok, {len(failed)} failed)")
    print(f"Done. {len(successful)} downloaded, {len(failed)} failed.")
    return successful, failed


def download_pdb_file(pdb_id, download_dir="."):
    """
    Downloads the .pdb file directly from RCSB using the PDB ID.
    """
    pdb_id = pdb_id.upper()
    pdb_file = f"{pdb_id}.pdb"
    file_path = os.path.join(download_dir, pdb_file)
    
    # Check if file already exists to save time/bandwidth
    if os.path.exists(file_path):
        print(f"File {pdb_file} already exists. Using local copy.")
        return file_path

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    try:
        print(f"Downloading {pdb_id}...")
        response = requests.get(url)
        response.raise_for_status() # Raises error for 404 (ID not found)
        
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Download complete: {file_path}")
        return file_path
    except requests.exceptions.HTTPError:
        print(f"Error: PDB ID '{pdb_id}' not found on RCSB.")
        return None
    except Exception as e:
        print(f"Error downloading PDB: {e}")
        return None
    
def extract_sequence(file_path):
    """
    Extracts the amino acid sequence from the PDB file.
    Uses PPBuilder to ensure it matches the indices used in the Ramachandran calculation.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('seq_extraction', file_path)
    ppb = PDB.PPBuilder()
    
    full_sequence = ""
    
    for model in structure:
        for chain in model:
            # build_peptides filters out waters and heteroatoms automatically
            for pp in ppb.build_peptides(chain):
                full_sequence += str(pp.get_sequence())
                
    return full_sequence

def extract_backbone_coords(pdb_path):
    """Extract N, CA, C, O coordinates for all residues in a PDB file.

    Uses PPBuilder for peptide segmentation — the same logic used by
    ``get_residues_with_positions`` — so residue indices are consistent
    across the pipeline.

    Only standard amino acids (one-letter codes in ACDEFGHIKLMNPQRSTVWY)
    are included; HETATM and non-standard residues are silently skipped.
    Only the first MODEL record is processed (relevant for NMR ensembles).

    Parameters
    ----------
    pdb_path : str — path to the PDB file

    Returns
    -------
    sequence : str of length L, or None on failure
    coords   : np.ndarray (L, 4, 3) float32, atom order [N, CA, C, O], or None
    """
    _AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('s', pdb_path)
    ppb = PDB.PPBuilder()
    seq_list, coord_list = [], []
    for model in structure:
        for chain in model:
            for poly in ppb.build_peptides(chain):
                for residue in poly:
                    try:
                        n  = residue['N'].get_vector().get_array()
                        ca = residue['CA'].get_vector().get_array()
                        c  = residue['C'].get_vector().get_array()
                        o  = residue['O'].get_vector().get_array()
                        aa = three_to_one(residue.resname)
                    except (KeyError, Exception):
                        continue
                    if aa not in _AA_LETTERS:
                        continue
                    coord_list.append(np.stack([n, ca, c, o]))
                    seq_list.append(aa)
        break  # first model only
    if not coord_list:
        return None, None
    return ''.join(seq_list), np.stack(coord_list).astype(np.float32)


def process_pdb_pipeline(pdb_id, filepath="."):
    """
    Takes a PDB ID, downloads the file, calculates angles, and extracts sequence.
    """
    # 1. Download PDB
    file_path = download_pdb_file(pdb_id, download_dir=filepath)
    if not file_path:
        return None

    # 2. Get Ramachandran Coordinates (legacy coords array)
    coords = get_ramachandran_coordinates(file_path)

    # 3. Get per-residue dihedral records (aa + phi + psi, interior residues only)
    residue_dihedrals = get_residue_dihedrals(file_path)

    # 4. Extract Sequence
    sequence = extract_sequence(file_path)

    # 5. Package Results
    result = {
        "pdb_id": pdb_id,
        "sequence": sequence,
        "ramachandran": coords,
        "residue_dihedrals": residue_dihedrals,
    }
    
    return result

