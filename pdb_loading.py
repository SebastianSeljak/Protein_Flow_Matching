from Bio import PDB
import math
import os
import requests
import numpy as np

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

def process_pdb_pipeline(pdb_id):
    """
    Takes a PDB ID, downloads the file, calculates angles, and extracts sequence.
    """
    # 1. Download PDB
    file_path = download_pdb_file(pdb_id)
    if not file_path:
        return None

    # 2. Get Ramachandran Coordinates
    coords = get_ramachandran_coordinates(file_path)

    # 3. Extract Sequence
    sequence = extract_sequence(file_path)

    # 4. Package Results
    result = {
        "pdb_id": pdb_id,
        "sequence": sequence,
        "ramachandran": coords
    }
    
    return result

