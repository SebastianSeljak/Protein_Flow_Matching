import os
import io
import math
import requests
import numpy as np
import pickle
from Bio import PDB
from tqdm import tqdm

def get_ramachandran_coordinates(structure):
    """
    Extracts phi (x) and psi (y) angles from a Bio.PDB Structure object.
    """
    ramachandran_data = {"x": [], "y": []}
    ppb = PDB.PPBuilder()
    for model in structure:
        for chain in model:
            for poly in ppb.build_peptides(chain):
                phi_psi = poly.get_phi_psi_list()
                for phi, psi in phi_psi:
                    if phi is not None and psi is not None:
                        ramachandran_data["x"].append(math.degrees(phi))
                        ramachandran_data["y"].append(math.degrees(psi))
    
    if not ramachandran_data["x"]:
        return np.array([])
    
    return np.column_stack((np.array(ramachandran_data["x"]), np.array(ramachandran_data["y"])))

def extract_sequence(structure):
    """
    Extracts the amino acid sequence from a Bio.PDB Structure object.
    """
    ppb = PDB.PPBuilder()
    full_sequence = ""
    for model in structure:
        for chain in model:
            for pp in ppb.build_peptides(chain):
                full_sequence += str(pp.get_sequence())
    return full_sequence

def download_pdb_content(pdb_id):
    """
    Downloads PDB content as a string.
    """
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception:
        return None

def process_protein(pdb_id):
    """
    Downloads and extracts features in-memory.
    """
    content = download_pdb_content(pdb_id)
    if not content:
        return None
    
    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, io.StringIO(content))
        
        coords = get_ramachandran_coordinates(structure)
        sequence = extract_sequence(structure)
        
        if len(sequence) == 0 or coords.size == 0:
            return None
            
        return {
            "sequence": sequence,
            "coords": coords
        }
    except Exception as e:
        print(f"Error parsing {pdb_id}: {e}")
        return None

def main():
    # Load codes
    with open("protein_codes.txt", "r", encoding="utf-8-sig") as f:
        codes = f.read().split()
    
    print(f"Loaded {len(codes)} protein codes.")
    
    output_file = "protein_features.pkl"
    
    # Load existing progress if any
    if os.path.exists(output_file):
        with open(output_file, "rb") as f:
            all_features = pickle.load(f)
        print(f"Resuming from {len(all_features)} already processed proteins.")
    else:
        all_features = {}

    batch_size = 100
    count = 0
    
    # Process codes (skipping already processed)
    to_process = [c for c in codes if c not in all_features]
    print(f"Remaining to process: {len(to_process)}")
    
    try:
        for pdb_id in tqdm(to_process, desc="Processing Proteins"):
            result = process_protein(pdb_id)
            if result:
                all_features[pdb_id] = result
            
            count += 1
            # Save every batch_size to avoid losing progress
            if count % batch_size == 0:
                with open(output_file, "wb") as f:
                    pickle.dump(all_features, f)
                    
    except KeyboardInterrupt:
        print("\nInterrupted. Saving progress...")
    finally:
        # Final save
        with open(output_file, "wb") as f:
            pickle.dump(all_features, f)
        print(f"Processing complete. Total proteins in feature set: {len(all_features)}")

if __name__ == "__main__":
    main()
