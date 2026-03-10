"""
preprocess_pipeline.py — Protein feature extraction pipeline.

Reads protein codes from a master text file (protein_codes.txt), downloads
each PDB file from RCSB, extracts features, and saves to either:
  - A pickle dict  (if total codes <= 1000)
  - An HDF5 file   (if total codes >  1000)

Features extracted per protein
--------------------------------
sequence     : str  (amino acid sequence)
true_coords  : (L, 4, 3) float32  — backbone atoms [N, CA, C, O], Ångstroms
phi_psi      : (L, 2)    float32  — torsion angles in degrees; NaN at terminals
esm2         : (L, 320)  float32  — ESM-2 8M per-residue hidden states

Temporary PDB files are written to /tmp and deleted immediately after parsing.

Usage
-----
python preprocess_pipeline.py [--codes protein_codes.txt]
                               [--out proteins.h5 | protein_features.pkl]
                               [--esm2-model models/esm2_8M.pkl]
                               [--esm2-tokenizer models/esm2_8M_tokenizer.pkl]
                               [--batch-size 100]
                               [--keep-tmp]
"""

import argparse
import io
import math
import os
import pickle
import tempfile

import h5py # type: ignore
import numpy as np
import requests
import torch
from Bio import PDB # type: ignore
from Bio.SeqUtils import seq1 as three_to_one # type: ignore
from tqdm import tqdm


# ── Constants ──────────────────────────────────────────────────────────────────

_AA_LETTERS      = 'ACDEFGHIKLMNPQRSTVWY'
_RCSB_URL        = 'https://files.rcsb.org/download/{pid}.pdb'
_DEFAULT_CODES   = 'protein_codes.txt'
_DEFAULT_ESM_MDL = 'models/esm2_8M.pkl'
_DEFAULT_ESM_TOK = 'models/esm2_8M_tokenizer.pkl'
_SMALL_THRESHOLD = 1000   # use pickle below this, HDF5 at or above


# ── PDB download ───────────────────────────────────────────────────────────────

def _download_pdb(pid: str) -> str | None:
    """Download PDB text for *pid* from RCSB. Returns None on failure."""
    url = _RCSB_URL.format(pid=pid.upper())
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


# ── Backbone + torsion extraction ──────────────────────────────────────────────

def _extract_backbone_and_angles(pdb_path: str):
    """Extract backbone coordinates and φ/ψ angles from a PDB file.

    Returns
    -------
    sequence : str of length L  (standard AA only)
    coords   : np.ndarray (L, 4, 3) float32 — [N, CA, C, O]
    phi_psi  : np.ndarray (L, 2)    float32 — degrees; NaN where undefined
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('s', pdb_path)
    ppb = PDB.PPBuilder()

    seq_list, coord_list, phi_psi_list = [], [], []

    for model in structure:                          # first model only
        for chain in model:
            for poly in ppb.build_peptides(chain):
                residues       = list(poly)
                dihedral_pairs = poly.get_phi_psi_list()

                for residue, (phi, psi) in zip(residues, dihedral_pairs):
                    try:
                        n  = residue['N'].get_vector().get_array()
                        ca = residue['CA'].get_vector().get_array()
                        c  = residue['C'].get_vector().get_array()
                        o  = residue['O'].get_vector().get_array()
                        aa = three_to_one(residue.resname)
                    except Exception:
                        continue
                    if aa not in _AA_LETTERS:
                        continue

                    phi_deg = math.degrees(phi) if phi is not None else float('nan')
                    psi_deg = math.degrees(psi) if psi is not None else float('nan')

                    coord_list.append(np.stack([n, ca, c, o]))
                    seq_list.append(aa)
                    phi_psi_list.append([phi_deg, psi_deg])

        break  # NMR ensembles: use first model only

    if not coord_list:
        return None, None, None

    sequence = ''.join(seq_list)
    coords   = np.stack(coord_list).astype(np.float32)   # (L, 4, 3)
    phi_psi  = np.array(phi_psi_list, dtype=np.float32)  # (L, 2)
    return sequence, coords, phi_psi


# ── ESM-2 embedding ────────────────────────────────────────────────────────────

def _compute_esm2(sequence: str, esm_model, tokenizer) -> np.ndarray:
    """Return (L, 320) float32 ESM-2 hidden states for *sequence*."""
    with torch.no_grad():
        inputs = tokenizer(sequence, return_tensors='pt')
        hidden = esm_model(**inputs).last_hidden_state[0]  # (L+2, 320)
    return hidden[1:len(sequence) + 1].numpy().astype(np.float32)  # (L, 320)


# ── Per-protein processing ─────────────────────────────────────────────────────

def _process_protein(pid: str, esm_model, tokenizer, keep_tmp: bool = False):
    """Download, parse, and featurise one protein.

    Returns a dict with keys sequence / true_coords / phi_psi / esm2,
    or None on any failure.
    """
    # 1. Download PDB content
    pdb_text = _download_pdb(pid)
    if pdb_text is None:
        return None

    # 2. Write to a temporary file, parse, then delete
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.pdb')
    try:
        with os.fdopen(tmp_fd, 'w') as fh:
            fh.write(pdb_text)

        sequence, coords, phi_psi = _extract_backbone_and_angles(tmp_path)
    finally:
        if not keep_tmp and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if sequence is None or len(sequence) < 5:
        return None

    # 3. ESM-2 embeddings
    try:
        esm2 = _compute_esm2(sequence, esm_model, tokenizer)
    except Exception as e:
        print(f'\n  {pid}: ESM-2 failed — {e}')
        return None

    if esm2.shape[0] != len(sequence):
        return None

    return {
        'sequence':    sequence,
        'true_coords': coords,    # (L, 4, 3)
        'phi_psi':     phi_psi,   # (L, 2)
        'esm2':        esm2,      # (L, 320)
    }


# ── HDF5 helpers ───────────────────────────────────────────────────────────────

def _write_hdf5(out_path: str, pid: str, data: dict):
    """Append one protein entry to an HDF5 file (create if absent)."""
    with h5py.File(out_path, 'a') as hf:
        proteins_grp = hf.require_group('proteins')
        if pid in proteins_grp:
            return  # already written (resume-safe)
        grp = proteins_grp.create_group(pid)
        grp.create_dataset('sequence',    data=data['sequence'].encode('ascii'))
        grp.create_dataset('true_coords', data=data['true_coords'], compression='gzip')
        grp.create_dataset('phi_psi',     data=data['phi_psi'],     compression='gzip')
        grp.create_dataset('esm2',        data=data['esm2'],        compression='gzip')


def _pid_in_hdf5(out_path: str, pid: str) -> bool:
    if not os.path.exists(out_path):
        return False
    with h5py.File(out_path, 'r') as hf:
        return pid in hf.get('proteins', {})


# ── Main pipeline ──────────────────────────────────────────────────────────────

def preprocess(
    codes_path:     str  = _DEFAULT_CODES,
    out_path:       str  | None = None,
    esm_model_path: str  = _DEFAULT_ESM_MDL,
    esm_tok_path:   str  = _DEFAULT_ESM_TOK,
    batch_size:     int  = 100,
    keep_tmp:       bool = False,
    total_limit:     int  | None = None,
):
    # ── Load codes ────────────────────────────────────────────────────────────
    with open(codes_path, 'r', encoding='utf-8-sig') as fh:
        codes = fh.read().split()
    print(f'Loaded {len(codes)} protein codes from {codes_path!r}.')
    if total_limit is None:
        total_limit = len(codes)
    use_hdf5 = total_limit >= _SMALL_THRESHOLD

    # ── Resolve output path ───────────────────────────────────────────────────
    if out_path is None:
        out_path = 'proteins.h5' if use_hdf5 else 'protein_features.pkl'

    storage_label = f'HDF5  → {out_path}' if use_hdf5 else f'pickle → {out_path}'
    print(f'Storage mode : {storage_label}')

    # ── Load ESM-2 ────────────────────────────────────────────────────────────
    print('Loading ESM-2 model and tokenizer...')
    with open(esm_model_path, 'rb') as fh:
        esm_model = pickle.load(fh)
    with open(esm_tok_path, 'rb') as fh:
        tokenizer = pickle.load(fh)
    esm_model.eval()
    print('ESM-2 8M loaded.')

    # ── Resume state ──────────────────────────────────────────────────────────
    if use_hdf5:
        # For HDF5: determine already-written PIDs from the file itself
        already_done = set()
        if os.path.exists(out_path):
            with h5py.File(out_path, 'r') as hf:
                already_done = set(hf.get('proteins', {}).keys())
        all_features = None   # not used in HDF5 mode
    else:
        # For pickle: load existing dict
        if os.path.exists(out_path):
            with open(out_path, 'rb') as fh:
                all_features = pickle.load(fh)
            print(f'Resuming — {len(all_features)} proteins already processed.')
        else:
            all_features = {}
        already_done = set(all_features.keys())

    to_process = [c for c in codes if c not in already_done][:total_limit]
    print(f'Remaining to process: {len(to_process)}')

    # ── Processing loop ───────────────────────────────────────────────────────
    written, failed = 0, 0
    failed_pids = []

    def _flush_pickle():
        with open(out_path, 'wb') as fh:
            pickle.dump(all_features, fh)

    try:
        for i, pid in enumerate(tqdm(to_process[:total_limit], desc='Processing'), start=1):
            result = _process_protein(pid, esm_model, tokenizer, keep_tmp=keep_tmp)

            if result is None:
                failed += 1
                failed_pids.append(pid)
            else:
                written += 1
                if use_hdf5:
                    _write_hdf5(out_path, pid, result)
                else:
                    all_features[pid] = result #type: ignore

            # Periodic checkpoint save (pickle only; HDF5 writes per-entry)
            if not use_hdf5 and i % batch_size == 0:
                _flush_pickle()

    except KeyboardInterrupt:
        print('\nInterrupted — saving progress...')
    finally:
        if not use_hdf5:
            _flush_pickle()

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(all_features) if not use_hdf5 else (len(already_done) + written) #type: ignore
    print(f'\nDone.')
    print(f'  Written  : {written}')
    print(f'  Skipped  : {len(already_done)} (already processed)')
    print(f'  Failed   : {failed}')
    if failed_pids:
        preview = failed_pids[:20]
        suffix  = '...' if len(failed_pids) > 20 else ''
        print(f'  Failed IDs: {preview}{suffix}')
    print(f'  Total in dataset : {total}')
    print(f'\nDataset saved → {out_path}')


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download PDB files and extract protein features into pickle or HDF5.'
    )
    parser.add_argument('--codes',          default=_DEFAULT_CODES,
                        help='Path to protein_codes.txt (default: protein_codes.txt)')
    parser.add_argument('--out',            default=None,
                        help='Output file path (auto-selected if omitted)')
    parser.add_argument('--esm2-model',     default=_DEFAULT_ESM_MDL,
                        help='Pickled ESM-2 model (default: models/esm2_8M.pkl)')
    parser.add_argument('--esm2-tokenizer', default=_DEFAULT_ESM_TOK,
                        help='Pickled ESM-2 tokenizer (default: models/esm2_8M_tokenizer.pkl)')
    parser.add_argument('--batch-size',     type=int, default=100,
                        help='Pickle checkpoint interval (default: 100)')
    parser.add_argument('--keep-tmp',       action='store_true',
                        help='Keep temporary PDB files instead of deleting them')
    parser.add_argument('--total-limit',    type=int, default=None,
                        help='Limit total proteins processed (for testing; default: no limit)')
    args = parser.parse_args()

    preprocess(
        codes_path     = args.codes,
        out_path       = args.out,
        esm_model_path = args.esm2_model,
        esm_tok_path   = args.esm2_tokenizer,
        batch_size     = args.batch_size,
        keep_tmp       = args.keep_tmp,
        total_limit    = args.total_limit,
    )