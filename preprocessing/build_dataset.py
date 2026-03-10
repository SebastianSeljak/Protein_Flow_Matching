"""
build_dataset.py — Build a randomly sampled HDF5 protein dataset.

Reads the full pool of available protein codes (from a .txt file or an
existing HDF5), draws a random subset, downloads each from RCSB, extracts
features, and writes to a new HDF5 file.

HDF5 schema (same as preprocess_dataset.py)
--------------------------------------------
/proteins/{pid}/sequence     — bytes (ASCII)
/proteins/{pid}/true_coords  — (L, 4, 3) float32  [N, CA, C, O], Å
/proteins/{pid}/phi_psi      — (L, 2)    float32  degrees; NaN at terminals
/proteins/{pid}/esm2         — (L, 320)  float32

Usage
-----
# Sample 200 random proteins from a codes file:
python preprocessing/build_dataset.py --n 200 --codes protein_codes.txt

# Sample 500 from an existing HDF5 (useful for sub-setting):
python preprocessing/build_dataset.py --n 500 --source-h5 proteins.h5 --out subset_500.h5

# Reproducible sample + custom output name:
python preprocessing/build_dataset.py --n 100 --codes protein_codes.txt \\
    --seed 7 --out my_100.h5

# Quick smoke-test (5 proteins):
python preprocessing/build_dataset.py --n 5 --codes protein_codes.txt --quick-test
"""

import argparse
import datetime
import os
import pickle
import random
import sys
import time

import h5py
import math
import numpy as np
import requests
import torch
from Bio import PDB
from Bio.SeqUtils import seq1 as three_to_one
from tqdm import tqdm


# ── Constants ──────────────────────────────────────────────────────────────────

_AA_LETTERS      = 'ACDEFGHIKLMNPQRSTVWY'
_RCSB_URL        = 'https://files.rcsb.org/download/{pid}.pdb'
_DEFAULT_CODES   = 'protein_codes.txt'
_DEFAULT_ESM_MDL = 'models/esm2_8M.pkl'
_DEFAULT_ESM_TOK = 'models/esm2_8M_tokenizer.pkl'


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Build a randomly sampled HDF5 protein dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Source ────────────────────────────────────────────────────────────────
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--codes',      metavar='FILE',
                     help='Text file of protein codes (one per line or whitespace-separated)')
    src.add_argument('--source-h5',  metavar='FILE',
                     help='Existing HDF5 dataset — sample protein IDs from it '
                          '(copies feature data directly, no re-download needed)')

    p.add_argument('--n',            type=int, required=True,
                   help='Number of proteins to include in the output dataset')
    p.add_argument('--seed',         type=int, default=42,
                   help='Random seed for reproducible sampling')

    # ── ESM-2 (only needed when sourcing from --codes) ────────────────────────
    p.add_argument('--esm2-model',     default=_DEFAULT_ESM_MDL,
                   help='Pickled ESM-2 model (needed when --codes is used)')
    p.add_argument('--esm2-tokenizer', default=_DEFAULT_ESM_TOK,
                   help='Pickled ESM-2 tokenizer (needed when --codes is used)')

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument('--out',          default=None,
                   help='Output HDF5 path (auto-named as dataset_N_TIMESTAMP.h5 if omitted)')
    p.add_argument('--min-len',      type=int, default=5,
                   help='Discard proteins shorter than this after extraction')
    p.add_argument('--max-len',      type=int, default=2048,
                   help='Discard proteins longer than this after extraction')

    # ── Quick test ────────────────────────────────────────────────────────────
    p.add_argument('--quick-test',   action='store_true',
                   help='Cap --n at 5 and print extra diagnostics')

    return p.parse_args()


# ── Logging ────────────────────────────────────────────────────────────────────

def _log(msg: str):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


# ── PDB download ───────────────────────────────────────────────────────────────

def _download_pdb_text(pid: str) -> str | None:
    try:
        resp = requests.get(_RCSB_URL.format(pid=pid.upper()), timeout=20)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


# ── Feature extraction ─────────────────────────────────────────────────────────

def _extract_from_text(pdb_text: str):
    """Parse PDB text in memory via a temp file. Returns (sequence, coords, phi_psi) or Nones."""
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix='.pdb')
    try:
        with os.fdopen(fd, 'w') as fh:
            fh.write(pdb_text)
        return _extract_backbone_and_angles(tmp)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _extract_backbone_and_angles(pdb_path: str):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('s', pdb_path)
    ppb = PDB.PPBuilder()
    seq_list, coord_list, phi_psi_list = [], [], []

    for model in structure:
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
                    coord_list.append(np.stack([n, ca, c, o]))
                    seq_list.append(aa)
                    phi_psi_list.append([
                        math.degrees(phi) if phi is not None else float('nan'),
                        math.degrees(psi) if psi is not None else float('nan'),
                    ])
        break  # first model only

    if not coord_list:
        return None, None, None

    return (
        ''.join(seq_list),
        np.stack(coord_list).astype(np.float32),
        np.array(phi_psi_list, dtype=np.float32),
    )


def _compute_esm2(sequence: str, esm_model, tokenizer) -> np.ndarray:
    with torch.no_grad():
        inputs = tokenizer(sequence, return_tensors='pt')
        hidden = esm_model(**inputs).last_hidden_state[0]
    return hidden[1:len(sequence) + 1].numpy().astype(np.float32)


# ── HDF5 helpers ───────────────────────────────────────────────────────────────

def _already_in_h5(out_path: str, pid: str) -> bool:
    if not os.path.exists(out_path):
        return False
    with h5py.File(out_path, 'r') as hf:
        return pid in hf.get('proteins', {})


def _write_entry(out_path: str, pid: str, sequence: str,
                 coords: np.ndarray, phi_psi: np.ndarray, esm2: np.ndarray):
    with h5py.File(out_path, 'a') as hf:
        grp = hf.require_group('proteins').require_group(pid)
        if 'sequence' in grp:
            return  # already written
        grp.create_dataset('sequence',    data=sequence.encode('ascii'))
        grp.create_dataset('true_coords', data=coords,   compression='gzip')
        grp.create_dataset('phi_psi',     data=phi_psi,  compression='gzip')
        grp.create_dataset('esm2',        data=esm2,     compression='gzip')


def _copy_entry(src_h5: str, dst_h5: str, pid: str):
    """Copy one protein group from src to dst HDF5."""
    with h5py.File(src_h5, 'r') as src, h5py.File(dst_h5, 'a') as dst:
        dst_grp = dst.require_group('proteins')
        if pid not in dst_grp:
            src['proteins'].copy(pid, dst_grp)


# ── Source: codes file → download + extract ────────────────────────────────────

def _run_from_codes(sample: list, out_path: str, esm_model, tokenizer,
                    min_len: int, max_len: int):
    written, failed = 0, 0
    t0 = time.time()

    for pid in tqdm(sample, desc='Processing', dynamic_ncols=True):
        if _already_in_h5(out_path, pid):
            written += 1
            continue

        pdb_text = _download_pdb_text(pid)
        if pdb_text is None:
            _log(f'  SKIP {pid} — download failed')
            failed += 1
            continue

        sequence, coords, phi_psi = _extract_from_text(pdb_text)
        if sequence is None or not (min_len <= len(sequence) <= max_len):
            failed += 1
            continue

        try:
            esm2 = _compute_esm2(sequence, esm_model, tokenizer)
        except Exception as e:
            _log(f'  SKIP {pid} — ESM-2 error: {e}')
            failed += 1
            continue

        if esm2.shape[0] != len(sequence):
            failed += 1
            continue

        _write_entry(out_path, pid, sequence, coords, phi_psi, esm2)
        written += 1

    elapsed = time.time() - t0
    return written, failed, elapsed


# ── Source: existing HDF5 → direct copy ───────────────────────────────────────

def _run_from_h5(sample: list, source_h5: str, out_path: str,
                 min_len: int, max_len: int):
    written, skipped = 0, 0
    t0 = time.time()

    with h5py.File(source_h5, 'r') as src:
        src_grp = src['proteins']
        for pid in tqdm(sample, desc='Copying', dynamic_ncols=True):
            if _already_in_h5(out_path, pid):
                written += 1
                continue

            L = src_grp[pid]['phi_psi'].shape[0]
            if not (min_len <= L <= max_len):
                skipped += 1
                continue

            _copy_entry(source_h5, out_path, pid)
            written += 1

    elapsed = time.time() - t0
    return written, skipped, elapsed


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.quick_test:
        args.n = min(args.n, 5)
        _log('[ quick-test mode: capped at 5 proteins ]')

    random.seed(args.seed)

    # ── Resolve output path ───────────────────────────────────────────────────
    if args.out is None:
        ts       = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.out = f'dataset_{args.n}_{ts}.h5'
    _log(f'Output → {args.out}')

    # ── Build the candidate pool and draw random sample ───────────────────────
    if args.codes:
        with open(args.codes, 'r', encoding='utf-8-sig') as fh:
            pool = fh.read().split()
        _log(f'Pool: {len(pool)} codes from {args.codes!r}')
    else:  # --source-h5
        with h5py.File(args.source_h5, 'r') as hf:
            pool = sorted(hf['proteins'].keys())
        _log(f'Pool: {len(pool)} proteins from {args.source_h5!r}')

    if args.n > len(pool):
        _log(f'WARNING: requested {args.n} but only {len(pool)} available — using all.')
        args.n = len(pool)

    sample = random.sample(pool, args.n)
    _log(f'Sampled {len(sample)} proteins (seed={args.seed})')

    # ── How many already written (resume) ─────────────────────────────────────
    already = sum(1 for p in sample if _already_in_h5(args.out, p))
    if already:
        _log(f'Resuming: {already} proteins already in {args.out}')

    # ── Process ───────────────────────────────────────────────────────────────
    if args.codes:
        _log('Loading ESM-2 model and tokenizer...')
        with open(args.esm2_model, 'rb') as fh:
            esm_model = pickle.load(fh)
        with open(args.esm2_tokenizer, 'rb') as fh:
            tokenizer = pickle.load(fh)
        esm_model.eval()
        _log('ESM-2 8M ready.')

        written, failed, elapsed = _run_from_codes(
            sample, args.out, esm_model, tokenizer, args.min_len, args.max_len
        )
        _log(f'\nDone in {elapsed:.0f}s — written: {written}  failed: {failed}')
    else:
        written, skipped, elapsed = _run_from_h5(
            sample, args.source_h5, args.out, args.min_len, args.max_len
        )
        _log(f'\nDone in {elapsed:.0f}s — written: {written}  skipped (length): {skipped}')

    # ── Final summary ─────────────────────────────────────────────────────────
    with h5py.File(args.out, 'r') as hf:
        total = len(hf.get('proteins', {}))
    _log(f'Dataset contains {total} proteins → {args.out}')


if __name__ == '__main__':
    main()
