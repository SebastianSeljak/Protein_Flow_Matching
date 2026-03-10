"""
train_all.py — Full training pipeline: flow model → NeRF precomputation → CoordRefiner.

Usage
-----
# Quick smoke-test (tiny model, few iterations, small batch):
python train_all.py --quick-test

# Full training with defaults:
python train_all.py --h5 proteins.h5

# Custom HPC run:
python train_all.py \\
    --h5 proteins.h5 \\
    --run-name myrun \\
    --flow-iters 60000 \\
    --flow-batch 2048 \\
    --flow-lr 5e-4 \\
    --flow-hidden 512 \\
    --refiner-epochs 50 \\
    --refiner-lr 3e-4 \\
    --flow-steps 50 \\
    --device cuda

Outputs
-------
All checkpoints saved to models/<run_name>_<timestamp>/
  flow_model.pt          — trained ConditionedFlow weights
  coord_refiner.pt       — trained CoordRefiner weights
  training_log.txt       — per-epoch / per-iteration loss log
  args.txt               — full argument dump for reproducibility
"""

import argparse
import datetime
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from preprocessing.dataset import ProteinDataset, build_flow_tensors
from pipeline.models import ConditionedFlow, CoordRefiner
from pipeline.nerf import build_backbone
from pipeline.geometry import center_coords, kabsch_align, ca_rmsd


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Train ConditionedFlow + CoordRefiner on an HDF5 protein dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument('--h5',           default='proteins.h5',
                   help='Path to HDF5 dataset (from preprocess_dataset.py)')
    p.add_argument('--test-frac',    type=float, default=0.1,
                   help='Fraction of proteins held out for evaluation')
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--min-len',      type=int,   default=5,
                   help='Skip proteins shorter than this')
    p.add_argument('--max-len',      type=int,   default=2048,
                   help='Skip proteins longer than this (pos-emb table limit)')

    # ── Flow model ────────────────────────────────────────────────────────────
    p.add_argument('--flow-iters',   type=int,   default=30_000)
    p.add_argument('--flow-batch',   type=int,   default=1024)
    p.add_argument('--flow-lr',      type=float, default=1e-3)
    p.add_argument('--flow-hidden',  type=int,   default=512,
                   help='Hidden width of ConditionedFlow MLP')
    p.add_argument('--flow-steps',   type=int,   default=20,
                   help='ODE integration steps for NeRF precomputation (20=fast, 50=accurate)')
    p.add_argument('--flow-ckpt',    default=None,
                   help='Load pretrained flow weights and skip flow training')

    # ── CoordRefiner ──────────────────────────────────────────────────────────
    p.add_argument('--refiner-epochs',  type=int,   default=30)
    p.add_argument('--refiner-lr',      type=float, default=3e-4)
    p.add_argument('--refiner-d-model', type=int,   default=256)
    p.add_argument('--refiner-layers',  type=int,   default=4)
    p.add_argument('--refiner-heads',   type=int,   default=8)
    p.add_argument('--refiner-ffn',     type=int,   default=1024)
    p.add_argument('--refiner-dropout', type=float, default=0.1)
    p.add_argument('--refiner-ckpt',    default=None,
                   help='Load pretrained refiner weights and skip refiner training')

    # ── Output & logging ──────────────────────────────────────────────────────
    p.add_argument('--run-name',     default='run',
                   help='Human-readable tag prepended to the output directory name')
    p.add_argument('--out-dir',      default='models',
                   help='Parent directory for run output folders')
    p.add_argument('--save-every',   type=int,   default=10,
                   help='Save a refiner checkpoint every N epochs')
    p.add_argument('--print-every',  type=int,   default=5000,
                   help='Flow: log loss + run NeRF RMSD probe every N iterations. '
                        'Refiner: RMSD is measured every epoch regardless.')
    p.add_argument('--rmsd-probes',  type=int,   default=8,
                   help='Number of proteins sampled for the in-training RMSD sanity check')
    p.add_argument('--device',       default='cpu',
                   help="'cpu', 'cuda', or 'cuda:N'")

    # ── Quick test ────────────────────────────────────────────────────────────
    p.add_argument('--quick-test',   action='store_true',
                   help='Override hyperparams for a fast smoke-test (no real training)')

    return p.parse_args()


# ── Logging ────────────────────────────────────────────────────────────────────

class Logger:
    """Writes to stdout and a log file simultaneously."""

    def __init__(self, path: str):
        self._file = open(path, 'w', buffering=1)

    def log(self, msg: str):
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        line = f'[{ts}] {msg}'
        print(line)
        self._file.write(line + '\n')

    def close(self):
        self._file.close()


# ── RMSD probe helpers ─────────────────────────────────────────────────────────

def _build_probe_data(pids, dataset, n):
    """Load (sequence, true_coords_centred, esm2) for up to *n* random proteins."""
    sample = random.sample(pids, min(n, len(pids)))
    probes = []
    for pid in sample:
        sequence = dataset.get_sequence(pid)
        true_raw = dataset.get_true_coords(pid).astype(np.float64)
        esm2_np  = dataset.get_esm2(pid)
        L        = len(sequence)
        if len(true_raw) != L or len(esm2_np) != L:
            continue
        true_c, _ = center_coords(true_raw)
        probes.append((sequence, true_c, esm2_np))
    return probes


@torch.no_grad()
def _flow_rmsd_probe(flow, probes, args):
    """Run flow ODE + NeRF on probe proteins; return mean Cα RMSD vs true coords."""
    flow.eval()
    rmsds = []
    for sequence, true_c, esm2_np in probes:
        L     = len(sequence)
        emb_t = torch.tensor(esm2_np, dtype=torch.float32, device=args.device)
        x     = torch.randn(L, 2, device=args.device)
        ts    = torch.linspace(0., 1., args.flow_steps + 1, device=args.device)
        for i in range(args.flow_steps):
            t_s = ts[i].view(1, 1).expand(L, 1)
            dt  = (ts[i + 1] - ts[i]).float()
            v_s = flow(t_s, x, emb_t)
            v_m = flow(t_s + dt / 2, x + v_s * dt / 2, emb_t)
            x   = x + dt * v_m
        phi_psi   = x.cpu().numpy() * 180.0
        noisy_raw = build_backbone(sequence, phi_psi)
        noisy_c, _= center_coords(noisy_raw)
        noisy_aln = kabsch_align(noisy_c, true_c)
        rmsds.append(ca_rmsd(noisy_aln, true_c))
    flow.train()
    return float(np.mean(rmsds)) if rmsds else float('nan')


@torch.no_grad()
def _refiner_rmsd_probe(refiner, triples, n, device):
    """Sample *n* triples; return (mean RMSD before, mean RMSD after) refinement."""
    refiner.eval()
    sample = random.sample(triples, min(n, len(triples)))
    before, after = [], []
    for noisy_coords, true_coords, esm2_emb in sample:
        nc = noisy_coords.to(device)
        tc = true_coords.to(device)
        em = esm2_emb.to(device)
        refined = refiner(nc, em)
        L = nc.shape[0] // 12 if nc.ndim == 1 else nc.shape[0]
        before.append(ca_rmsd(
            nc.cpu().numpy().reshape(L, 4, 3),
            tc.cpu().numpy().reshape(L, 4, 3),
        ))
        after.append(ca_rmsd(
            refined.cpu().numpy().reshape(L, 4, 3),
            tc.cpu().numpy().reshape(L, 4, 3),
        ))
    refiner.train()
    return float(np.mean(before)), float(np.mean(after))


# ── Flow model training ────────────────────────────────────────────────────────

def train_flow(flow, angles_tensor, esm2_tensor, args, logger, run_dir, probes):
    logger.log(
        f'=== Flow model training ({args.flow_iters:,} iterations, '
        f'RMSD probe every {args.print_every} iters on {len(probes)} proteins) ==='
    )

    n         = angles_tensor.shape[0]
    optimizer = torch.optim.Adam(flow.parameters(), lr=args.flow_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.flow_iters
    )
    loss_fn   = nn.MSELoss()
    losses    = []
    t0        = time.time()

    flow.train()
    pbar = tqdm(range(args.flow_iters), desc='Flow', unit='it', dynamic_ncols=True)
    for i in pbar:
        idx  = torch.randint(0, n, (args.flow_batch,))
        x_1  = angles_tensor[idx].to(args.device)
        emb  = esm2_tensor[idx].to(args.device)
        x_0  = torch.randn_like(x_1)
        t    = torch.rand(args.flow_batch, 1, device=args.device)
        x_t  = (1 - t) * x_0 + t * x_1

        optimizer.zero_grad()
        loss = loss_fn(flow(t, x_t, emb), x_1 - x_0)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if (i + 1) % args.print_every == 0:
            window   = losses[-args.print_every:]
            avg_loss = float(np.mean(window))
            rmsd     = _flow_rmsd_probe(flow, probes, args)
            elapsed  = time.time() - t0
            logger.log(
                f'  Flow iter {i+1:6d}/{args.flow_iters} | '
                f'loss {avg_loss:.4f} | NeRF Cα RMSD {rmsd:.2f} Å | '
                f'lr {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s'
            )
            pbar.set_postfix(loss=f'{avg_loss:.4f}', rmsd=f'{rmsd:.2f}Å')

    flow.eval()
    ckpt = os.path.join(run_dir, 'flow_model.pt')
    torch.save(flow.state_dict(), ckpt)
    logger.log(f'Flow training complete. Saved → {ckpt}')
    return losses


# ── NeRF precomputation ────────────────────────────────────────────────────────

@torch.no_grad()
def precompute_nerf_triples(flow, train_pids, dataset, args, logger):
    """Run flow ODE on every training protein → NeRF coords → aligned triples."""
    logger.log(f'=== Precomputing NeRF training triples ({len(train_pids)} proteins) ===')

    flow.eval()
    triples = []
    skipped = 0
    t0 = time.time()

    for pid in tqdm(train_pids, desc='Flow→NeRF', dynamic_ncols=True):
        sequence = dataset.get_sequence(pid)
        true_raw = dataset.get_true_coords(pid).astype(np.float64)
        esm2_np  = dataset.get_esm2(pid)
        L        = len(sequence)

        if len(true_raw) != L or len(esm2_np) != L:
            skipped += 1
            continue

        emb_t = torch.tensor(esm2_np, dtype=torch.float32, device=args.device)
        x     = torch.randn(L, 2, device=args.device)
        ts    = torch.linspace(0., 1., args.flow_steps + 1, device=args.device)
        for i in range(args.flow_steps):
            t_s = ts[i].view(1, 1).expand(L, 1)
            dt  = (ts[i + 1] - ts[i]).float()
            v_s = flow(t_s, x, emb_t)
            v_m = flow(t_s + dt / 2, x + v_s * dt / 2, emb_t)
            x   = x + dt * v_m
        phi_psi = x.cpu().numpy() * 180.0

        true_c, _  = center_coords(true_raw)
        noisy_raw  = build_backbone(sequence, phi_psi)
        noisy_c, _ = center_coords(noisy_raw)
        noisy_aln  = kabsch_align(noisy_c, true_c)

        triples.append((
            torch.tensor(noisy_aln.reshape(L, 12), dtype=torch.float32),
            torch.tensor(true_c.reshape(L, 12),    dtype=torch.float32),
            torch.tensor(esm2_np,                   dtype=torch.float32),
        ))

    elapsed = time.time() - t0
    rmsds = [
        ca_rmsd(nc.numpy().reshape(-1, 4, 3), tc.numpy().reshape(-1, 4, 3))
        for nc, tc, _ in triples
    ]
    logger.log(
        f'Precomputed {len(triples)} triples ({skipped} skipped) in {elapsed:.0f}s | '
        f'Cα RMSD mean={np.mean(rmsds):.2f} Å  median={np.median(rmsds):.2f} Å'
    )
    return triples


# ── Refiner training ───────────────────────────────────────────────────────────

def train_refiner(refiner, triples, args, logger, run_dir):
    logger.log(
        f'=== CoordRefiner training ({args.refiner_epochs} epochs, '
        f'{len(triples)} proteins/epoch) ==='
    )

    optimizer = torch.optim.Adam(refiner.parameters(), lr=args.refiner_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.refiner_epochs * len(triples)
    )
    losses = []
    t0     = time.time()

    for epoch in range(args.refiner_epochs):
        refiner.train()
        epoch_losses = []
        indices = list(range(len(triples)))
        random.shuffle(indices)

        pbar = tqdm(indices, desc=f'Epoch {epoch+1:3d}/{args.refiner_epochs}',
                    dynamic_ncols=True, leave=False)
        for idx in pbar:
            noisy_coords, true_coords, esm2_emb = triples[idx]
            noisy_coords = noisy_coords.to(args.device)
            true_coords  = true_coords.to(args.device)
            esm2_emb     = esm2_emb.to(args.device)

            optimizer.zero_grad()
            refined = refiner(noisy_coords, esm2_emb)
            loss    = F.mse_loss(refined, true_coords)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        mean_loss = float(np.mean(epoch_losses))
        losses.extend(epoch_losses)
        elapsed = time.time() - t0

        # RMSD sanity check on a random sample of training triples
        rmsd_before, rmsd_after = _refiner_rmsd_probe(
            refiner, triples, args.rmsd_probes, args.device
        )
        logger.log(
            f'  Epoch {epoch+1:3d}/{args.refiner_epochs} | '
            f'loss {mean_loss:.5f} | '
            f'Cα RMSD {rmsd_before:.2f} → {rmsd_after:.2f} Å '
            f'(Δ {rmsd_after - rmsd_before:+.2f}) | '
            f'lr {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s'
        )

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt = os.path.join(run_dir, f'coord_refiner_ep{epoch+1:04d}.pt')
            torch.save(refiner.state_dict(), ckpt)
            logger.log(f'  Checkpoint → {ckpt}')

    refiner.eval()
    ckpt = os.path.join(run_dir, 'coord_refiner.pt')
    torch.save(refiner.state_dict(), ckpt)
    logger.log(f'Refiner training complete. Saved → {ckpt}')
    return losses


# ── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(flow, refiner, test_pids, dataset, args, logger):
    logger.log(f'=== Evaluation on {len(test_pids)} test proteins ===')

    flow.eval()
    refiner.eval()
    before_rmsds, after_rmsds = [], []

    for pid in tqdm(sorted(test_pids), desc='Eval', dynamic_ncols=True):
        sequence = dataset.get_sequence(pid)
        true_raw = dataset.get_true_coords(pid).astype(np.float64)
        esm2_np  = dataset.get_esm2(pid)
        L        = len(sequence)

        if len(true_raw) != L or len(esm2_np) != L:
            continue

        emb_t = torch.tensor(esm2_np, dtype=torch.float32, device=args.device)
        x     = torch.randn(L, 2, device=args.device)
        ts    = torch.linspace(0., 1., args.flow_steps + 1, device=args.device)
        for i in range(args.flow_steps):
            t_s = ts[i].view(1, 1).expand(L, 1)
            dt  = (ts[i + 1] - ts[i]).float()
            v_s = flow(t_s, x, emb_t)
            v_m = flow(t_s + dt / 2, x + v_s * dt / 2, emb_t)
            x   = x + dt * v_m
        phi_psi = x.cpu().numpy() * 180.0

        true_c, _  = center_coords(true_raw)
        noisy_raw  = build_backbone(sequence, phi_psi)
        noisy_c, _ = center_coords(noisy_raw)
        noisy_aln  = kabsch_align(noisy_c, true_c)

        noisy_t   = torch.tensor(noisy_aln.reshape(L, 12), dtype=torch.float32, device=args.device)
        refined_t = refiner(noisy_t, emb_t)

        before_rmsds.append(ca_rmsd(noisy_aln, true_c))
        after_rmsds.append(ca_rmsd(refined_t.cpu().numpy().reshape(L, 4, 3), true_c))

    b = np.array(before_rmsds)
    a = np.array(after_rmsds)
    logger.log(f'  Cα RMSD before : {b.mean():.3f} ± {b.std():.3f} Å')
    logger.log(f'  Cα RMSD after  : {a.mean():.3f} ± {a.std():.3f} Å')
    logger.log(
        f'  Improvement    : {(b - a).mean():+.3f} Å  '
        f'({(b > a).mean()*100:.0f}% of proteins improved)'
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Apply quick-test overrides
    if args.quick_test:
        args.flow_iters      = 200
        args.flow_batch      = 64
        args.flow_hidden     = 64
        args.flow_steps      = 5
        args.print_every     = 100
        args.rmsd_probes     = 4
        args.refiner_epochs  = 2
        args.refiner_d_model = 64
        args.refiner_layers  = 2
        args.refiner_heads   = 4
        args.refiner_ffn     = 128
        args.save_every      = 1
        print('[ quick-test mode: hyperparams overridden for fast smoke-test ]')

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Output directory — unique per run
    ts      = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.out_dir, f'{args.run_name}_{ts}')
    os.makedirs(run_dir, exist_ok=True)

    logger = Logger(os.path.join(run_dir, 'training_log.txt'))
    logger.log(f'Run directory : {run_dir}')
    logger.log(f'Device        : {args.device}')

    # Save args for reproducibility
    with open(os.path.join(run_dir, 'args.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k}: {v}\n')

    # ── Dataset ───────────────────────────────────────────────────────────────
    logger.log(f'Loading dataset from {args.h5} ...')
    dataset = ProteinDataset(args.h5, min_len=args.min_len, max_len=args.max_len)
    logger.log(f'Dataset: {len(dataset)} proteins (min_len={args.min_len}, max_len={args.max_len})')

    if args.quick_test:
        # Use at most 20 proteins for a quick test
        all_pids = dataset.pids[:20]
    else:
        all_pids = dataset.pids

    n_test     = max(1, int(len(all_pids) * args.test_frac))
    test_pids  = set(random.sample(all_pids, n_test))
    train_pids = [p for p in all_pids if p not in test_pids]
    logger.log(f'Split: {len(train_pids)} train / {len(test_pids)} test')

    # ── Flow model ────────────────────────────────────────────────────────────
    flow = ConditionedFlow(dim=2, embed_dim=320, h=args.flow_hidden).to(args.device)
    n_flow_params = sum(p.numel() for p in flow.parameters())
    logger.log(f'ConditionedFlow: {n_flow_params:,} parameters  (hidden={args.flow_hidden})')

    # Build probe proteins once — used for RMSD checks during both training stages
    probe_data = _build_probe_data(train_pids, dataset, args.rmsd_probes)
    logger.log(f'RMSD probe set: {len(probe_data)} proteins')

    if args.flow_ckpt:
        flow.load_state_dict(torch.load(args.flow_ckpt, map_location=args.device))
        flow.eval()
        logger.log(f'Loaded flow weights from {args.flow_ckpt}  (skipping training)')
    else:
        logger.log('Building flow training tensors ...')
        angles_tensor, esm2_tensor, pdb_ids_list = build_flow_tensors(dataset)
        train_set  = set(train_pids)
        train_mask = torch.tensor([pid in train_set for pid in pdb_ids_list])
        train_flow(
            flow,
            angles_tensor[train_mask],
            esm2_tensor[train_mask],
            args, logger, run_dir,
            probes=probe_data,
        )

    # ── NeRF precomputation ───────────────────────────────────────────────────
    triples = precompute_nerf_triples(flow, train_pids, dataset, args, logger)

    if len(triples) == 0:
        logger.log('ERROR: no training triples generated — check your HDF5 dataset.')
        sys.exit(1)

    # ── CoordRefiner ──────────────────────────────────────────────────────────
    refiner = CoordRefiner(
        coord_dim=12,
        esm2_dim=320,
        d_model=args.refiner_d_model,
        nhead=args.refiner_heads,
        num_layers=args.refiner_layers,
        ffn_dim=args.refiner_ffn,
        dropout=args.refiner_dropout,
        max_len=args.max_len,
    ).to(args.device)
    n_ref_params = sum(p.numel() for p in refiner.parameters())
    logger.log(
        f'CoordRefiner: {n_ref_params:,} parameters  '
        f'(d_model={args.refiner_d_model}, layers={args.refiner_layers})'
    )

    if args.refiner_ckpt:
        refiner.load_state_dict(torch.load(args.refiner_ckpt, map_location=args.device))
        refiner.eval()
        logger.log(f'Loaded refiner weights from {args.refiner_ckpt}  (skipping training)')
    else:
        train_refiner(refiner, triples, args, logger, run_dir)

    # ── Evaluation ────────────────────────────────────────────────────────────
    evaluate(flow, refiner, test_pids, dataset, args, logger)

    logger.log(f'All done. Outputs in {run_dir}')
    logger.close()


if __name__ == '__main__':
    main()
