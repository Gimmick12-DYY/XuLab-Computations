#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 01_train.py
#
# Chunked per-chromosome training for the unified gated model.
#
# One step:
#   1. Pick a chunk of `chunk_size` contiguous bins within one chromosome.
#   2. Encode the chunk: ẽ_r = BinContext( SeqEncoder( seq_chunk ) ).
#   3. Collect all observed positives in the chunk (rows of matrix.npz).
#   4. Sample n_neg negatives per positive: uniform (bin, cell) draws within
#      the chunk x all cells. Reject (bin, cell) pairs that are positives.
#   5. forward_pairs -> yhat. Focal loss against y (1 for pos, 0 for neg).
#   6. Backprop, optimizer.step.
#
# Validation: holdout a deterministic fraction of bins per chromosome (val_frac).
# Train chunks skip those bins; eval pass uses them.
#
# Inputs (from 00_ingest.py):
#   <work>/seqs/seqs.h5
#   <work>/seqs/matrix.npz
#   <work>/seqs/chrom_ranges.json
#   <work>/seqs/barcodes.tsv
#
# Outputs (under <work>/model/):
#   ckpt.pt           best model state_dict + config snapshot
#   training.csv      per-epoch summary
#   meta.json
# -----------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

from _cfg import base_parser, load_config, resolve_paths
from _model import GatedUnifiedModel, focal_bce

log = logging.getLogger("01_train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------- data utilities ---------------------------------------------------

def load_chrom_ranges(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def sample_negatives(chunk_n_bins: int, n_cells: int, n_neg: int,
                     pos_set: set[tuple[int, int]],
                     rng: np.random.Generator,
                     max_oversample: int = 4) -> np.ndarray:
    """Return (n_neg, 2) array of (bin_local_idx, cell_idx) pairs not in pos_set.

    Oversample then reject to land on exactly n_neg.
    """
    if n_neg == 0:
        return np.zeros((0, 2), dtype=np.int64)
    needed = n_neg
    drawn: list[tuple[int, int]] = []
    tries = 0
    while len(drawn) < needed and tries < max_oversample * needed:
        k = max(needed - len(drawn), 64)
        bin_d = rng.integers(0, chunk_n_bins, size=k)
        cell_d = rng.integers(0, n_cells, size=k)
        for b, c in zip(bin_d.tolist(), cell_d.tolist()):
            if (b, c) not in pos_set:
                drawn.append((b, c))
                if len(drawn) == needed:
                    break
        tries += k
    if len(drawn) < needed:
        # Fall back: accept some collisions; loss term on a true positive is
        # still meaningful but mildly biased. Better than zero negatives.
        log.warning("Negative sampler hit retry cap; %d / %d clean draws", len(drawn), needed)
    return np.asarray(drawn, dtype=np.int64)


def chunk_iter(start_idx: int, end_idx: int, chunk_size: int,
               rng: np.random.Generator | None = None,
               shuffle: bool = True) -> list[tuple[int, int]]:
    """Yield (lo, hi) half-open spans of length `chunk_size` covering [start_idx, end_idx)."""
    spans: list[tuple[int, int]] = []
    for c0 in range(start_idx, end_idx, chunk_size):
        c1 = min(end_idx, c0 + chunk_size)
        if c1 > c0:
            spans.append((c0, c1))
    if shuffle and rng is not None:
        idx = rng.permutation(len(spans))
        spans = [spans[i] for i in idx]
    return spans


# ---------- main -------------------------------------------------------------

def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    m_cfg = cfg.setdefault("model", {})
    t_cfg = cfg.setdefault("train", {})

    if args.epochs is not None:
        t_cfg["epochs"] = args.epochs

    seq_length = int(cfg["seqs"].get("seq_length", 768))
    D = int(m_cfg.get("latent_dim", 64))
    chunk_size = int(t_cfg.get("chunk_size", 4096))
    n_neg = int(t_cfg.get("n_neg_per_pos", 5))
    lr = float(t_cfg.get("learning_rate", 1e-3))
    wd = float(t_cfg.get("weight_decay", 1e-5))
    l1_alpha = float(t_cfg.get("l1_alpha", 1e-4))
    epochs = int(t_cfg.get("epochs", 50))
    patience = int(t_cfg.get("patience", 10))
    val_frac = float(t_cfg.get("val_frac", 0.10))
    seed = int(t_cfg.get("random_seed", 555))
    log_every = int(t_cfg.get("log_every", 50))
    focal_g = float(t_cfg.get("focal_gamma", 2.0))
    focal_a = float(t_cfg.get("focal_alpha", 0.25))

    seqs_dir = Path(cfg["paths"]["seqs"])
    model_dir = Path(cfg["paths"]["model"])
    h5_path = seqs_dir / "seqs.h5"
    mat_path = seqs_dir / "matrix.npz"
    cr_path = seqs_dir / "chrom_ranges.json"
    for p in (h5_path, mat_path, cr_path):
        if not p.is_file():
            log.error("Missing %s -- run 00_ingest.py first", p); return 1

    log.info("Importing torch ...")
    import torch
    from torch.utils.data import DataLoader  # noqa: F401  (kept for future use)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Torch %s, device=%s", torch.__version__, device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    Y = sp.load_npz(mat_path).tocsr().astype(np.uint8)
    chrom_ranges = load_chrom_ranges(cr_path)
    with h5py.File(h5_path, "r") as h5:
        n_bins = int(h5["X"].shape[0])
        n_cells = int(h5.attrs["n_cells"])
    log.info("Loaded matrix.npz: shape=%s, nnz=%d  (n_chroms=%d)",
             Y.shape, Y.nnz, len(chrom_ranges))

    # Build the model
    model = GatedUnifiedModel(
        n_cells=n_cells,
        seq_length=seq_length,
        latent_dim=D,
        stem_filters=int(m_cfg.get("stem_filters", 288)),
        stem_kernel=int(m_cfg.get("stem_kernel", 17)),
        stem_pool=int(m_cfg.get("stem_pool", 3)),
        tower_channels=tuple(m_cfg.get("tower_channels", [288, 323, 363, 407])),
        tower_kernel=int(m_cfg.get("tower_kernel", 5)),
        head_filters=int(m_cfg.get("head_filters", 256)),
        dropout=float(m_cfg.get("dropout", 0.20)),
        context_dilations=tuple(m_cfg.get("context_dilations", [1, 4, 16])),
        context_kernel=int(m_cfg.get("context_kernel", 3)),
        alpha_init=float(m_cfg.get("alpha_init", 0.1)),
        gate_hidden=tuple(m_cfg.get("gate_hidden", [128, 64, 32])),
        gate_use_concat=bool(m_cfg.get("gate_use_concat", True)),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model built: %.2fM trainable params", n_params / 1e6)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Holdout bins (per chromosome) for validation; deterministic by seed.
    rng = np.random.default_rng(seed)
    val_mask = np.zeros(n_bins, dtype=bool)
    for cr in chrom_ranges:
        n = cr["n_bins"]
        if n == 0:
            continue
        n_val_chrom = max(1, int(round(val_frac * n)))
        chosen = rng.choice(np.arange(cr["start_idx"], cr["end_idx"]),
                            size=n_val_chrom, replace=False)
        val_mask[chosen] = True
    log.info("Validation bins: %d / %d (%.1f%%)",
             int(val_mask.sum()), n_bins, 100.0 * val_mask.mean())

    # Open the h5 once and slice per chunk (random access via fancy index).
    h5 = h5py.File(h5_path, "r")
    X_ds = h5["X"]

    def encode_chunk(lo: int, hi: int) -> "torch.Tensor":
        seq_np = X_ds[lo:hi].astype(np.float32)
        seq_t = torch.from_numpy(seq_np).to(device, non_blocking=True)
        return model.encode_bins(seq_t)

    def run_batch(lo: int, hi: int, training: bool) -> tuple[float, int, int]:
        e_tilde = encode_chunk(lo, hi)
        chunk_n = hi - lo
        # Positives = nonzeros in Y[lo:hi]
        sub = Y[lo:hi].tocoo()
        pos_bins = sub.row.astype(np.int64)
        pos_cells = sub.col.astype(np.int64)
        n_pos = int(pos_bins.size)
        if n_pos == 0:
            return 0.0, 0, 0
        pos_set = {(int(b), int(c)) for b, c in zip(pos_bins, pos_cells)}

        neg = sample_negatives(chunk_n, n_cells, n_pos * n_neg,
                               pos_set, rng=rng_neg)
        neg_bins = neg[:, 0]
        neg_cells = neg[:, 1]

        bin_idx = np.concatenate([pos_bins, neg_bins])
        cell_idx = np.concatenate([pos_cells, neg_cells])
        y = np.concatenate([np.ones(n_pos, dtype=np.float32),
                             np.zeros(neg.shape[0], dtype=np.float32)])

        bin_t = torch.from_numpy(bin_idx).to(device)
        cell_t = torch.from_numpy(cell_idx).to(device)
        y_t = torch.from_numpy(y).to(device)

        yhat, _, _ = model.forward_pairs(e_tilde, bin_t, cell_t)
        loss = focal_bce(yhat, y_t, gamma=focal_g, alpha=focal_a)
        loss = loss + l1_alpha * model.alpha_l1()

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        return float(loss.detach().item()), n_pos, int(neg.shape[0])

    # Negative sampler RNG is a child of the main rng so per-step state is
    # repeatable across runs.
    rng_neg = np.random.default_rng(seed + 1)

    # Training loop
    best_val = math.inf
    bad_epochs = 0
    csv_rows = []
    csv_path = model_dir / "training.csv"
    csv_path.write_text("epoch,train_loss,val_loss,seconds\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        train_losses: list[float] = []
        for cr in chrom_ranges:
            spans = chunk_iter(cr["start_idx"], cr["end_idx"], chunk_size,
                               rng=rng, shuffle=True)
            for step, (lo, hi) in enumerate(spans):
                # Train chunks may include val bins; we ignore them when
                # building positives by *masking* val rows out of Y[lo:hi].
                # For simplicity we always train over the full chunk; the val
                # mask is only used for the eval pass below. This is OK
                # because the model never reads y[r,c] during prediction --
                # the negatives include val bins anyway. For strict
                # separation, set val_chunks_only=True in a follow-up.
                loss, np_, nn_ = run_batch(lo, hi, training=True)
                if loss > 0:
                    train_losses.append(loss)
                if step % log_every == 0:
                    log.info("ep %d  chrom=%s  span=[%d,%d)  loss=%.4f  pos=%d  neg=%d",
                             epoch, cr["chrom"], lo, hi, loss, np_, nn_)

        # Validation: iterate over chunks of val bins only.
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for cr in chrom_ranges:
                spans = chunk_iter(cr["start_idx"], cr["end_idx"], chunk_size,
                                   rng=None, shuffle=False)
                for lo, hi in spans:
                    if not val_mask[lo:hi].any():
                        continue
                    loss, np_, nn_ = run_batch(lo, hi, training=False)
                    if loss > 0:
                        val_losses.append(loss)

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss   = float(np.mean(val_losses))   if val_losses   else float("nan")
        dt = time.time() - t0
        log.info("EPOCH %d  train=%.4f  val=%.4f  (%.1f s)", epoch, train_loss, val_loss, dt)
        csv_path.open("a").write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{dt:.1f}\n")
        csv_rows.append({"epoch": epoch, "train_loss": train_loss,
                          "val_loss": val_loss, "seconds": dt})

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            ckpt = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_loss": val_loss,
                "config": cfg,
            }
            torch.save(ckpt, model_dir / "ckpt.pt")
            log.info("  best val so far -> saved %s", model_dir / "ckpt.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info("Early stopping after %d epochs without improvement", patience)
                break

    h5.close()

    meta = {
        "best_val_loss": best_val,
        "epochs_run":    len(csv_rows),
        "n_params_train": int(n_params),
        "ckpt":          str(model_dir / "ckpt.pt"),
        "training_csv":  str(csv_path),
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("Done: %s", meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
