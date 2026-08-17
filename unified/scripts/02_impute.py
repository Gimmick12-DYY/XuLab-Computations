#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 02_impute.py
#
# Inference: forward the trained model over every kept bin, compute g*p per
# (bin, cell), threshold by sparsity_match (same vocab as scBasset / cisTopic
# / PUscOpen), and write the standard cross-pipeline schema.
#
# Strategy:
#   pass 1: sample a subset of (bin, cell) sigmoid values to estimate the
#           threshold cutoff.
#   pass 2: for each chromosome, encode all bins in chunks, compute the dense
#           (chunk_n_bins, n_cells) yhat tensor via predict_chunk, threshold,
#           accumulate (row, col, val) triples; combine into CSR.
#
# Inputs:
#   <work>/seqs/seqs.h5              one-hot bin tensor
#   <work>/seqs/chrom_ranges.json
#   <work>/seqs/barcodes.tsv
#   <work>/mm/matrix.mtx.gz          raw nnz (for sparsity_match)
#   <work>/model/ckpt.pt             trained model
#
# Outputs (under <work>/impute/):
#   matrix_csr.npz   sparse CSR (n_bins_full, n_cells)
#   regions.tsv      one per row, "chr:start-end" -- mm row order
#   barcodes.tsv     cell barcodes
#   meta.json        threshold, mode, paths
# -----------------------------------------------------------------------------
from __future__ import annotations

import gzip
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from _cfg import base_parser, load_config, resolve_paths
from _model import GatedUnifiedModel
from _regions import open_chromatin_bin_mask

log = logging.getLogger("02_impute")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def apply_open_chromatin_mask(
    pred: sp.spmatrix,
    open_mask: np.ndarray,
) -> sp.csr_matrix:
    """Zero all predicted entries on bins outside open chromatin."""
    pred = pred.tocsr()
    if open_mask.shape[0] != pred.shape[0]:
        raise SystemExit(
            f"open_mask length {open_mask.shape[0]} != n_bins {pred.shape[0]}"
        )
    closed = np.where(~open_mask)[0]
    if closed.size == 0:
        return pred
    n_before = int(pred.nnz)
    pred_l = pred.tolil(copy=True)
    pred_l[closed] = sp.lil_matrix((closed.size, pred.shape[1]), dtype=pred.dtype)
    out = pred_l.tocsr()
    out.eliminate_zeros()
    log.info(
        "Open-chromatin mask: zeroed pred on %d closed bins; nnz %d -> %d",
        int(closed.size), n_before, int(out.nnz),
    )
    return out


def read_lines_gz(p: Path) -> list[str]:
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rt") as fh:
        return [ln.rstrip("\n") for ln in fh]


def parse_threshold_mode(spec: str) -> tuple[str, float]:
    if ":" in spec:
        mode, val = spec.split(":", 1)
        return mode.strip(), float(val)
    return spec.strip(), 0.0


def pick_threshold(values: np.ndarray, mode: str, param: float,
                   raw_nnz: int, n_bins_full: int, n_cells: int) -> float:
    if mode == "fixed":
        return float(param)
    if mode == "quantile":
        q = float(np.clip(param, 0.0, 1.0))
        return float(np.quantile(values, q))
    if mode == "sparsity_match":
        K = float(param)
        target = K * raw_nnz
        denom = float(n_bins_full) * float(n_cells)
        keep_frac = min(max(target / denom, 1e-9), 1.0)
        return float(np.quantile(values, 1.0 - keep_frac))
    raise SystemExit(f"Unknown threshold_mode: {mode}")


def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Override path to model ckpt (.pt)")
    ap.add_argument("--chunk-bins", type=int, default=None,
                    help="Bins per inference chunk (default 4096)")
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    i_cfg = cfg.setdefault("impute", {})
    m_cfg = cfg.setdefault("model", {})
    t_cfg = cfg.setdefault("train", {})

    seq_length = int(cfg["seqs"].get("seq_length", 768))
    D = int(m_cfg.get("latent_dim", 64))
    chunk_size = int(args.chunk_bins if args.chunk_bins is not None
                     else t_cfg.get("chunk_size", 4096))

    threshold_mode_str = str(i_cfg.get("threshold_mode", "sparsity_match:3"))
    threshold_sample_size = int(i_cfg.get("threshold_sample_size", 4_000_000))
    binarize = bool(i_cfg.get("binarize_output", True))
    keep_raw = bool(i_cfg.get("keep_raw", True))
    weight = float(i_cfg.get("weight", 1.0))
    open_bed = i_cfg.get("open_chromatin_bed", None)
    # imputed_only=True: mask pred only; raw outside ATAC kept via keep_raw.
    # False: also wipe raw outside ATAC in the final matrix (rarely wanted).
    mask_imputed_only = bool(i_cfg.get("open_chromatin_mask_imputed_only", True))

    seqs_dir = Path(cfg["paths"]["seqs"])
    mm_dir = Path(cfg["paths"]["mm"])
    model_dir = Path(cfg["paths"]["model"])
    out_dir = Path(cfg["paths"]["impute"])

    h5_path = seqs_dir / "seqs.h5"
    cr_path = seqs_dir / "chrom_ranges.json"
    ckpt_path = Path(args.ckpt) if args.ckpt else (model_dir / "ckpt.pt")
    for p in (h5_path, cr_path, ckpt_path, mm_dir / "matrix.mtx.gz"):
        if not p.is_file():
            log.error("Missing %s", p); return 1

    log.info("Importing torch ...")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Torch %s, device=%s", torch.__version__, device)

    # Load raw to size things + sparsity_match target.
    log.info("Loading raw mm %s", mm_dir)
    raw = sio.mmread(mm_dir / "matrix.mtx.gz").tocsr()
    n_bins_full, n_cells = raw.shape
    raw_nnz = int(raw.nnz)
    raw_regions  = read_lines_gz(mm_dir / "regions.tsv.gz")
    raw_barcodes = read_lines_gz(mm_dir / "barcodes.tsv.gz")
    log.info("Raw: %d x %d, nnz=%d", n_bins_full, n_cells, raw_nnz)

    # Open seqs.h5 (keep open through inference)
    h5 = h5py.File(h5_path, "r")
    X_ds = h5["X"]
    n_bins_kept = int(X_ds.shape[0])
    mm_row_idx = np.asarray(h5["mm_row_idx"])
    chrom_ranges = json.loads(cr_path.read_text())

    # Prefer sparsity flags saved in the checkpoint config (train-time truth).
    ckpt_cfg = {}
    ckpt_probe = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt_probe.get("config"), dict):
        ckpt_cfg = ckpt_probe["config"]
    del ckpt_probe
    ckpt_m = (ckpt_cfg.get("model") or {}) if ckpt_cfg else {}
    sparsity_aware = bool(ckpt_m.get("sparsity_aware", m_cfg.get("sparsity_aware", False)))
    use_cell_depth = bool(ckpt_m.get("use_cell_depth", m_cfg.get("use_cell_depth", True)))
    use_obs_count = bool(ckpt_m.get("use_obs_count", m_cfg.get("use_obs_count", True)))

    counts = None
    cell_depth_t = None
    if sparsity_aware:
        counts_path = seqs_dir / "counts.npz"
        depth_path = seqs_dir / "cell_depth.npy"
        for p in (counts_path, depth_path):
            if not p.is_file():
                log.error("sparsity_aware ckpt but missing %s -- re-run 00_ingest.py", p)
                return 1
        counts = sp.load_npz(counts_path).tocsr()
        if counts.shape != (n_bins_kept, n_cells):
            log.error("counts.npz shape %s != (%d, %d)", counts.shape, n_bins_kept, n_cells)
            return 1
        cell_depth_log = np.log1p(np.load(depth_path).astype(np.float64).ravel()).astype(np.float32)
        if cell_depth_log.size != n_cells:
            log.error("cell_depth length mismatch"); return 1
        log.info("Sparsity-aware impute: cell_depth + counts.npz")

    # Build model + load weights
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
        sparsity_aware=sparsity_aware,
        use_cell_depth=use_cell_depth,
        use_obs_count=use_obs_count,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    log.info("Loaded ckpt epoch=%s val_loss=%.4f sparsity_aware=%s",
             ckpt.get("epoch"), float(ckpt.get("val_loss", float("nan"))),
             sparsity_aware)
    if sparsity_aware and use_cell_depth:
        cell_depth_t = torch.from_numpy(cell_depth_log).to(device)

    def encode_chunk(lo: int, hi: int) -> "torch.Tensor":
        seq_np = X_ds[lo:hi].astype(np.float32)
        seq_t = torch.from_numpy(seq_np).to(device, non_blocking=True)
        return model.encode_bins(seq_t)

    def predict_lo_hi(lo: int, hi: int) -> "torch.Tensor":
        e_tilde = encode_chunk(lo, hi)
        if not sparsity_aware:
            return model.predict_chunk(e_tilde)
        obs_t = None
        if use_obs_count:
            assert counts is not None
            obs_np = np.log1p(counts[lo:hi].toarray().astype(np.float32))
            obs_t = torch.from_numpy(np.ascontiguousarray(obs_np)).to(device)
        return model.predict_chunk(
            e_tilde,
            cell_depth=cell_depth_t if use_cell_depth else None,
            obs_counts=obs_t,
        )

    mode, param = parse_threshold_mode(threshold_mode_str)

    # ---- pass 1: threshold estimation --------------------------------------
    log.info("Pass 1: threshold estimation (mode=%s)", threshold_mode_str)
    sample_target = threshold_sample_size // max(n_cells, 1)
    sample_target = max(1, sample_target)

    # Sample uniformly across chromosomes proportional to size.
    rng = np.random.default_rng(0)
    chrom_weights = np.array([cr["n_bins"] for cr in chrom_ranges], dtype=np.float64)
    chrom_weights /= chrom_weights.sum()

    sample_vals: list[np.ndarray] = []
    with torch.no_grad():
        gathered = 0
        while gathered < sample_target:
            ci = int(rng.choice(len(chrom_ranges), p=chrom_weights))
            cr = chrom_ranges[ci]
            if cr["n_bins"] == 0:
                continue
            lo = cr["start_idx"] + int(rng.integers(0, cr["n_bins"]))
            hi = min(cr["end_idx"], lo + min(chunk_size, sample_target - gathered + 16))
            yhat = predict_lo_hi(lo, hi)
            sample_vals.append(yhat.detach().cpu().numpy().ravel())
            gathered += yhat.shape[0]
    sv = np.concatenate(sample_vals)[:threshold_sample_size]
    threshold = pick_threshold(sv, mode, param,
                                raw_nnz=raw_nnz, n_bins_full=n_bins_full,
                                n_cells=n_cells)
    log.info("Threshold (%s) = %.6f  [sample mean=%.4f max=%.4f]",
             threshold_mode_str, threshold, float(sv.mean()), float(sv.max()))
    del sample_vals, sv

    # ---- pass 2: predict + threshold per chrom -----------------------------
    log.info("Pass 2: dense predict + threshold per chrom")
    rows_l: list[np.ndarray] = []
    cols_l: list[np.ndarray] = []
    vals_l: list[np.ndarray] = []
    n_kept = 0

    with torch.no_grad():
        for ci, cr in enumerate(chrom_ranges):
            chrom = cr["chrom"]
            for lo in range(cr["start_idx"], cr["end_idx"], chunk_size):
                hi = min(cr["end_idx"], lo + chunk_size)
                yhat = predict_lo_hi(lo, hi).detach().cpu().numpy()
                keep = yhat > threshold
                if not keep.any():
                    continue
                rr, cc = np.where(keep)
                # Map (local bin row in chunk) -> mm_row_idx (full mm space)
                rows_l.append(mm_row_idx[lo + rr])
                cols_l.append(cc.astype(np.int64))
                vals_l.append(yhat[rr, cc].astype(np.float32))
                n_kept += int(keep.sum())
            log.info("[%d/%d] chrom=%s done; total kept=%d",
                     ci + 1, len(chrom_ranges), chrom, n_kept)

    h5.close()

    if rows_l:
        rows = np.concatenate(rows_l)
        cols = np.concatenate(cols_l)
        vals = np.concatenate(vals_l)
    else:
        rows = np.zeros(0, dtype=np.int64); cols = rows.copy()
        vals = np.zeros(0, dtype=np.float32)

    log.info("Assembling CSR (%d x %d, nnz=%d)", n_bins_full, n_cells, len(vals))
    pred = sp.coo_matrix((vals, (rows, cols)),
                         shape=(n_bins_full, n_cells)).tocsr()
    pred.sum_duplicates()
    pred.eliminate_zeros()

    # Optional binarize / weight + open-chromatin mask + raw union
    if binarize:
        pred = (pred > 0).astype(np.float32).tocsr()
    elif weight != 1.0:
        pred = pred.copy()
        pred.data *= weight

    open_mask = open_chromatin_bin_mask(raw_regions, open_bed)
    if open_mask is not None:
        n_open = int(open_mask.sum())
        log.info(
            "Open-chromatin BED %s: %d / %d bins open (%.2f%%)",
            open_bed, n_open, n_bins_full, 100.0 * n_open / max(n_bins_full, 1),
        )
        pred_nnz_pre = int(pred.nnz)
        pred = apply_open_chromatin_mask(pred, open_mask)
        log.info("Masked imputed calls outside open chromatin: pred nnz %d -> %d",
                 pred_nnz_pre, int(pred.nnz))

    raw_bin = (raw != 0).astype(np.float32)
    if keep_raw:
        out = raw_bin.maximum(pred).tocsr()
    else:
        out = pred.tocsr()

    if open_mask is not None and not mask_imputed_only:
        # Also drop raw outside open chromatin (hard mask of final matrix).
        closed = np.where(~open_mask)[0]
        if closed.size:
            out_l = out.tolil(copy=True)
            out_l[closed] = sp.lil_matrix((closed.size, out.shape[1]), dtype=out.dtype)
            out = out_l.tocsr()
            log.info("Hard open-chromatin mask: wiped %d closed bins including raw",
                     int(closed.size))

    out.eliminate_zeros()

    out_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out_dir / "matrix_csr.npz", out)
    (out_dir / "regions.tsv").write_text("\n".join(raw_regions) + "\n")
    (out_dir / "barcodes.tsv").write_text("\n".join(raw_barcodes) + "\n")

    meta = {
        "pipeline":         "unified",
        "shape":            [int(n_bins_full), int(n_cells)],
        "raw_nnz":          raw_nnz,
        "predicted_nnz":    int(pred.nnz),
        "output_nnz":       int(out.nnz),
        "gain_vs_raw":      int(out.nnz) - int(raw_bin.nnz),
        "threshold_mode":   threshold_mode_str,
        "threshold":        float(threshold),
        "binarize_output":  binarize,
        "keep_raw":         keep_raw,
        "weight":           weight,
        "open_chromatin_bed": str(open_bed) if open_bed else None,
        "open_chromatin_n_open_bins": int(open_mask.sum()) if open_mask is not None else None,
        "open_chromatin_mask_imputed_only": mask_imputed_only,
        "ckpt":             str(ckpt_path),
        "mm_dir":           str(mm_dir),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("Wrote %s -- gain over raw: %+d entries",
             out_dir / "matrix_csr.npz", out.nnz - raw_bin.nnz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
