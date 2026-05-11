#!/usr/bin/env python
# -----------------------------------------------------------------------------
# sensitivity_specificity.py
#
# For each imputation pipeline, treat raw[r, c] > 0 as the ground-truth label
# and imputed[r, c] as the score; sweep thresholds via a streaming histogram
# pass over the modeled rows; report sensitivity, specificity, precision, F1,
# MCC at every threshold, plus AUROC / AUPR and the sparsity-matched threshold
# (where #(imputed > t) == #(raw nonzeros)).
#
# Each --input PATH is one of:
#   * directory with factors.npz + regions.tsv   (cisTopic / scOpen)
#   * directory with matrix.npy  + regions.tsv   (FITS / MAGIC)
#   * legacy HDF5 file with /Prc                 (older runs)
#
# The raw matrix is loaded from --mm-dir (must contain matrix.mtx.gz +
# regions.tsv.gz + barcodes.tsv.gz). Rows of the imputed matrix are aligned to
# raw by region name; rows present in raw but absent from the imputed input
# count as predicted-negative everywhere (so a pipeline that models a small
# region subset is penalised by FNs from unmodeled positives, as expected).
#
# Columns are aligned by barcode: if the impute directory has barcodes.tsv
# (or .gz), each imputed column maps to the matching raw column; otherwise we
# assume the same column order and width as the raw matrix.
#
# Memory: we stream the imputed matrix in row blocks (default 2000 rows) and
# never materialise the full dense product. The per-pipeline cost is one pass
# through the modeled rows + one histogram per chunk.
#
# Usage:
#   python sensitivity_specificity.py \
#     --mm-dir   /work/.../cistopic_ctcf/mm \
#     --out-dir  /work/.../downstream/sensitivity_specificity \
#     --input cisTopic=/work/.../cistopic_ctcf/impute \
#     --input FITS=/work/.../FITS/work/ctcf/impute \
#     --input scOpen=/work/.../scOpen/work/ctcf/impute \
#     --input MAGIC=/work/.../MAGIC/work/ctcf/impute
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path
from typing import Iterator

import h5py
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402
import scipy.sparse as sp  # noqa: E402

log = logging.getLogger('sens_spec')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


# -----------------------------------------------------------------------------
# Input parsing
# -----------------------------------------------------------------------------
def parse_input_arg(s: str) -> tuple[str, Path]:
    if '=' not in s:
        raise argparse.ArgumentTypeError(f"--input expects LABEL=PATH, got {s!r}")
    label, path = s.split('=', 1)
    label = label.strip()
    path = Path(path.strip())
    if not label:
        raise argparse.ArgumentTypeError(f"--input has empty LABEL: {s!r}")
    if not path.exists():
        raise argparse.ArgumentTypeError(f"--input path does not exist: {path}")
    return label, path


def detect_input_kind(path: Path) -> str:
    if path.is_file():
        if path.suffix.lower() in ('.h5', '.hdf5'):
            return 'hdf5'
        raise SystemExit(f'Unsupported input file: {path}')
    if path.is_dir():
        if (path / 'factors.npz').exists() and (path / 'regions.tsv').exists():
            return 'factored'
        if (path / 'matrix.npy').exists() and (path / 'regions.tsv').exists():
            return 'dense'
        cand = sorted(list(path.glob('*.h5')) + list(path.glob('*.hdf5')))
        if cand:
            return 'hdf5'
    raise SystemExit(f'Cannot determine input kind from {path}')


# -----------------------------------------------------------------------------
# Raw loader
# -----------------------------------------------------------------------------
def _read_lines_gz(path: Path) -> list[str]:
    with gzip.open(path, 'rt') as fh:
        return [ln.rstrip('\n') for ln in fh]


def load_raw_mm(mm_dir: Path) -> tuple[sp.csr_matrix, dict[str, int], list[str]]:
    mtx_path = mm_dir / 'matrix.mtx.gz'
    reg_path = mm_dir / 'regions.tsv.gz'
    bar_path = mm_dir / 'barcodes.tsv.gz'
    for p in (mtx_path, reg_path, bar_path):
        if not p.exists():
            raise SystemExit(f'Missing {p}')
    log.info('Reading raw matrix %s', mtx_path)
    mat = sio.mmread(str(mtx_path)).tocsr()
    mat = (mat > 0).astype(np.int8)  # binarise
    regions = _read_lines_gz(reg_path)
    barcodes = _read_lines_gz(bar_path)
    if mat.shape != (len(regions), len(barcodes)):
        raise SystemExit(
            f'Raw shape mismatch: matrix {mat.shape} vs regions/barcodes '
            f'{len(regions)}/{len(barcodes)}'
        )
    name_to_row = {r: i for i, r in enumerate(regions)}
    log.info('  raw shape=%s, nnz=%d (density=%.3g)',
             mat.shape, mat.nnz, mat.nnz / float(mat.shape[0] * mat.shape[1]))
    return mat, name_to_row, barcodes


def _read_barcodes_plain(path: Path) -> list[str]:
    return [ln.rstrip('\n') for ln in path.read_text().splitlines() if ln.strip()]


def load_impute_raw_column_indices(impute_path: Path, raw_barcodes: list[str]) -> np.ndarray:
    """Map each imputed cell column to a raw matrix column index (int64).

    If ``impute_path/barcodes.tsv`` or ``barcodes.tsv.gz`` exists, its lines
    must be a subset of ``raw_barcodes`` (same strings as in the raw MM).
    Otherwise returns ``0..C_raw-1`` (caller must ensure imputed width matches).
    """
    base = impute_path.parent if impute_path.is_file() else impute_path
    lines: list[str] | None = None
    for name in ('barcodes.tsv', 'barcodes.tsv.gz'):
        p = base / name
        if not p.exists():
            continue
        if name.endswith('.gz'):
            lines = _read_lines_gz(p)
        else:
            lines = _read_barcodes_plain(p)
        break
    n_raw = len(raw_barcodes)
    if lines is None:
        return np.arange(n_raw, dtype=np.int64)
    raw_to_col = {b: i for i, b in enumerate(raw_barcodes)}
    out = np.empty(len(lines), dtype=np.int64)
    for j, bc in enumerate(lines):
        if bc not in raw_to_col:
            raise SystemExit(
                f'Impute barcode {bc!r} (line {j + 1}) not found in raw '
                f'barcodes.tsv.gz ({base})'
            )
        out[j] = raw_to_col[bc]
    return out


def imputed_n_columns(path: Path, kind: str) -> int:
    if kind == 'factored':
        with np.load(path / 'factors.npz') as z:
            return int(z['H'].shape[1])
    if kind == 'dense':
        return int(np.load(path / 'matrix.npy', mmap_mode='r').shape[1])
    if kind == 'hdf5':
        h5_path = path if path.is_file() else (
            sorted(list(path.glob('*.h5')) + list(path.glob('*.hdf5')))[0]
        )
        with h5py.File(h5_path, 'r') as h5:
            return int(h5['Prc'].shape[1])
    raise SystemExit(f'Unhandled input kind {kind}')


# -----------------------------------------------------------------------------
# Imputed chunk iterators
#
# Each yields (raw_row_idx_block, imputed_dense_block_float32) where
# raw_row_idx_block[i] is the raw matrix row index for imputed row i (or -1
# if the region name is missing from raw, which shouldn't happen in practice).
# -----------------------------------------------------------------------------
def _name_to_raw_idx(regions_path: Path, raw_name_to_row: dict[str, int]) -> np.ndarray:
    names = regions_path.read_text().splitlines()
    return np.array([raw_name_to_row.get(n, -1) for n in names], dtype=np.int64)


def iter_chunks_factored(impute_dir: Path, raw_name_to_row: dict[str, int],
                          chunk_rows: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    raw_idx = _name_to_raw_idx(impute_dir / 'regions.tsv', raw_name_to_row)
    with np.load(impute_dir / 'factors.npz') as z:
        W = z['W']
        H = z['H']
        per_cell_factor = float(z['per_cell_factor']) if 'per_cell_factor' in z.files else 1.0
        # Avoid keeping a reference to the closed npz: copy into RAM. W is small
        # (n_regions x k), H is small (k x n_cells); both float32.
        # np.asarray(..., copy=) needs NumPy >= 2; np.array(..., copy=) works on 1.x.
        W = np.array(W, dtype=np.float32, copy=True)
        H = np.array(H, dtype=np.float32, copy=True)
    Rm = W.shape[0]
    for s in range(0, Rm, chunk_rows):
        e = min(s + chunk_rows, Rm)
        block = (W[s:e] @ H) * per_cell_factor
        yield raw_idx[s:e], block.astype(np.float32, copy=False)


def iter_chunks_dense(impute_dir: Path, raw_name_to_row: dict[str, int],
                       chunk_rows: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    raw_idx = _name_to_raw_idx(impute_dir / 'regions.tsv', raw_name_to_row)
    arr = np.load(impute_dir / 'matrix.npy', mmap_mode='r')
    Rm = arr.shape[0]
    for s in range(0, Rm, chunk_rows):
        e = min(s + chunk_rows, Rm)
        block = np.asarray(arr[s:e], dtype=np.float32)
        yield raw_idx[s:e], block


def iter_chunks_hdf5(h5_path: Path, raw_name_to_row: dict[str, int],
                      chunk_rows: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    with h5py.File(h5_path, 'r') as h5:
        ds = h5['Prc']
        if 'regions' in h5:
            regs = [s.decode() if isinstance(s, bytes) else str(s)
                    for s in h5['regions'][:]]
            raw_idx = np.array([raw_name_to_row.get(r, -1) for r in regs],
                               dtype=np.int64)
        else:
            # Legacy: HDF5 row index == raw row index.
            raw_idx = np.arange(ds.shape[0], dtype=np.int64)
        Rm = ds.shape[0]
        for s in range(0, Rm, chunk_rows):
            e = min(s + chunk_rows, Rm)
            block = ds[s:e, :].astype(np.float32, copy=False)
            yield raw_idx[s:e], block


def iter_chunks(path: Path, kind: str, raw_name_to_row: dict[str, int],
                 chunk_rows: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if kind == 'factored':
        yield from iter_chunks_factored(path, raw_name_to_row, chunk_rows)
    elif kind == 'dense':
        yield from iter_chunks_dense(path, raw_name_to_row, chunk_rows)
    elif kind == 'hdf5':
        h5_path = path if path.is_file() else (
            sorted(list(path.glob('*.h5')) + list(path.glob('*.hdf5')))[0]
        )
        yield from iter_chunks_hdf5(h5_path, raw_name_to_row, chunk_rows)
    else:
        raise SystemExit(f'Unhandled input kind {kind}')


# -----------------------------------------------------------------------------
# Threshold grid
# -----------------------------------------------------------------------------
def auto_thresholds(path: Path, kind: str, raw_name_to_row: dict[str, int],
                     n_thresholds: int, sample_size: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Sample imputed values across a few row blocks, pick quantile thresholds.

    Quantiles are weighted toward the right tail (where signal lives) on a
    log-style spacing in [0.001, 0.999]. We always include 0.0 so the ROC
    curve has a (FPR=1, TPR=1) anchor.
    """
    samples: list[np.ndarray] = []
    collected = 0
    target_per_chunk = max(50_000, sample_size // 4)
    for raw_idx, block in iter_chunks(path, kind, raw_name_to_row, chunk_rows=2000):
        flat = block.ravel()
        if flat.size > target_per_chunk:
            sub = rng.choice(flat, size=target_per_chunk, replace=False)
        else:
            sub = flat
        samples.append(np.asarray(sub, dtype=np.float64))
        collected += sub.size
        if collected >= sample_size:
            break
    if not samples:
        raise SystemExit('Could not sample any imputed values; is the input empty?')
    s = np.concatenate(samples)
    if s.size > sample_size:
        s = rng.choice(s, size=sample_size, replace=False)

    qs = np.concatenate([
        np.linspace(0.001, 0.5,  n_thresholds // 4, endpoint=False),
        np.linspace(0.5,   0.9,  n_thresholds // 4, endpoint=False),
        np.linspace(0.9,   0.99, n_thresholds // 4, endpoint=False),
        np.linspace(0.99,  0.999, n_thresholds - 3 * (n_thresholds // 4), endpoint=True),
    ])
    thresholds = np.quantile(s, np.clip(qs, 0.0, 1.0))
    # Anchor at 0 so we get a top-right ROC point.
    thresholds = np.concatenate([[0.0], thresholds])
    thresholds = np.unique(thresholds)
    log.info('  auto thresholds: %d values in [%.4g, %.4g] (%d unique)',
             len(thresholds), float(thresholds.min()), float(thresholds.max()),
             len(thresholds))
    return thresholds


# -----------------------------------------------------------------------------
# Streaming confusion matrix
# -----------------------------------------------------------------------------
def stream_confusion(path: Path, kind: str, raw: sp.csr_matrix,
                      raw_name_to_row: dict[str, int],
                      thresholds: np.ndarray,
                      chunk_rows: int,
                      raw_col_indices: np.ndarray,
                      ) -> dict[str, np.ndarray | int]:
    """For each threshold t, accumulate (TP, FP) by histogramming imputed
    values stratified by raw label. Returns (TP, TN, FP, FN, pos_total, neg_total)."""
    # bin edges = [-inf, t_0, t_1, ..., t_{T-1}, +inf]
    # np.histogram bin i = count of values in [edges[i], edges[i+1])
    # Pred at threshold t_i = "imp > t_i" approximated by values in bins [i+1, T]
    # (continuous imputed values; equality is measure-zero, so this is fine.)
    T = len(thresholds)
    edges = np.concatenate([[-np.inf], np.asarray(thresholds, dtype=np.float64), [np.inf]])
    pos_hist = np.zeros(T + 1, dtype=np.int64)
    neg_hist = np.zeros(T + 1, dtype=np.int64)
    n_rows_seen = 0
    n_rows_unmatched = 0
    pos_total_unmodeled = 0  # raw nnz in rows that were NOT modeled (always FN)
    pos_extra_modeled_cols = 0  # raw positives in modeled rows, raw cols absent from impute

    # Pre-compute set of modeled raw row indices so we can credit unmodeled FNs.
    raw_total_pos = int(raw.nnz)
    raw_total_size = int(raw.shape[0]) * int(raw.shape[1])

    impute_cover = np.zeros(raw.shape[1], dtype=bool)
    impute_cover[raw_col_indices] = True

    modeled_raw_rows: set[int] = set()
    for raw_idx, block in iter_chunks(path, kind, raw_name_to_row, chunk_rows):
        ok = raw_idx >= 0
        n_rows_seen += int(ok.sum())
        n_rows_unmatched += int((~ok).sum())
        if not ok.any():
            continue
        sub = raw[raw_idx[ok]][:, raw_col_indices]
        raw_rows = (sub.toarray() > 0)  # (n_chunk, n_impute_cols) aligned to block
        imp = block[ok]
        modeled_raw_rows.update(int(i) for i in raw_idx[ok].tolist())

        pos_vals = imp[raw_rows]
        neg_vals = imp[~raw_rows]
        if pos_vals.size:
            ph, _ = np.histogram(pos_vals, bins=edges)
            pos_hist += ph.astype(np.int64)
        if neg_vals.size:
            nh, _ = np.histogram(neg_vals, bins=edges)
            neg_hist += nh.astype(np.int64)

    # Cumulative sum from the right: count of values strictly greater than t_i
    pos_cum_rev = np.cumsum(pos_hist[::-1])[::-1]
    neg_cum_rev = np.cumsum(neg_hist[::-1])[::-1]
    TP_modeled = pos_cum_rev[1:].astype(np.int64)
    FP_modeled = neg_cum_rev[1:].astype(np.int64)
    pos_total_modeled = int(pos_hist.sum())
    neg_total_modeled = int(neg_hist.sum())

    # Unmodeled FN: raw nnz in rows we never visited via the impute input.
    if len(modeled_raw_rows) < raw.shape[0]:
        all_rows = np.arange(raw.shape[0], dtype=np.int64)
        modeled_arr = np.zeros(raw.shape[0], dtype=bool)
        if modeled_raw_rows:
            modeled_arr[np.fromiter(modeled_raw_rows, dtype=np.int64)] = True
        unmodeled_rows = all_rows[~modeled_arr]
        if unmodeled_rows.size:
            sub = raw[unmodeled_rows]
            pos_total_unmodeled = int(sub.nnz)

    # Raw positives in modeled rows at raw columns with no imputed score (dropped cells).
    if modeled_raw_rows and not bool(np.all(impute_cover)):
        for r in modeled_raw_rows:
            inds = raw.getrow(int(r)).indices
            if inds.size:
                pos_extra_modeled_cols += int(np.sum(~impute_cover[inds]))

    full_pos_total = pos_total_modeled + pos_extra_modeled_cols + pos_total_unmodeled
    full_neg_total = raw_total_size - full_pos_total

    out = {
        # modeled-only view (per-pipeline universe)
        'thresholds':         np.asarray(thresholds, dtype=np.float64),
        'TP_modeled':         TP_modeled,
        'FP_modeled':         FP_modeled,
        'FN_modeled':         pos_total_modeled - TP_modeled,
        'TN_modeled':         neg_total_modeled - FP_modeled,
        'pos_total_modeled':  pos_total_modeled,
        'neg_total_modeled':  neg_total_modeled,
        # full-universe view: unmodeled rows contribute predicted-zero (TN
        # for raw zeros, FN for raw ones).
        'TP_full':            TP_modeled,
        'FP_full':            FP_modeled,
        'FN_full':            full_pos_total - TP_modeled,
        'TN_full':            full_neg_total - FP_modeled,
        'pos_total_full':     full_pos_total,
        'neg_total_full':     full_neg_total,
        'n_rows_modeled':     int(n_rows_seen),
        'n_rows_unmatched':   int(n_rows_unmatched),
        'pos_total_unmodeled': int(pos_total_unmodeled),
        'pos_extra_modeled_cols': int(pos_extra_modeled_cols),
    }
    return out


# -----------------------------------------------------------------------------
# Metric / curve helpers
# -----------------------------------------------------------------------------
def derive_metrics(TP: np.ndarray, FP: np.ndarray, FN: np.ndarray,
                    TN: np.ndarray) -> dict[str, np.ndarray]:
    TP = TP.astype(np.float64)
    FP = FP.astype(np.float64)
    FN = FN.astype(np.float64)
    TN = TN.astype(np.float64)
    eps = 1e-30
    sens = TP / np.maximum(TP + FN, eps)         # recall, TPR
    spec = TN / np.maximum(TN + FP, eps)         # TNR
    prec = np.where(TP + FP > 0, TP / np.maximum(TP + FP, eps), 0.0)
    f1   = np.where(prec + sens > 0,
                    2 * prec * sens / np.maximum(prec + sens, eps),
                    0.0)
    acc  = (TP + TN) / np.maximum(TP + TN + FP + FN, eps)
    fpr  = 1.0 - spec
    num  = TP * TN - FP * FN
    denom = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc  = np.where(denom > 0, num / np.maximum(denom, eps), 0.0)
    return {
        'sens': sens, 'spec': spec, 'prec': prec,
        'f1':   f1,   'acc':  acc,  'mcc':  mcc,
        'fpr':  fpr,  'tpr':  sens,
    }


def auroc_from_fpr_tpr(fpr: np.ndarray, tpr: np.ndarray) -> float:
    # Anchor curve at (0, 0) and (1, 1).
    f = np.concatenate([[0.0], fpr, [1.0]])
    t = np.concatenate([[0.0], tpr, [1.0]])
    order = np.argsort(f)
    return float(np.trapz(t[order], f[order]))


def aupr_from_precision_recall(precision: np.ndarray,
                                recall: np.ndarray) -> float:
    # Sort by recall ascending; treat as average precision.
    order = np.argsort(recall)
    r = recall[order]
    p = precision[order]
    r = np.concatenate([[0.0], r])
    p = np.concatenate([[float(p[0]) if p.size else 1.0], p])
    return float(np.sum(np.diff(r) * p[1:]))


def sparsity_match_idx(TP: np.ndarray, FP: np.ndarray,
                        target_positives: int) -> int:
    """Pick threshold index where TP+FP is closest to target_positives."""
    pred_pos = TP + FP
    if pred_pos.size == 0:
        return 0
    return int(np.argmin(np.abs(pred_pos.astype(np.int64) - int(target_positives))))


# -----------------------------------------------------------------------------
# Per-input pipeline
# -----------------------------------------------------------------------------
def evaluate_input(label: str, path: Path,
                    raw: sp.csr_matrix, raw_name_to_row: dict[str, int],
                    raw_barcodes: list[str],
                    thresholds: np.ndarray | None, n_thresholds: int,
                    chunk_rows: int, rng: np.random.Generator
                    ) -> dict:
    kind = detect_input_kind(path)
    log.info('Input %r (%s) -> %s', label, kind, path)

    raw_col_indices = load_impute_raw_column_indices(path, raw_barcodes)
    n_impute = imputed_n_columns(path, kind)
    if int(raw_col_indices.size) != n_impute:
        raise SystemExit(
            f'{label}: barcode map length {raw_col_indices.size} != imputed '
            f'n_columns={n_impute} ({path}). Add barcodes.tsv matching columns.'
        )
    if raw_col_indices.size != len(raw_barcodes) or not np.array_equal(
            raw_col_indices, np.arange(len(raw_barcodes), dtype=np.int64)):
        log.info('  impute barcodes: %d cells aligned to raw (%d raw barcodes)',
                 raw_col_indices.size, len(raw_barcodes))

    if thresholds is None:
        log.info('  picking auto thresholds (sample 4M values)...')
        thr = auto_thresholds(path, kind, raw_name_to_row,
                              n_thresholds=n_thresholds,
                              sample_size=4_000_000, rng=rng)
    else:
        thr = np.asarray(sorted(thresholds), dtype=np.float64)
        log.info('  using %d user-supplied thresholds in [%.4g, %.4g]',
                 thr.size, float(thr.min()), float(thr.max()))

    log.info('  streaming confusion-matrix histogram (chunk_rows=%d)', chunk_rows)
    conf = stream_confusion(path, kind, raw, raw_name_to_row, thr, chunk_rows,
                            raw_col_indices)
    log.info('  modeled rows seen=%d (unmatched=%d); modeled raw pos=%d, neg=%d; '
             'unmodeled raw pos=%d; modeled-row raw pos in dropped cols=%d',
             conf['n_rows_modeled'], conf['n_rows_unmatched'],
             conf['pos_total_modeled'], conf['neg_total_modeled'],
             conf['pos_total_unmodeled'], conf['pos_extra_modeled_cols'])

    metrics: dict[str, dict] = {}
    for view in ('modeled', 'full'):
        TP = conf[f'TP_{view}']; FP = conf[f'FP_{view}']
        FN = conf[f'FN_{view}']; TN = conf[f'TN_{view}']
        m = derive_metrics(TP, FP, FN, TN)
        m['TP'] = TP; m['FP'] = FP; m['FN'] = FN; m['TN'] = TN
        m['auroc'] = auroc_from_fpr_tpr(m['fpr'], m['tpr'])
        m['aupr']  = aupr_from_precision_recall(m['prec'], m['sens'])
        # Best F1 and sparsity-matched
        if m['f1'].size:
            bf = int(np.argmax(m['f1']))
            m['best_f1_idx']       = bf
            m['best_f1_threshold'] = float(thr[bf])
            m['best_f1']           = float(m['f1'][bf])
            m['best_f1_sens']      = float(m['sens'][bf])
            m['best_f1_spec']      = float(m['spec'][bf])
        target_pos = conf[f'pos_total_{view}']
        sm = sparsity_match_idx(TP, FP, target_pos) if (TP.size and target_pos) else 0
        m['sparsity_match_idx']       = int(sm)
        m['sparsity_match_threshold'] = float(thr[sm]) if thr.size else float('nan')
        m['sparsity_match_sens']      = float(m['sens'][sm]) if m['sens'].size else float('nan')
        m['sparsity_match_spec']      = float(m['spec'][sm]) if m['spec'].size else float('nan')
        m['sparsity_match_f1']        = float(m['f1'][sm])   if m['f1'].size   else float('nan')
        m['pos_total']  = int(conf[f'pos_total_{view}'])
        m['neg_total']  = int(conf[f'neg_total_{view}'])
        metrics[view] = m

    return {
        'label':       label,
        'path':        str(path),
        'kind':        kind,
        'thresholds':  thr,
        'metrics':     metrics,
        'n_rows_modeled':      conf['n_rows_modeled'],
        'pos_total_unmodeled': conf['pos_total_unmodeled'],
        'pos_extra_modeled_cols': conf['pos_extra_modeled_cols'],
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def _color_cycle(n: int) -> list[str]:
    cmap = plt.get_cmap('tab10' if n <= 10 else 'tab20')
    return [cmap(i % cmap.N) for i in range(n)]


def plot_roc(per_input: list[dict], view: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    colors = _color_cycle(len(per_input))
    ax.plot([0, 1], [0, 1], color='k', lw=0.5, ls=':')
    for info, c in zip(per_input, colors):
        m = info['metrics'][view]
        order = np.argsort(m['fpr'])
        f = np.concatenate([[0.0], m['fpr'][order], [1.0]])
        t = np.concatenate([[0.0], m['tpr'][order], [1.0]])
        ax.plot(f, t, lw=1.4, color=c,
                label=f'{info["label"]}  AUROC={m["auroc"]:.3f}')
    ax.set_xlabel('False positive rate (1 − specificity)')
    ax.set_ylabel('True positive rate (sensitivity)')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_title(f'ROC vs raw observed  [{view}]')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_pr(per_input: list[dict], view: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    colors = _color_cycle(len(per_input))
    for info, c in zip(per_input, colors):
        m = info['metrics'][view]
        order = np.argsort(m['sens'])
        r = m['sens'][order]
        p = m['prec'][order]
        ax.plot(r, p, lw=1.4, color=c,
                label=f'{info["label"]}  AUPR={m["aupr"]:.3f}')
    ax.set_xlabel('Recall (sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_title(f'Precision-recall vs raw observed  [{view}]')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_sens_spec_vs_threshold(per_input: list[dict], view: str,
                                  out_path: Path) -> None:
    n = len(per_input)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows),
                              squeeze=False)
    flat = axes.flatten()
    for i, info in enumerate(per_input):
        m = info['metrics'][view]
        thr = info['thresholds']
        ax = flat[i]
        # Some pipelines have thresholds spanning many orders of magnitude.
        # Use a symlog x-axis with a small linear span around 0.
        x = thr
        ax.plot(x, m['sens'], color='#c33', lw=1.4, label='sensitivity')
        ax.plot(x, m['spec'], color='#33c', lw=1.4, label='specificity')
        ax.plot(x, m['f1'],   color='#888', lw=1.0, ls='--', label='F1')
        if (x > 0).any():
            try:
                ax.set_xscale('symlog', linthresh=max(1e-6, float(x[x > 0].min())))
            except Exception:
                pass
        bf = m.get('best_f1_idx')
        if bf is not None:
            ax.axvline(thr[bf], color='k', lw=0.6, ls=':')
        ax.set_xlabel('threshold')
        ax.set_ylim(0, 1.02)
        ax.set_title(f'{info["label"]}  AUROC={m["auroc"]:.3f}  AUPR={m["aupr"]:.3f}',
                     fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    for j in range(len(per_input), len(flat)):
        flat[j].axis('off')
    fig.suptitle(f'Sensitivity / specificity vs threshold  [{view}]')
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_metric_bars(per_input: list[dict], view: str, out_path: Path) -> None:
    labels = [i['label'] for i in per_input]
    auroc = [i['metrics'][view]['auroc'] for i in per_input]
    aupr  = [i['metrics'][view]['aupr']  for i in per_input]
    f1max = [i['metrics'][view]['best_f1'] for i in per_input]
    sens_at_match = [i['metrics'][view]['sparsity_match_sens'] for i in per_input]
    spec_at_match = [i['metrics'][view]['sparsity_match_spec'] for i in per_input]
    x = np.arange(len(labels))
    width = 0.16
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels) + 3), 4))
    bars = []
    bars.append(ax.bar(x - 2*width, auroc, width, label='AUROC',          color='#3B7EA1'))
    bars.append(ax.bar(x -   width, aupr,  width, label='AUPR',           color='#888'))
    bars.append(ax.bar(x,           f1max, width, label='max F1',         color='#C44E52'))
    bars.append(ax.bar(x +   width, sens_at_match, width, label='sens @ sparsity-match', color='#55A868'))
    bars.append(ax.bar(x + 2*width, spec_at_match, width, label='spec @ sparsity-match', color='#CCB974'))
    for grp, vals in zip(bars, (auroc, aupr, f1max, sens_at_match, spec_at_match)):
        for b, v in zip(grp, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=7)
    ax.axhline(0.5, color='k', lw=0.4, ls=':')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylim(0, 1.05); ax.set_ylabel('metric')
    ax.set_title(f'Cross-pipeline sensitivity / specificity summary  [{view}]')
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------
def write_tsv(per_input: list[dict], out_path: Path) -> None:
    cols = ['input', 'view', 'kind', 'threshold',
            'TP', 'FP', 'FN', 'TN',
            'sensitivity', 'specificity', 'precision', 'F1', 'MCC', 'accuracy']
    with open(out_path, 'w') as fh:
        fh.write('\t'.join(cols) + '\n')
        for info in per_input:
            thr = info['thresholds']
            for view in ('modeled', 'full'):
                m = info['metrics'][view]
                for k in range(thr.size):
                    fh.write('\t'.join([
                        info['label'], view, info['kind'],
                        f'{thr[k]:.6g}',
                        str(int(m['TP'][k])),
                        str(int(m['FP'][k])),
                        str(int(m['FN'][k])),
                        str(int(m['TN'][k])),
                        f"{m['sens'][k]:.6g}",
                        f"{m['spec'][k]:.6g}",
                        f"{m['prec'][k]:.6g}",
                        f"{m['f1'][k]:.6g}",
                        f"{m['mcc'][k]:.6g}",
                        f"{m['acc'][k]:.6g}",
                    ]) + '\n')


def build_summary(per_input: list[dict]) -> dict:
    out: list[dict] = []
    for info in per_input:
        for view in ('modeled', 'full'):
            m = info['metrics'][view]
            out.append({
                'label':                info['label'],
                'view':                 view,
                'kind':                 info['kind'],
                'n_thresholds':         int(info['thresholds'].size),
                'pos_total':            int(m['pos_total']),
                'neg_total':            int(m['neg_total']),
                'auroc':                float(m['auroc']),
                'aupr':                 float(m['aupr']),
                'best_f1':              float(m.get('best_f1', float('nan'))),
                'best_f1_threshold':    float(m.get('best_f1_threshold', float('nan'))),
                'best_f1_sens':         float(m.get('best_f1_sens', float('nan'))),
                'best_f1_spec':         float(m.get('best_f1_spec', float('nan'))),
                'sparsity_match_threshold': float(m['sparsity_match_threshold']),
                'sparsity_match_sens':  float(m['sparsity_match_sens']),
                'sparsity_match_spec':  float(m['sparsity_match_spec']),
                'sparsity_match_f1':    float(m['sparsity_match_f1']),
            })
    return {'inputs': out}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__ or '',
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mm-dir',  type=Path, required=True,
                    help='Directory with raw matrix.mtx.gz + regions.tsv.gz + barcodes.tsv.gz')
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--input',   type=parse_input_arg, action='append',
                    required=True, dest='inputs',
                    metavar='LABEL=PATH',
                    help='LABEL=PATH; PATH = factored/dense impute/ directory '
                         'or legacy HDF5 file. May be passed multiple times.')
    ap.add_argument('--thresholds', type=str, default=None,
                    help='Comma-separated absolute thresholds. Default: auto '
                         '(per-pipeline quantile-based).')
    ap.add_argument('--n-thresholds', type=int, default=48,
                    help='Number of auto thresholds per pipeline (default 48).')
    ap.add_argument('--chunk-rows', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-name', type=str, default='sensitivity_specificity')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    raw, raw_name_to_row, barcodes = load_raw_mm(args.mm_dir)

    if args.thresholds:
        user_thr = np.asarray(sorted(float(t.strip()) for t in args.thresholds.split(',')
                                     if t.strip()), dtype=np.float64)
        if user_thr.size == 0:
            user_thr = None
    else:
        user_thr = None

    per_input: list[dict] = []
    for label, path in args.inputs:
        info = evaluate_input(label, path, raw, raw_name_to_row, barcodes,
                              thresholds=user_thr,
                              n_thresholds=args.n_thresholds,
                              chunk_rows=args.chunk_rows,
                              rng=rng)
        per_input.append(info)
        for view in ('modeled', 'full'):
            m = info['metrics'][view]
            log.info(
                '  %s [%s]  AUROC=%.4f  AUPR=%.4f  max-F1=%.3f@t=%.3g  '
                'sens@match=%.3f spec@match=%.3f (t=%.3g)',
                label, view, m['auroc'], m['aupr'],
                m.get('best_f1', float('nan')),
                m.get('best_f1_threshold', float('nan')),
                m['sparsity_match_sens'], m['sparsity_match_spec'],
                m['sparsity_match_threshold'],
            )

    tsv_path = args.out_dir / f'{args.out_name}.tsv'
    write_tsv(per_input, tsv_path)
    log.info('Wrote %s', tsv_path)

    summary = build_summary(per_input)
    json_path = args.out_dir / f'{args.out_name}.json'
    json_path.write_text(json.dumps(summary, indent=2) + '\n')
    log.info('Wrote %s', json_path)

    for view in ('modeled', 'full'):
        plot_roc(per_input, view, args.out_dir / f'{args.out_name}_roc_{view}.png')
        plot_pr( per_input, view, args.out_dir / f'{args.out_name}_pr_{view}.png')
        plot_sens_spec_vs_threshold(
            per_input, view,
            args.out_dir / f'{args.out_name}_sens_spec_{view}.png',
        )
        plot_metric_bars(
            per_input, view,
            args.out_dir / f'{args.out_name}_bars_{view}.png',
        )
        log.info('Wrote ROC / PR / sens-spec / bars figures for view=%s', view)

    return 0


if __name__ == '__main__':
    sys.exit(main())
