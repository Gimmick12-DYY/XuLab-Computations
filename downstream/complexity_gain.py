#!/usr/bin/env python
# -----------------------------------------------------------------------------
# complexity_gain.py
#
# "How many MORE accessible bins per cell do we recover through imputation?"
#
# For every cell:
#   raw_complexity[c]  = number of bins where raw[r, c] > 0
#   imp_complexity[c]  = number of bins where imputed[r, c] > t
#   gain[c]            = imp_complexity[c] - raw_complexity[c]
#
# A threshold `t` has to be picked per pipeline (imputed scales differ). We
# offer three modes:
#
#   --threshold-mode sparsity_match (default)
#       Pick t per pipeline so #(imp > t) on the modeled rows equals
#       #(raw > 0) on those same rows. Sum-of-gain is then 0 by
#       construction; the distribution shows where each pipeline is moving
#       signal *to* (negative gain in some cells -> positive gain in
#       others, i.e. denoising redistributes mass).
#
#   --threshold-mode sparsity_match:K
#       Sparsity-match but with a multiplier on the target call count.
#       K = 2 lets each pipeline make twice as many calls as raw, so
#       sum-of-gain is positive and we measure the *complexity uplift*
#       at a per-cell level.
#
#   --threshold-mode quantile:Q
#       Top (1 - Q) fraction of imputed values genome-wide on the modeled
#       rows. e.g. quantile:0.99 keeps the top 1% of imputed values.
#
#   --threshold-mode fixed:T
#       Numeric threshold (same value across pipelines; only meaningful when
#       the pipelines are on a shared scale, e.g. CPM-like).
#
# Cells are aligned to the raw barcode order. Cells missing from a pipeline
# (e.g. cisTopic's filter dropped them) are reported as NaN gain and
# excluded from the distribution.
#
# Each --input PATH is one of:
#   * directory with factors.npz + regions.tsv + barcodes.tsv  (cisTopic / scOpen)
#   * directory with matrix.npy + regions.tsv + barcodes.tsv   (FITS / MAGIC)
#   * directory with matrix_csr.npz + regions.tsv + barcodes.tsv  (PUscOpen csr_full_mm)
#   * legacy HDF5 file with /Prc + /regions + /barcodes
#
# Outputs (under --out-dir):
#   <out>.tsv                 per-cell long-form: input, barcode, raw_complexity,
#                             imp_complexity, gain, threshold_used
#   <out>_summary.json        per-pipeline median / mean / quantiles of gain
#                             + the chosen threshold and how it was derived
#   <out>_gain_hist.png       overlaid histograms of per-cell gain
#   <out>_scatter.png         per-pipeline panels of raw_complexity vs
#                             imp_complexity (catches "low-raw cells gain
#                             disproportionately")
#   <out>_bars.png            median per-cell gain bar chart
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402
from scipy import sparse  # noqa: E402

log = logging.getLogger('complexity_gain')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


# -----------------------------------------------------------------------------
# Argument parsing + IO helpers
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


def _read_lines(path: Path) -> list[str]:
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as fh:
            return [ln.rstrip('\n') for ln in fh if ln.strip()]
    return [ln for ln in path.read_text().splitlines() if ln.strip()]


def _read_lines_either(dir_path: Path, base: str) -> list[str] | None:
    for name in (f'{base}.tsv', f'{base}.tsv.gz'):
        p = dir_path / name
        if p.exists():
            return _read_lines(p)
    return None


def detect_input_kind(path: Path) -> str:
    if path.is_file():
        if path.suffix.lower() in ('.h5', '.hdf5'):
            return 'hdf5'
        raise SystemExit(f'Unsupported input file: {path}')
    if path.is_dir():
        if (path / 'matrix_csr.npz').exists():
            return 'csr'
        if (path / 'factors.npz').exists():
            return 'factored'
        if (path / 'matrix.npy').exists():
            return 'dense'
        if list(path.glob('*.h5')) + list(path.glob('*.hdf5')):
            return 'hdf5'
    raise SystemExit(f'Cannot determine input kind from {path}')


def load_raw(mm_dir: Path) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    mtx_path = mm_dir / 'matrix.mtx.gz'
    reg_path = mm_dir / 'regions.tsv.gz'
    bar_path = mm_dir / 'barcodes.tsv.gz'
    for p in (mtx_path, reg_path, bar_path):
        if not p.exists():
            raise SystemExit(f'Missing {p}')
    log.info('Reading raw mm matrix %s', mtx_path)
    mat = sio.mmread(str(mtx_path)).tocsr()
    mat = (mat > 0).astype(np.int8)
    regs = _read_lines(reg_path)
    bars = _read_lines(bar_path)
    if mat.shape != (len(regs), len(bars)):
        raise SystemExit(
            f'Raw shape {mat.shape} vs regions/barcodes {len(regs)}/{len(bars)}'
        )
    log.info('  raw: %d regions x %d cells, nnz=%d', *mat.shape, mat.nnz)
    return mat, regs, bars


# -----------------------------------------------------------------------------
# Per-input metadata
# -----------------------------------------------------------------------------
def load_impute_regions_barcodes(path: Path, kind: str
                                  ) -> tuple[list[str], list[str]]:
    if kind == 'hdf5':
        h5_path = path if path.is_file() else (
            sorted(list(path.glob('*.h5')) + list(path.glob('*.hdf5')))[0]
        )
        with h5py.File(h5_path, 'r') as h5:
            regs = ([s.decode() if isinstance(s, (bytes, bytearray)) else str(s)
                     for s in h5['regions'][:]] if 'regions' in h5 else None)
            bars = ([s.decode() if isinstance(s, (bytes, bytearray)) else str(s)
                     for s in h5['barcodes'][:]] if 'barcodes' in h5 else None)
        if regs is None or bars is None:
            raise SystemExit(f'HDF5 {h5_path} missing /regions or /barcodes')
        return regs, bars
    regs = _read_lines_either(path, 'regions')
    bars = _read_lines_either(path, 'barcodes')
    if regs is None or bars is None:
        raise SystemExit(f'Missing regions.tsv or barcodes.tsv in {path}')
    return regs, bars


def load_pcf(npz_path: Path) -> tuple[bool, float | np.ndarray]:
    """Return (is_array, pcf) from a factors.npz. Falls back to scalar 1.0 if
    per_cell_factor is missing entirely."""
    with np.load(npz_path) as z:
        if 'per_cell_factor' not in z.files:
            return False, 1.0
        raw = z['per_cell_factor']
        if raw.ndim == 0:
            return False, float(raw)
        return True, np.asarray(raw, dtype=np.float64).ravel()


# -----------------------------------------------------------------------------
# Block iterator: yields (start, end, block_dense_C_arr) for each input.
# block has shape (end-start, n_cells) in float32 / float64.
# -----------------------------------------------------------------------------
def iter_blocks_factored(path: Path, chunk: int = 2000):
    factors = path / 'factors.npz'
    with np.load(factors) as z:
        W = np.asarray(z['W'], dtype=np.float64)
        H = np.asarray(z['H'], dtype=np.float64)
    is_arr, pcf = load_pcf(factors)
    Rm = W.shape[0]
    for s in range(0, Rm, chunk):
        e = min(s + chunk, Rm)
        block = (W[s:e] @ H)
        if is_arr:
            block = block * pcf[None, :]
        else:
            block = block * float(pcf)
        yield s, e, block


def iter_blocks_dense(path: Path, chunk: int = 2000):
    arr = np.load(path / 'matrix.npy', mmap_mode='r')
    Rm = arr.shape[0]
    for s in range(0, Rm, chunk):
        e = min(s + chunk, Rm)
        yield s, e, np.asarray(arr[s:e], dtype=np.float64)


def iter_blocks_csr(path: Path, chunk: int = 2000):
    mat = sparse.load_npz(path / 'matrix_csr.npz').tocsr()
    Rm = mat.shape[0]
    for s in range(0, Rm, chunk):
        e = min(s + chunk, Rm)
        yield s, e, mat[s:e].toarray().astype(np.float64)


def iter_blocks_hdf5(path: Path, chunk: int = 2000):
    h5_path = path if path.is_file() else (
        sorted(list(path.glob('*.h5')) + list(path.glob('*.hdf5')))[0]
    )
    with h5py.File(h5_path, 'r') as h5:
        ds = h5['Prc']
        Rm = ds.shape[0]
        for s in range(0, Rm, chunk):
            e = min(s + chunk, Rm)
            yield s, e, ds[s:e, :].astype(np.float64)


def iter_blocks(path: Path, kind: str, chunk: int = 2000):
    if kind == 'factored':  yield from iter_blocks_factored(path, chunk)
    elif kind == 'dense':   yield from iter_blocks_dense(path, chunk)
    elif kind == 'csr':     yield from iter_blocks_csr(path, chunk)
    elif kind == 'hdf5':    yield from iter_blocks_hdf5(path, chunk)
    else:
        raise SystemExit(f'Unsupported kind {kind!r}')


# -----------------------------------------------------------------------------
# Threshold estimation
# -----------------------------------------------------------------------------
def parse_threshold_mode(s: str) -> tuple[str, float]:
    """Returns (mode, param). mode in {sparsity_match, quantile, fixed};
    param is multiplier (for sparsity_match), q (for quantile), or t (for fixed)."""
    s = (s or 'sparsity_match').strip()
    if s == 'sparsity_match':
        return 'sparsity_match', 1.0
    if s.startswith('sparsity_match:'):
        try:
            return 'sparsity_match', float(s.split(':', 1)[1])
        except ValueError as e:
            raise SystemExit(f'Cannot parse multiplier from {s!r}: {e}')
    if s.startswith('quantile:'):
        try:
            q = float(s.split(':', 1)[1])
        except ValueError as e:
            raise SystemExit(f'Cannot parse quantile from {s!r}: {e}')
        if not (0.0 <= q <= 1.0):
            raise SystemExit(f'quantile must be in [0, 1], got {q}')
        return 'quantile', q
    if s.startswith('fixed:'):
        try:
            return 'fixed', float(s.split(':', 1)[1])
        except ValueError as e:
            raise SystemExit(f'Cannot parse fixed threshold from {s!r}: {e}')
    raise SystemExit(
        f'Unknown threshold mode {s!r}. '
        f'Expected sparsity_match[:K], quantile:Q, fixed:T'
    )


def sample_imputed_values(path: Path, kind: str, n_rows: int, n_cells: int,
                           n_sample: int, rng: np.random.Generator) -> np.ndarray:
    """Random sample of imputed values from the modeled subspace."""
    n_sample = min(int(n_sample), int(n_rows) * int(n_cells))
    if n_sample <= 0:
        return np.zeros(0, dtype=np.float64)
    # Sample rows and cols uniformly. We do the per-cell lookup inline rather
    # than via iter_blocks so we don't materialise a chunk we don't need.
    sample_rows = rng.integers(0, n_rows, size=n_sample, dtype=np.int64)
    sample_cols = rng.integers(0, n_cells, size=n_sample, dtype=np.int64)
    if kind == 'factored':
        with np.load(path / 'factors.npz') as z:
            W = np.asarray(z['W'], dtype=np.float64)
            H = np.asarray(z['H'], dtype=np.float64)
        is_arr, pcf = load_pcf(path / 'factors.npz')
        base = np.einsum('ik,ki->i', W[sample_rows], H[:, sample_cols])
        return base * (pcf[sample_cols] if is_arr else float(pcf))
    if kind == 'dense':
        arr = np.load(path / 'matrix.npy', mmap_mode='r')
        return np.asarray(arr[sample_rows, sample_cols], dtype=np.float64)
    if kind == 'csr':
        mat = sparse.load_npz(path / 'matrix_csr.npz').tocsr()
        return np.asarray(mat[sample_rows, sample_cols]).ravel().astype(np.float64)
    if kind == 'hdf5':
        h5_path = path if path.is_file() else (
            sorted(list(path.glob('*.h5')) + list(path.glob('*.hdf5')))[0]
        )
        out = np.empty(n_sample, dtype=np.float64)
        with h5py.File(h5_path, 'r') as h5:
            ds = h5['Prc']
            order = np.argsort(sample_rows, kind='stable')
            sorted_rows = sample_rows[order]
            sorted_cols = sample_cols[order]
            uniq, inv = np.unique(sorted_rows, return_inverse=True)
            # For a 1-shot sample, load the unique rows once.
            loaded = ds[list(uniq), :]
            out[order] = loaded[inv, sorted_cols].astype(np.float64)
        return out
    raise SystemExit(f'Unsupported kind {kind!r}')


def estimate_threshold(label: str, path: Path, kind: str,
                        n_modeled_rows: int, n_cells: int,
                        target_nnz_in_modeled: int,
                        mode: str, param: float,
                        rng: np.random.Generator,
                        sample_size: int = 4_000_000
                        ) -> tuple[float, str]:
    """Return (threshold, origin_string) per the threshold mode."""
    if mode == 'fixed':
        return float(param), f'fixed:{param:.6g}'

    sample = sample_imputed_values(path, kind, n_modeled_rows, n_cells,
                                   sample_size, rng)
    if sample.size == 0:
        return 0.0, 'empty_sample'

    if mode == 'sparsity_match':
        total_entries = int(n_modeled_rows) * int(n_cells)
        target = int(round(target_nnz_in_modeled * float(param)))
        target = max(1, min(target, total_entries))
        # Fraction of entries that should exceed t = target / total
        target_fraction = target / total_entries
        # We want quantile q = 1 - target_fraction
        q = max(0.0, min(1.0, 1.0 - target_fraction))
        t = float(np.quantile(sample, q))
        return t, (f'sparsity_match:{param} (target_calls={target}, '
                   f'q={q:.4g} of {sample.size} samples -> t={t:.4g})')

    if mode == 'quantile':
        q = float(param)
        t = float(np.quantile(sample, q))
        return t, f'quantile:{q} -> t={t:.4g}'

    raise SystemExit(f'Unknown mode {mode!r}')


# -----------------------------------------------------------------------------
# Per-cell counts
# -----------------------------------------------------------------------------
def per_cell_count_above(path: Path, kind: str, threshold: float,
                          chunk: int = 2000) -> np.ndarray:
    """Stream the imputed matrix and accumulate per-cell counts of values > t."""
    counts = None
    for _s, _e, block in iter_blocks(path, kind, chunk):
        mask = block > threshold
        col_counts = mask.sum(axis=0).astype(np.int64)
        if counts is None:
            counts = np.zeros(col_counts.shape[0], dtype=np.int64)
        counts += col_counts
    if counts is None:
        return np.zeros(0, dtype=np.int64)
    return counts


def raw_count_per_cell_on_modeled(raw: sparse.csr_matrix,
                                   modeled_row_idx: np.ndarray,
                                   raw_col_idx: np.ndarray | None = None
                                   ) -> np.ndarray:
    """Raw nonzero count per cell on the rows in `modeled_row_idx`.

    If raw_col_idx is given, also restrict / reorder to those columns.
    """
    sub = raw[modeled_row_idx]                      # (n_modeled, n_raw_cells)
    if raw_col_idx is not None:
        sub = sub[:, raw_col_idx]
    # Treat any nonzero entry as a positive raw observation.
    return np.asarray((sub != 0).sum(axis=0)).ravel().astype(np.int64)


# -----------------------------------------------------------------------------
# Per-input evaluation
# -----------------------------------------------------------------------------
def evaluate_input(label: str, path: Path,
                    raw: sparse.csr_matrix, raw_regs: list[str],
                    raw_bars: list[str], raw_bar_to_idx: dict[str, int],
                    mode: str, mode_param: float,
                    chunk_rows: int, rng: np.random.Generator,
                    sample_size: int) -> dict:
    kind = detect_input_kind(path)
    log.info('Input %r (%s) -> %s', label, kind, path)
    imp_regs, imp_bars = load_impute_regions_barcodes(path, kind)
    n_modeled_rows = len(imp_regs)
    n_imp_cells = len(imp_bars)
    log.info('  modeled shape: %d regions x %d cells', n_modeled_rows, n_imp_cells)

    # Map imputed rows back to raw mm row indices for raw_count computation.
    raw_name_to_row = {r: i for i, r in enumerate(raw_regs)}
    modeled_raw_rows = np.fromiter(
        (raw_name_to_row.get(r, -1) for r in imp_regs),
        dtype=np.int64, count=n_modeled_rows,
    )
    n_unmatched = int((modeled_raw_rows < 0).sum())
    if n_unmatched:
        log.warning('  %d / %d imputed region names absent from raw mm; '
                    'ignored for raw_count computation.',
                    n_unmatched, n_modeled_rows)
    modeled_in_raw = modeled_raw_rows[modeled_raw_rows >= 0]

    # Map imputed cells to raw mm col indices.
    raw_cols_for_imp = np.fromiter(
        (raw_bar_to_idx.get(b, -1) for b in imp_bars),
        dtype=np.int64, count=n_imp_cells,
    )
    n_missing_cells = int((raw_cols_for_imp < 0).sum())
    if n_missing_cells:
        log.warning('  %d / %d imputed cell barcodes absent from raw mm; '
                    'their gains will be NaN.',
                    n_missing_cells, n_imp_cells)

    # ---- Raw count per imputed cell, over modeled rows -------------------
    raw_per_imp_cell = np.zeros(n_imp_cells, dtype=np.int64)
    valid_cells = raw_cols_for_imp >= 0
    if valid_cells.any() and modeled_in_raw.size:
        raw_counts_on_valid = raw_count_per_cell_on_modeled(
            raw, modeled_in_raw, raw_cols_for_imp[valid_cells]
        )
        raw_per_imp_cell[valid_cells] = raw_counts_on_valid

    target_nnz = int(raw_per_imp_cell.sum())   # raw nnz on modeled subspace
    log.info('  raw nonzeros on modeled subspace: %d (median per cell=%d)',
             target_nnz, int(np.median(raw_per_imp_cell)) if raw_per_imp_cell.size else 0)

    # ---- Threshold ---------------------------------------------------------
    t, t_origin = estimate_threshold(label, path, kind,
                                     n_modeled_rows, n_imp_cells,
                                     target_nnz, mode, mode_param,
                                     rng, sample_size)
    log.info('  threshold: %.6g  (%s)', t, t_origin)

    # ---- Per-cell count above threshold ----------------------------------
    log.info('  streaming per-cell counts (chunk_rows=%d) ...', chunk_rows)
    imp_per_imp_cell = per_cell_count_above(path, kind, t, chunk_rows)
    if imp_per_imp_cell.size != n_imp_cells:
        log.warning('  per_cell_count length %d != n_imp_cells %d',
                    imp_per_imp_cell.size, n_imp_cells)
        imp_per_imp_cell = np.resize(imp_per_imp_cell, n_imp_cells)

    gain_per_imp_cell = imp_per_imp_cell.astype(np.int64) - raw_per_imp_cell.astype(np.int64)

    log.info('  median raw=%.0f, median imp=%.0f, median gain=%+.0f, '
             'total gain=%+d',
             float(np.median(raw_per_imp_cell)),
             float(np.median(imp_per_imp_cell)),
             float(np.median(gain_per_imp_cell)),
             int(gain_per_imp_cell.sum()))

    # ---- Align to raw barcode order --------------------------------------
    raw_aligned = np.full(len(raw_bars), np.nan, dtype=np.float64)
    imp_aligned = np.full(len(raw_bars), np.nan, dtype=np.float64)
    gain_aligned = np.full(len(raw_bars), np.nan, dtype=np.float64)
    for j, raw_c in enumerate(raw_cols_for_imp):
        if raw_c < 0:
            continue
        raw_aligned[raw_c] = float(raw_per_imp_cell[j])
        imp_aligned[raw_c] = float(imp_per_imp_cell[j])
        gain_aligned[raw_c] = float(gain_per_imp_cell[j])

    return {
        'label':        label,
        'path':         str(path),
        'kind':         kind,
        'threshold':    float(t),
        'threshold_origin': t_origin,
        'mode':         mode,
        'mode_param':   float(mode_param),
        'n_modeled_rows': int(n_modeled_rows),
        'n_imp_cells':  int(n_imp_cells),
        'raw_aligned':  raw_aligned,
        'imp_aligned':  imp_aligned,
        'gain_aligned': gain_aligned,
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def _color_cycle(n: int) -> list[str]:
    cmap = plt.get_cmap('tab10' if n <= 10 else 'tab20')
    return [cmap(i % cmap.N) for i in range(n)]


def plot_gain_hist(per_input: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = _color_cycle(len(per_input))
    for d, c in zip(per_input, colors):
        v = d['gain_aligned']
        v = v[~np.isnan(v)]
        if v.size == 0:
            continue
        ax.hist(v, bins=80, alpha=0.45, label=d['label'], color=c)
    ax.axvline(0.0, color='k', lw=0.6, ls=':')
    ax.set_xlabel('per-cell complexity gain (imp_count − raw_count)')
    ax.set_ylabel('# cells')
    ax.set_title('Per-cell complexity gain across pipelines')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_scatter(per_input: list[dict], out_path: Path) -> None:
    n = len(per_input)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows),
                              squeeze=False)
    flat = axes.flatten()
    for i, d in enumerate(per_input):
        ax = flat[i]
        raw_v = d['raw_aligned']; imp_v = d['imp_aligned']
        ok = ~np.isnan(raw_v) & ~np.isnan(imp_v)
        if not ok.any():
            ax.set_title(f'{d["label"]} (empty)'); ax.axis('off'); continue
        x = raw_v[ok] + 1.0   # log scale: +1 to handle zeros
        y = imp_v[ok] + 1.0
        ax.scatter(x, y, s=4, alpha=0.4, color='#3B7EA1', rasterized=True)
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], color='k', lw=0.6, ls=':')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('raw_complexity_per_cell (+1)')
        ax.set_ylabel('imp_complexity_per_cell (+1)')
        gain = (imp_v[ok] - raw_v[ok])
        ax.set_title(f'{d["label"]}  t={d["threshold"]:.3g}\n'
                     f'median gain = {float(np.median(gain)):+.0f}',
                     fontsize=10)
    for k in range(len(per_input), len(flat)):
        flat[k].axis('off')
    fig.suptitle('Per-cell complexity: raw vs imputed (diagonal = no gain)')
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_bars(per_input: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(per_input) + 2), 4.5))
    labels = [d['label'] for d in per_input]
    medians = []
    for d in per_input:
        v = d['gain_aligned']
        v = v[~np.isnan(v)]
        medians.append(float(np.median(v)) if v.size else float('nan'))
    x = np.arange(len(labels))
    bars = ax.bar(x, medians, color='#55A868')
    for b, val in zip(bars, medians):
        if not np.isnan(val):
            ax.text(b.get_x() + b.get_width() / 2, val,
                    f'{val:+.0f}', ha='center',
                    va='bottom' if val >= 0 else 'top', fontsize=9)
    ax.axhline(0.0, color='k', lw=0.5, ls=':')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('median per-cell complexity gain')
    ax.set_title('Median per-cell complexity gain across pipelines')
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------
def _stats(v: np.ndarray) -> dict:
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {k: float('nan') for k in
                ('mean', 'median', 'min', 'max', 'p10', 'p90', 'std')}
    return {
        'mean':   float(v.mean()),
        'median': float(np.median(v)),
        'std':    float(v.std()),
        'min':    float(v.min()),
        'max':    float(v.max()),
        'p10':    float(np.quantile(v, 0.10)),
        'p90':    float(np.quantile(v, 0.90)),
    }


def write_tsv(per_input: list[dict], raw_bars: list[str], out_path: Path) -> None:
    with open(out_path, 'w') as fh:
        fh.write('input\tkind\tbarcode\tthreshold\traw_complexity\t'
                 'imp_complexity\tgain\n')
        for d in per_input:
            t = d['threshold']
            for j, bc in enumerate(raw_bars):
                raw_v = d['raw_aligned'][j]
                imp_v = d['imp_aligned'][j]
                gain_v = d['gain_aligned'][j]
                if np.isnan(gain_v):
                    continue
                fh.write(f'{d["label"]}\t{d["kind"]}\t{bc}\t{t:.6g}\t'
                         f'{int(raw_v)}\t{int(imp_v)}\t{int(gain_v)}\n')


def build_summary(per_input: list[dict]) -> dict:
    out: list[dict] = []
    for d in per_input:
        gain = d['gain_aligned']
        raw_v = d['raw_aligned']
        imp_v = d['imp_aligned']
        ok = ~np.isnan(gain)
        out.append({
            'label':            d['label'],
            'kind':             d['kind'],
            'path':             d['path'],
            'threshold':        float(d['threshold']),
            'threshold_origin': d['threshold_origin'],
            'mode':             d['mode'],
            'mode_param':       float(d['mode_param']),
            'n_modeled_rows':   int(d['n_modeled_rows']),
            'n_cells_evaluated': int(ok.sum()),
            'raw_complexity_stats': _stats(raw_v),
            'imp_complexity_stats': _stats(imp_v),
            'gain_stats':            _stats(gain),
            'total_gain':            float(np.nansum(gain)),
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
                    required=True, dest='inputs', metavar='LABEL=PATH',
                    help='LABEL=PATH; repeatable.')
    ap.add_argument('--threshold-mode', type=str, default='sparsity_match',
                    help='sparsity_match[:K] | quantile:Q | fixed:T '
                         '(default: sparsity_match, i.e. each pipeline picks t '
                         'so #(imp > t) on modeled rows == #(raw > 0) on modeled rows)')
    ap.add_argument('--chunk-rows', type=int, default=2000)
    ap.add_argument('--sample-size', type=int, default=4_000_000,
                    help='Random sample size for threshold quantile estimation.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-name', type=str, default='complexity_gain')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    mode, mode_param = parse_threshold_mode(args.threshold_mode)
    log.info('Threshold mode: %s (param=%s)', mode, mode_param)

    raw, raw_regs, raw_bars = load_raw(args.mm_dir)
    raw_bar_to_idx = {b: i for i, b in enumerate(raw_bars)}

    per_input: list[dict] = []
    for label, path in args.inputs:
        info = evaluate_input(label, path, raw, raw_regs, raw_bars,
                              raw_bar_to_idx, mode, mode_param,
                              args.chunk_rows, rng, args.sample_size)
        per_input.append(info)

    tsv_path = args.out_dir / f'{args.out_name}.tsv'
    log.info('Writing %s', tsv_path)
    write_tsv(per_input, raw_bars, tsv_path)

    summary = build_summary(per_input)
    json_path = args.out_dir / f'{args.out_name}_summary.json'
    json_path.write_text(json.dumps(summary, indent=2) + '\n')
    log.info('Wrote %s', json_path)
    for r in summary['inputs']:
        g = r['gain_stats']
        log.info('  %-12s  t=%.4g  raw_med=%.0f  imp_med=%.0f  '
                 'gain_med=%+.0f  gain_p10=%+.0f  gain_p90=%+.0f',
                 r['label'], r['threshold'],
                 r['raw_complexity_stats']['median'],
                 r['imp_complexity_stats']['median'],
                 g['median'], g['p10'], g['p90'])

    plot_gain_hist(per_input, args.out_dir / f'{args.out_name}_gain_hist.png')
    plot_scatter (per_input, args.out_dir / f'{args.out_name}_scatter.png')
    plot_bars    (per_input, args.out_dir / f'{args.out_name}_bars.png')
    log.info('Wrote gain_hist / scatter / bars figures.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
