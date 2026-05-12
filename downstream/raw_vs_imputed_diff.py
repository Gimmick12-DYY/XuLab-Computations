#!/usr/bin/env python
# -----------------------------------------------------------------------------
# raw_vs_imputed_diff.py
#
# At every raw-nonzero (region, cell) coordinate, look up the imputed value
# from each pipeline and quantify the "delta":
#
#     delta = imputed[r, c] - raw[r, c]
#
# A coordinate where the raw matrix saw a fragment (raw == 1 after binarising)
# but the imputed value at the same position is small is a "lost positive".
# We report:
#
#   * Distribution of imputed values at raw=1 coordinates (histogram).
#   * Distribution of delta = imputed - raw (negative values = imputer
#     downgraded a raw positive).
#   * Lost-positive counts at multiple thresholds: a raw-positive coordinate
#     is "lost" if its imputed value is <= t.
#   * Per-cell and per-bin lost-positive counts (catch pipelines that lose
#     signal in specific cells / regions).
#   * For context, the imputed-value distribution at a random sample of
#     raw-zero coordinates ("synthetic positives" the imputer added).
#
# Alignment is by NAME, not row index: region names (chr:start-end) and cell
# barcodes are matched across the raw mm matrix and each imputed input. Raw
# nonzero coordinates that fall outside an imputed input's modeled region set
# (or whose cell barcode isn't present) are counted as `imputed = 0`, since
# downstream tools rightly treat them as predicted-negative.
#
# Each --input PATH is one of:
#   * directory with factors.npz + regions.tsv + barcodes.tsv   (cisTopic / scOpen)
#   * directory with matrix.npy + regions.tsv + barcodes.tsv    (FITS / MAGIC)
#   * directory with matrix_csr.npz + regions.tsv + barcodes.tsv (PUscOpen full-mm)
#   * legacy HDF5 file with /Prc + /regions + /barcodes
#
# Outputs (under --out-dir):
#   raw_vs_imputed_diff.tsv          per-pipeline per-threshold lost-positive table
#   raw_vs_imputed_diff_per_cell.tsv per-cell lost-positive counts at default threshold
#   raw_vs_imputed_diff.json         summary stats
#   raw_vs_imputed_diff_imp_at_raw_pos.png   imputed-value hist at raw=1 coords
#   raw_vs_imputed_diff_delta.png            delta = imputed - raw histogram
#   raw_vs_imputed_diff_per_cell_lost.png    boxplot of per-cell lost counts
#   raw_vs_imputed_diff_imp_at_raw_zero.png  imputed-value hist at sampled raw=0 coords
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

log = logging.getLogger('rvi_diff')
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
    log.info('  raw: %d regions x %d cells, nnz=%d (density=%.3g)',
             mat.shape[0], mat.shape[1], mat.nnz,
             mat.nnz / float(mat.shape[0] * mat.shape[1]))
    return mat, regs, bars


def load_impute_regions_barcodes(path: Path, kind: str
                                  ) -> tuple[list[str], list[str]]:
    if kind == 'hdf5':
        h5_path = path if path.is_file() else (
            sorted(list(path.glob('*.h5')) + list(path.glob('*.hdf5')))[0]
        )
        with h5py.File(h5_path, 'r') as h5:
            regs = [s.decode() if isinstance(s, (bytes, bytearray)) else str(s)
                    for s in h5['regions'][:]] if 'regions' in h5 else None
            bars = [s.decode() if isinstance(s, (bytes, bytearray)) else str(s)
                    for s in h5['barcodes'][:]] if 'barcodes' in h5 else None
        if regs is None or bars is None:
            raise SystemExit(f'HDF5 {h5_path} missing /regions or /barcodes')
        return regs, bars
    regs = _read_lines_either(path, 'regions')
    bars = _read_lines_either(path, 'barcodes')
    if regs is None or bars is None:
        raise SystemExit(f'Missing regions.tsv or barcodes.tsv in {path}')
    return regs, bars


# -----------------------------------------------------------------------------
# Imputed value lookup at coordinate pairs
# -----------------------------------------------------------------------------
def imputed_at_coords(path: Path, kind: str,
                      rows: np.ndarray, cols: np.ndarray,
                      chunk: int = 200_000) -> np.ndarray:
    """Return imputed[rows[i], cols[i]] for each i. rows/cols are in the
    imputed matrix's row/col index space (already mapped from raw)."""
    if rows.size == 0:
        return np.zeros(0, dtype=np.float64)

    if kind == 'factored':
        with np.load(path / 'factors.npz') as z:
            W = np.asarray(z['W'], dtype=np.float64)
            H = np.asarray(z['H'], dtype=np.float64)
            if 'per_cell_factor' in z.files:
                pcf_raw = z['per_cell_factor']
                pcf_is_array = pcf_raw.ndim > 0
                pcf_arr = (np.asarray(pcf_raw, dtype=np.float64).ravel()
                           if pcf_is_array else None)
                pcf_scalar = float(pcf_raw) if not pcf_is_array else None
            else:
                pcf_is_array = False
                pcf_arr = None
                pcf_scalar = 1.0
        if pcf_is_array and pcf_arr.shape[0] != H.shape[1]:
            raise SystemExit(
                f'per_cell_factor length {pcf_arr.shape[0]} != H.shape[1] {H.shape[1]} '
                f'in {path}')
        out = np.empty(rows.size, dtype=np.float64)
        for s in range(0, rows.size, chunk):
            e = min(s + chunk, rows.size)
            base = np.einsum('ik,ki->i', W[rows[s:e]], H[:, cols[s:e]])
            # Per-coord: imp[r,c] = (W[r] @ H[:, c]) * pcf[c]
            if pcf_is_array:
                out[s:e] = base * pcf_arr[cols[s:e]]
            else:
                out[s:e] = base * pcf_scalar
        return out

    if kind == 'dense':
        arr = np.load(path / 'matrix.npy', mmap_mode='r')
        out = np.empty(rows.size, dtype=np.float64)
        for s in range(0, rows.size, chunk):
            e = min(s + chunk, rows.size)
            out[s:e] = np.asarray(arr[rows[s:e], cols[s:e]], dtype=np.float64)
        return out

    if kind == 'csr':
        mat = sparse.load_npz(path / 'matrix_csr.npz').tocsr()
        # Scipy CSR supports paired (row, col) advanced indexing as 1D output.
        out = np.empty(rows.size, dtype=np.float64)
        for s in range(0, rows.size, chunk):
            e = min(s + chunk, rows.size)
            sub = np.asarray(mat[rows[s:e], cols[s:e]]).ravel()
            out[s:e] = sub.astype(np.float64)
        return out

    if kind == 'hdf5':
        h5_path = path if path.is_file() else (
            sorted(list(path.glob('*.h5')) + list(path.glob('*.hdf5')))[0]
        )
        out = np.empty(rows.size, dtype=np.float64)
        with h5py.File(h5_path, 'r') as h5:
            ds = h5['Prc']
            order = np.argsort(rows, kind='stable')
            sorted_rows = rows[order]
            sorted_cols = cols[order]
            for s in range(0, sorted_rows.size, chunk):
                e = min(s + chunk, sorted_rows.size)
                sub_rows = sorted_rows[s:e]
                sub_cols = sorted_cols[s:e]
                uniq, inv = np.unique(sub_rows, return_inverse=True)
                loaded = ds[list(uniq), :]
                out[order[s:e]] = loaded[inv, sub_cols].astype(np.float64)
        return out

    raise SystemExit(f'Unsupported kind {kind!r} for value lookup')


# -----------------------------------------------------------------------------
# Per-input evaluation
# -----------------------------------------------------------------------------
def evaluate_input(label: str, path: Path,
                    raw_rows: np.ndarray, raw_cols: np.ndarray,
                    raw_vals: np.ndarray,
                    raw_regs: list[str], raw_bars: list[str],
                    zero_sample_rows: np.ndarray,
                    zero_sample_cols: np.ndarray,
                    thresholds: list[float]) -> dict:
    kind = detect_input_kind(path)
    log.info('Input %r (%s) -> %s', label, kind, path)
    imp_regs, imp_bars = load_impute_regions_barcodes(path, kind)
    log.info('  imputed shape: %d regions x %d cells',
             len(imp_regs), len(imp_bars))

    # Row map: raw_row_idx -> imputed_row_idx (or -1)
    name_to_imp_row = {nm: i for i, nm in enumerate(imp_regs)}
    raw_row_to_imp = np.fromiter(
        (name_to_imp_row.get(raw_regs[r], -1) for r in range(len(raw_regs))),
        dtype=np.int64, count=len(raw_regs),
    )
    bar_to_imp_col = {b: i for i, b in enumerate(imp_bars)}
    raw_col_to_imp = np.fromiter(
        (bar_to_imp_col.get(raw_bars[c], -1) for c in range(len(raw_bars))),
        dtype=np.int64, count=len(raw_bars),
    )

    # ---- Map raw-nonzero coords ---------------------------------------------
    imp_rows = raw_row_to_imp[raw_rows]
    imp_cols = raw_col_to_imp[raw_cols]
    in_imp = (imp_rows >= 0) & (imp_cols >= 0)
    n_total = int(in_imp.size)
    n_modeled = int(in_imp.sum())
    n_unmodeled = n_total - n_modeled
    log.info('  raw-positive coords: %d total; %d in modeled space (%.2f%%); '
             '%d unmodeled (counted as imputed=0)',
             n_total, n_modeled, 100.0 * n_modeled / max(n_total, 1),
             n_unmodeled)

    # ---- Look up imputed values ---------------------------------------------
    imp_vals = np.zeros(n_total, dtype=np.float64)
    if n_modeled:
        imp_vals[in_imp] = imputed_at_coords(
            path, kind,
            imp_rows[in_imp].astype(np.int64),
            imp_cols[in_imp].astype(np.int64),
        )
    delta = imp_vals - raw_vals.astype(np.float64)
    log.info('  imp@raw1: mean=%.4g median=%.4g min=%.4g max=%.4g  '
             'delta(mean)=%+.4g',
             float(imp_vals.mean()), float(np.median(imp_vals)),
             float(imp_vals.min()), float(imp_vals.max()),
             float(delta.mean()))

    # ---- Lost positive counts at thresholds ---------------------------------
    rows_out: list[dict] = []
    for t in thresholds:
        lost_mask = imp_vals <= t
        lost = int(lost_mask.sum())
        recovered = n_total - lost
        rows_out.append({
            'input':             label,
            'kind':              kind,
            'threshold':         float(t),
            'n_raw_positive':    int(n_total),
            'n_lost':            lost,
            'n_recovered':       recovered,
            'lost_fraction':     lost / max(n_total, 1),
            'recovered_fraction': recovered / max(n_total, 1),
        })

    # ---- Per-cell and per-bin lost counts at the default threshold ----------
    default_t = thresholds[0] if thresholds else 0.0
    lost_mask_default = imp_vals <= default_t
    per_cell_lost = np.bincount(raw_cols[lost_mask_default],
                                minlength=len(raw_bars)).astype(np.int64)
    per_bin_lost = np.bincount(raw_rows[lost_mask_default],
                               minlength=len(raw_regs)).astype(np.int64)

    # ---- Imputed-value distribution at sampled raw=0 coordinates ------------
    if zero_sample_rows.size:
        zr_imp = raw_row_to_imp[zero_sample_rows]
        zc_imp = raw_col_to_imp[zero_sample_cols]
        zin = (zr_imp >= 0) & (zc_imp >= 0)
        zero_imp_vals = np.zeros(zero_sample_rows.size, dtype=np.float64)
        if zin.any():
            zero_imp_vals[zin] = imputed_at_coords(
                path, kind,
                zr_imp[zin].astype(np.int64),
                zc_imp[zin].astype(np.int64),
            )
        log.info('  imp@raw0 (n=%d): mean=%.4g median=%.4g max=%.4g  '
                 'frac>%.3g=%.3g',
                 zero_imp_vals.size, float(zero_imp_vals.mean()),
                 float(np.median(zero_imp_vals)),
                 float(zero_imp_vals.max()),
                 default_t,
                 float((zero_imp_vals > default_t).mean()))
    else:
        zero_imp_vals = np.zeros(0, dtype=np.float64)

    return {
        'label':           label,
        'path':            str(path),
        'kind':            kind,
        'n_raw_positive':  int(n_total),
        'n_modeled':       n_modeled,
        'n_unmodeled':     n_unmodeled,
        'imp_vals':        imp_vals,
        'delta':           delta,
        'per_threshold':   rows_out,
        'per_cell_lost':   per_cell_lost,
        'per_bin_lost':    per_bin_lost,
        'zero_imp_vals':   zero_imp_vals,
        'default_threshold': default_t,
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def _color_cycle(n: int) -> list[str]:
    cmap = plt.get_cmap('tab10' if n <= 10 else 'tab20')
    return [cmap(i % cmap.N) for i in range(n)]


def plot_imp_at_raw_pos(per_input: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = _color_cycle(len(per_input))
    # Use log1p so the spike at imp=0 is visible alongside large imp values.
    for d, c in zip(per_input, colors):
        v = d['imp_vals']
        if v.size == 0:
            continue
        ax.hist(np.log1p(np.maximum(v, 0.0)), bins=80, alpha=0.45,
                label=d['label'], color=c)
    ax.set_xlabel('log1p(imputed value)  at  raw == 1  coordinates')
    ax.set_ylabel('# coordinates')
    ax.set_title('Imputed-value distribution at raw-positive coordinates')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_delta(per_input: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = _color_cycle(len(per_input))
    for d, c in zip(per_input, colors):
        v = d['delta']
        if v.size == 0:
            continue
        # Clip the extreme right tail so the histogram stays interpretable
        # when imputed values are on a CPM-like scale.
        v_clip = np.clip(v, -1.0, max(1.0, float(np.quantile(v, 0.99))))
        ax.hist(v_clip, bins=80, alpha=0.45, label=d['label'], color=c)
    ax.axvline(0.0, color='k', lw=0.6, ls=':')
    ax.set_xlabel('delta = imputed - raw    (raw is 1 at every plotted coord)')
    ax.set_ylabel('# coordinates')
    ax.set_title('Imputed - raw at raw-positive coordinates  '
                 '(negative = imputer underestimated)')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_per_cell_lost(per_input: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(per_input) + 2), 5))
    labels = [d['label'] for d in per_input]
    arrays = [d['per_cell_lost'][d['per_cell_lost'] > 0] for d in per_input]
    arrays = [np.clip(a, 1, None) if a.size else np.array([1.0])
              for a in arrays]
    ax.boxplot(arrays, labels=labels, showfliers=False, widths=0.6)
    ax.set_yscale('log')
    ax.set_ylabel('# raw-positive coords lost per cell (log scale)')
    title_t = per_input[0]['default_threshold'] if per_input else 0.0
    ax.set_title(f'Per-cell lost-positive count at imp <= {title_t:g}')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_imp_at_raw_zero(per_input: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = _color_cycle(len(per_input))
    for d, c in zip(per_input, colors):
        v = d['zero_imp_vals']
        if v.size == 0:
            continue
        ax.hist(np.log1p(np.maximum(v, 0.0)), bins=80, alpha=0.45,
                label=d['label'], color=c)
    ax.set_xlabel('log1p(imputed value)  at  raw == 0  coordinates')
    ax.set_ylabel('# coordinates')
    ax.set_title('Imputed-value distribution at sampled raw-zero coordinates  '
                 '(imputed-only "gains")')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


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
    ap.add_argument('--out-name', type=str, default='raw_vs_imputed_diff')
    ap.add_argument('--thresholds', type=str,
                    default='0,0.001,0.01,0.05,0.1,0.5,1.0',
                    help='Comma-separated thresholds. A raw-positive coord is '
                         '"lost" when imputed <= t. Default covers a few '
                         'orders of magnitude.')
    ap.add_argument('--n-zero-sample', type=int, default=200_000,
                    help='Number of raw-zero coordinates to sample for the '
                         '"imputed at raw=0" view (0 disables).')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    thresholds = sorted(float(t.strip()) for t in args.thresholds.split(',')
                        if t.strip())

    raw, raw_regs, raw_bars = load_raw(args.mm_dir)
    coo = raw.tocoo()
    raw_rows = coo.row.astype(np.int64)
    raw_cols = coo.col.astype(np.int64)
    raw_vals = coo.data.astype(np.int8)   # all 1 after binarising
    log.info('Raw-positive coordinates: %d', raw_rows.size)

    # Sample raw=0 coordinates for the "imputed at raw=0" view.
    R, C = raw.shape
    n_zero = int(args.n_zero_sample)
    if n_zero > 0:
        # Reject-sample from the global grid; raw is ~99.998 % zero so
        # rejection has ~1 round.
        zero_rows = np.empty(0, dtype=np.int64)
        zero_cols = np.empty(0, dtype=np.int64)
        nz_lin_sorted = np.sort(
            raw_rows.astype(np.int64) * C + raw_cols.astype(np.int64)
        )
        target = n_zero
        while zero_rows.size < target:
            need = target - zero_rows.size
            batch = int(need * 1.05) + 16
            tr = rng.integers(0, R, size=batch, dtype=np.int64)
            tc = rng.integers(0, C, size=batch, dtype=np.int64)
            lin = tr * C + tc
            keep = ~np.isin(lin, nz_lin_sorted, assume_unique=False)
            zero_rows = np.concatenate([zero_rows, tr[keep]])[:target]
            zero_cols = np.concatenate([zero_cols, tc[keep]])[:target]
        log.info('Sampled %d raw-zero coordinates for context', zero_rows.size)
    else:
        zero_rows = np.empty(0, dtype=np.int64)
        zero_cols = np.empty(0, dtype=np.int64)

    per_input: list[dict] = []
    for label, path in args.inputs:
        info = evaluate_input(label, path,
                              raw_rows, raw_cols, raw_vals,
                              raw_regs, raw_bars,
                              zero_rows, zero_cols, thresholds)
        per_input.append(info)

    # ---- TSV: per-threshold table -------------------------------------------
    tsv_path = args.out_dir / f'{args.out_name}.tsv'
    log.info('Writing %s', tsv_path)
    with open(tsv_path, 'w') as fh:
        cols = ['input', 'kind', 'threshold',
                'n_raw_positive', 'n_lost', 'n_recovered',
                'lost_fraction', 'recovered_fraction']
        fh.write('\t'.join(cols) + '\n')
        for d in per_input:
            for r in d['per_threshold']:
                fh.write('\t'.join(str(r[k]) if k in ('input', 'kind') else
                                   f'{r[k]:.6g}' if isinstance(r[k], float) else
                                   str(r[k])
                                   for k in cols) + '\n')

    # ---- Per-cell lost counts TSV -------------------------------------------
    pc_path = args.out_dir / f'{args.out_name}_per_cell.tsv'
    log.info('Writing %s', pc_path)
    with open(pc_path, 'w') as fh:
        fh.write('input\tthreshold\tbarcode\tn_lost_at_t\n')
        for d in per_input:
            for c, bc in enumerate(raw_bars):
                if d['per_cell_lost'][c] == 0:
                    continue
                fh.write(f'{d["label"]}\t{d["default_threshold"]:.6g}\t{bc}\t'
                         f'{int(d["per_cell_lost"][c])}\n')

    # ---- JSON summary -------------------------------------------------------
    summary: dict = {
        'mm_dir':           str(args.mm_dir),
        'n_raw_positive':   int(raw_rows.size),
        'thresholds':       thresholds,
        'n_zero_sample':    int(zero_rows.size),
        'inputs': [
            {
                'label':           d['label'],
                'path':            d['path'],
                'kind':            d['kind'],
                'n_raw_positive':  d['n_raw_positive'],
                'n_modeled':       d['n_modeled'],
                'n_unmodeled':     d['n_unmodeled'],
                'imp_at_raw_pos_stats': {
                    'mean':    float(d['imp_vals'].mean()),
                    'median':  float(np.median(d['imp_vals'])),
                    'p10':     float(np.quantile(d['imp_vals'], 0.10)),
                    'p90':     float(np.quantile(d['imp_vals'], 0.90)),
                    'max':     float(d['imp_vals'].max()),
                    'frac_eq_zero': float((d['imp_vals'] == 0).mean()),
                },
                'delta_stats': {
                    'mean':    float(d['delta'].mean()),
                    'median':  float(np.median(d['delta'])),
                    'p10':     float(np.quantile(d['delta'], 0.10)),
                    'p90':     float(np.quantile(d['delta'], 0.90)),
                    'frac_negative': float((d['delta'] < 0).mean()),
                },
                'imp_at_raw_zero_stats': (
                    {
                        'n':       int(d['zero_imp_vals'].size),
                        'mean':    float(d['zero_imp_vals'].mean()),
                        'median':  float(np.median(d['zero_imp_vals'])),
                        'p90':     float(np.quantile(d['zero_imp_vals'], 0.90)),
                        'max':     float(d['zero_imp_vals'].max()),
                        'frac_eq_zero': float((d['zero_imp_vals'] == 0).mean()),
                    } if d['zero_imp_vals'].size else None
                ),
                'per_threshold': d['per_threshold'],
                'per_cell_lost_stats': {
                    'n_cells_with_loss': int((d['per_cell_lost'] > 0).sum()),
                    'mean':   float(d['per_cell_lost'].mean()),
                    'median': float(np.median(d['per_cell_lost'])),
                    'max':    int(d['per_cell_lost'].max()),
                },
                'per_bin_lost_stats': {
                    'n_bins_with_loss': int((d['per_bin_lost'] > 0).sum()),
                    'mean':   float(d['per_bin_lost'].mean()),
                    'median': float(np.median(d['per_bin_lost'])),
                    'max':    int(d['per_bin_lost'].max()),
                },
            }
            for d in per_input
        ],
    }
    json_path = args.out_dir / f'{args.out_name}.json'
    json_path.write_text(json.dumps(summary, indent=2) + '\n')
    log.info('Wrote %s', json_path)
    for r in summary['inputs']:
        log.info('  %-12s  raw_pos=%d  modeled=%d  '
                 'imp@raw1 median=%.3g  delta median=%+.3g  frac_imp=0=%.3f',
                 r['label'], r['n_raw_positive'], r['n_modeled'],
                 r['imp_at_raw_pos_stats']['median'],
                 r['delta_stats']['median'],
                 r['imp_at_raw_pos_stats']['frac_eq_zero'])

    # ---- Plots --------------------------------------------------------------
    plot_imp_at_raw_pos(per_input,
                         args.out_dir / f'{args.out_name}_imp_at_raw_pos.png')
    plot_delta(per_input,
               args.out_dir / f'{args.out_name}_delta.png')
    plot_per_cell_lost(per_input,
                       args.out_dir / f'{args.out_name}_per_cell_lost.png')
    if zero_rows.size:
        plot_imp_at_raw_zero(per_input,
                             args.out_dir / f'{args.out_name}_imp_at_raw_zero.png')
    log.info('Wrote figures.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
