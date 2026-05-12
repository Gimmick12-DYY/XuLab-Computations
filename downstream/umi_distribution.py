#!/usr/bin/env python
# -----------------------------------------------------------------------------
# umi_distribution.py
#
# Per-cell UMI / signal-sum distribution for the raw matrix and every imputed
# matrix. For each pipeline we sum the matrix over rows to get a per-cell
# total, then (optionally) align every pipeline's per-cell totals to a single
# reference barcode order so the distributions are reported over the same
# cells where possible.
#
# Each --input PATH is one of:
#   * directory with matrix.mtx.gz                  (raw counts; <work>/mm)
#   * directory with factors.npz + barcodes.tsv     (cisTopic / scOpen)
#   * directory with matrix.npy + barcodes.tsv      (FITS / MAGIC)
#   * directory with matrix_csr.npz + barcodes.tsv  (PUscOpen csr_full_mm)
#   * legacy HDF5 file with /Prc + /barcodes        (older runs)
#
# Outputs (under --out-dir):
#   <out>.tsv                long-form: input, kind, barcode, umi_total
#   <out>_wide.tsv           wide-form: barcode <tab> input1 <tab> input2 ...
#                            (only when alignment is enabled; default)
#   <out>_stats.json         per-input mean / median / quantiles
#   <out>_hist.png           overlaid log-x histogram of per-cell totals
#   <out>_ecdf.png           overlaid ECDF
#   <out>_box.png            boxplot across pipelines
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

log = logging.getLogger('umi_dist')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


# -----------------------------------------------------------------------------
# Argument parsing
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


# -----------------------------------------------------------------------------
# Per-input loaders. Each returns (per_cell_sum, barcodes, kind).
# per_cell_sum is a 1D float64 array aligned to the input's own barcode order.
# -----------------------------------------------------------------------------
def _read_lines(path: Path) -> list[str]:
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as fh:
            return [ln.rstrip('\n') for ln in fh if ln.strip()]
    return [ln for ln in path.read_text().splitlines() if ln.strip()]


def _maybe_barcodes_in(dir_path: Path) -> list[str] | None:
    for name in ('barcodes.tsv', 'barcodes.tsv.gz'):
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
        if (path / 'matrix.mtx.gz').exists():
            return 'mm'
        if (path / 'matrix_csr.npz').exists():
            return 'csr'
        if (path / 'factors.npz').exists():
            return 'factored'
        if (path / 'matrix.npy').exists():
            return 'dense'
        if (list(path.glob('*.h5')) + list(path.glob('*.hdf5'))):
            return 'hdf5'
    raise SystemExit(f'Cannot determine input kind from {path}')


def percell_from_mm(mm_dir: Path) -> tuple[np.ndarray, list[str]]:
    log.info('  loading mm matrix %s', mm_dir / 'matrix.mtx.gz')
    mat = sio.mmread(str(mm_dir / 'matrix.mtx.gz')).tocsc()
    bars = _read_lines(mm_dir / 'barcodes.tsv.gz')
    if mat.shape[1] != len(bars):
        raise SystemExit(f'mm shape {mat.shape} vs {len(bars)} barcodes')
    return np.asarray(mat.sum(axis=0)).ravel().astype(np.float64), bars


def percell_from_factored(impute_dir: Path) -> tuple[np.ndarray, list[str]]:
    log.info('  loading factors.npz %s', impute_dir)
    bars = _maybe_barcodes_in(impute_dir)
    if bars is None:
        raise SystemExit(f'Missing barcodes.tsv next to {impute_dir}/factors.npz')
    with np.load(impute_dir / 'factors.npz') as z:
        W = np.asarray(z['W'], dtype=np.float64)
        H = np.asarray(z['H'], dtype=np.float64)
        if 'per_cell_factor' in z.files:
            pcf_raw = z['per_cell_factor']
            # Scalar (0-d) -> scalar broadcast; vector (1-d) -> element-wise.
            f: float | np.ndarray
            f = float(pcf_raw) if pcf_raw.ndim == 0 else np.asarray(pcf_raw, dtype=np.float64).ravel()
        else:
            f = 1.0
    sums = (W.sum(axis=0) @ H) * f
    return np.asarray(sums, dtype=np.float64), bars


def percell_from_dense(impute_dir: Path, chunk_rows: int = 20_000
                        ) -> tuple[np.ndarray, list[str]]:
    log.info('  loading matrix.npy %s', impute_dir)
    bars = _maybe_barcodes_in(impute_dir)
    if bars is None:
        raise SystemExit(f'Missing barcodes.tsv next to {impute_dir}/matrix.npy')
    arr = np.load(impute_dir / 'matrix.npy', mmap_mode='r')
    n_rows, n_cols = arr.shape
    if n_cols != len(bars):
        raise SystemExit(f'matrix.npy cols={n_cols} vs barcodes={len(bars)}')
    out = np.zeros(n_cols, dtype=np.float64)
    for s in range(0, n_rows, chunk_rows):
        e = min(s + chunk_rows, n_rows)
        out += np.asarray(arr[s:e], dtype=np.float64).sum(axis=0)
    return out, bars


def percell_from_csr(impute_dir: Path) -> tuple[np.ndarray, list[str]]:
    log.info('  loading matrix_csr.npz %s', impute_dir)
    bars = _maybe_barcodes_in(impute_dir)
    if bars is None:
        raise SystemExit(f'Missing barcodes.tsv next to {impute_dir}/matrix_csr.npz')
    mat = sparse.load_npz(impute_dir / 'matrix_csr.npz')
    if mat.shape[1] != len(bars):
        raise SystemExit(f'csr cols={mat.shape[1]} vs barcodes={len(bars)}')
    return np.asarray(mat.sum(axis=0)).ravel().astype(np.float64), bars


def percell_from_hdf5(h5_path: Path, chunk_rows: int = 20_000
                       ) -> tuple[np.ndarray, list[str]]:
    if h5_path.is_dir():
        cand = sorted(list(h5_path.glob('*.h5')) + list(h5_path.glob('*.hdf5')))
        if not cand:
            raise SystemExit(f'No .h5 in {h5_path}')
        h5_path = cand[0]
    log.info('  loading HDF5 %s', h5_path)
    with h5py.File(h5_path, 'r') as h5:
        if 'Prc' not in h5:
            raise SystemExit(f'Expected /Prc dataset in {h5_path}')
        ds = h5['Prc']
        n_rows, n_cols = ds.shape
        if 'barcodes' in h5:
            bars = [s.decode() if isinstance(s, (bytes, bytearray)) else str(s)
                    for s in h5['barcodes'][:]]
        else:
            sib_gz = h5_path.parent / 'barcodes.tsv.gz'
            sib = h5_path.parent / 'barcodes.tsv'
            if sib.exists():    bars = _read_lines(sib)
            elif sib_gz.exists(): bars = _read_lines(sib_gz)
            else:
                raise SystemExit(f'No barcodes alongside {h5_path}')
        if len(bars) != n_cols:
            raise SystemExit(f'HDF5 cols={n_cols} vs barcodes={len(bars)}')
        out = np.zeros(n_cols, dtype=np.float64)
        for s in range(0, n_rows, chunk_rows):
            e = min(s + chunk_rows, n_rows)
            out += ds[s:e, :].sum(axis=0, dtype=np.float64)
        return out, bars


def percell_for_input(label: str, path: Path) -> tuple[np.ndarray, list[str], str]:
    kind = detect_input_kind(path)
    if   kind == 'mm':       sums, bars = percell_from_mm(path)
    elif kind == 'csr':      sums, bars = percell_from_csr(path)
    elif kind == 'factored': sums, bars = percell_from_factored(path)
    elif kind == 'dense':    sums, bars = percell_from_dense(path)
    elif kind == 'hdf5':     sums, bars = percell_from_hdf5(path)
    else:
        raise SystemExit(f'Unhandled kind {kind} for {label}={path}')
    return sums, bars, kind


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def _color_cycle(n: int) -> list[str]:
    cmap = plt.get_cmap('tab10' if n <= 10 else 'tab20')
    return [cmap(i % cmap.N) for i in range(n)]


def plot_hist(per_input: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = _color_cycle(len(per_input))
    nonempty = [d for d in per_input if d['sums_aligned'].size]
    if not nonempty:
        log.warning('No data for histogram'); return
    all_max = max(float(np.max(d['sums_aligned'])) for d in nonempty)
    bins = np.logspace(0, np.log10(max(all_max, 10.0)) + 0.05, 80)
    for d, c in zip(per_input, colors):
        s = d['sums_aligned']
        if s.size == 0:
            continue
        ax.hist(np.clip(s, 1e-1, None), bins=bins, alpha=0.45,
                label=d['label'], color=c)
    ax.set_xscale('log')
    ax.set_xlabel('per-cell total signal (log scale)')
    ax.set_ylabel('# cells')
    ax.set_title('Per-cell UMI / signal distribution across pipelines')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_ecdf(per_input: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = _color_cycle(len(per_input))
    for d, c in zip(per_input, colors):
        s = d['sums_aligned']
        if s.size == 0:
            continue
        sorted_s = np.sort(np.clip(s, 1e-3, None))
        y = np.linspace(0, 1, sorted_s.size, endpoint=False)
        ax.plot(sorted_s, y, color=c, lw=1.4, label=d['label'])
    ax.set_xscale('log')
    ax.set_xlabel('per-cell total signal (log scale)')
    ax.set_ylabel('ECDF (fraction of cells)')
    ax.set_ylim(0, 1)
    ax.set_title('Per-cell UMI / signal ECDF across pipelines')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_box(per_input: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(per_input) + 2), 5))
    labels = [d['label'] for d in per_input]
    arrays = [np.clip(d['sums_aligned'], 1e-3, None) for d in per_input]
    ax.boxplot(arrays, labels=labels, showfliers=False, widths=0.6)
    ax.set_yscale('log')
    ax.set_ylabel('per-cell total signal (log scale)')
    ax.set_title('Per-cell UMI / signal distribution (boxplot, outliers hidden)')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def _stats(v: np.ndarray) -> dict:
    if v.size == 0:
        return {k: float('nan') for k in
                ('mean', 'median', 'min', 'max', 'p10', 'p90', 'p99', 'std')}
    return {
        'mean':   float(v.mean()),
        'median': float(np.median(v)),
        'std':    float(v.std()),
        'min':    float(v.min()),
        'max':    float(v.max()),
        'p10':    float(np.quantile(v, 0.10)),
        'p90':    float(np.quantile(v, 0.90)),
        'p99':    float(np.quantile(v, 0.99)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__ or '',
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--input',   type=parse_input_arg, action='append',
                    required=True, dest='inputs', metavar='LABEL=PATH',
                    help='LABEL=PATH; repeatable.')
    ap.add_argument('--out-name', type=str, default='umi_distribution')
    ap.add_argument('--align-to', type=str, default=None,
                    help='Restrict every input to the cell barcodes present in '
                         'the named input (default: first --input). Use "none" '
                         'to skip alignment.')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load every input ----------------------------------------------------
    raw_results: list[dict] = []
    for label, path in args.inputs:
        log.info('Input %r -> %s', label, path)
        sums, bars, kind = percell_for_input(label, path)
        raw_results.append({
            'label': label, 'path': str(path), 'kind': kind,
            'sums_native': sums,                       # native barcode order
            'barcodes_native': bars,
            'bar_to_idx': {b: i for i, b in enumerate(bars)},
        })
        log.info('  kind=%s  n_cells=%d  median=%.4g  mean=%.4g  max=%.4g',
                 kind, len(bars),
                 float(np.median(sums)) if sums.size else float('nan'),
                 float(sums.mean()) if sums.size else float('nan'),
                 float(sums.max()) if sums.size else float('nan'))

    # ---- Pick alignment anchor ----------------------------------------------
    align_label = (args.align_to or raw_results[0]['label']).strip()
    do_align = align_label.lower() != 'none'
    if do_align:
        anchor = next((d for d in raw_results if d['label'] == align_label), None)
        if anchor is None:
            log.error('--align-to %r not in --input labels.', align_label)
            return 1
        anchor_bars = list(anchor['barcodes_native'])
        log.info('Aligning every input to %s barcode order (%d cells).',
                 align_label, len(anchor_bars))
    else:
        anchor_bars = None
        log.info('Alignment disabled; reporting each input on its native cell list.')

    # ---- Build aligned views -------------------------------------------------
    per_input: list[dict] = []
    for d in raw_results:
        if do_align:
            aligned = np.full(len(anchor_bars), np.nan, dtype=np.float64)
            missing = 0
            for j, bc in enumerate(anchor_bars):
                idx = d['bar_to_idx'].get(bc)
                if idx is None:
                    missing += 1
                else:
                    aligned[j] = d['sums_native'][idx]
            kept = aligned[~np.isnan(aligned)]
            per_input.append({
                **d,
                'sums_aligned':       kept,
                'sums_aligned_full':  aligned,    # length == len(anchor_bars), with NaN
                'n_missing_in_anchor': int(missing),
            })
            if missing:
                log.warning('  %s: %d / %d anchor barcodes absent from this input.',
                            d['label'], missing, len(anchor_bars))
        else:
            per_input.append({
                **d,
                'sums_aligned':       d['sums_native'],
                'sums_aligned_full':  d['sums_native'],
                'n_missing_in_anchor': 0,
            })

    # ---- Long-form TSV (one row per (input, barcode, umi_total)) ------------
    tsv_path = args.out_dir / f'{args.out_name}.tsv'
    log.info('Writing %s', tsv_path)
    with open(tsv_path, 'w') as fh:
        fh.write('input\tkind\tbarcode\tumi_total\n')
        for d in per_input:
            for bc, val in zip(d['barcodes_native'], d['sums_native']):
                fh.write(f'{d["label"]}\t{d["kind"]}\t{bc}\t{float(val):.6g}\n')

    # ---- Wide-form TSV when alignment is enabled ----------------------------
    if do_align:
        wide_path = args.out_dir / f'{args.out_name}_wide.tsv'
        log.info('Writing %s', wide_path)
        labels = [d['label'] for d in per_input]
        with open(wide_path, 'w') as fh:
            fh.write('barcode\t' + '\t'.join(labels) + '\n')
            for j, bc in enumerate(anchor_bars):
                cells = [bc]
                for d in per_input:
                    val = d['sums_aligned_full'][j]
                    cells.append('NA' if np.isnan(val) else f'{float(val):.6g}')
                fh.write('\t'.join(cells) + '\n')

    # ---- JSON summary --------------------------------------------------------
    summary = {
        'aligned_to': align_label if do_align else None,
        'inputs': [
            {
                'label':              d['label'],
                'kind':               d['kind'],
                'path':               d['path'],
                'n_cells_native':     int(len(d['barcodes_native'])),
                'n_cells_evaluated':  int(d['sums_aligned'].size),
                'n_missing_in_anchor': int(d['n_missing_in_anchor']),
                'stats':              _stats(d['sums_aligned']),
            }
            for d in per_input
        ],
    }
    json_path = args.out_dir / f'{args.out_name}_stats.json'
    json_path.write_text(json.dumps(summary, indent=2) + '\n')
    log.info('Wrote %s', json_path)
    for r in summary['inputs']:
        s = r['stats']
        log.info('  %-12s  n=%d  median=%.4g  mean=%.4g  p10=%.4g  p90=%.4g',
                 r['label'], r['n_cells_evaluated'],
                 s['median'], s['mean'], s['p10'], s['p90'])

    # ---- Plots ---------------------------------------------------------------
    plot_hist(per_input, args.out_dir / f'{args.out_name}_hist.png')
    plot_ecdf(per_input, args.out_dir / f'{args.out_name}_ecdf.png')
    plot_box (per_input, args.out_dir / f'{args.out_name}_box.png')
    log.info('Wrote hist / ecdf / boxplot figures.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
