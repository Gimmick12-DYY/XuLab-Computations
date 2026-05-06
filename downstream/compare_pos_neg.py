#!/usr/bin/env python
# -----------------------------------------------------------------------------
# compare_pos_neg.py
#
# Companion to prepare_pos_neg_bins.R. For each input matrix (raw mm or
# imputed HDF5), compute per-bin total signal at the positive and negative
# bins, and report:
#   * AUROC of "is positive bin" classification using per-bin total signal
#   * Kolmogorov-Smirnov statistic + p-value (pos vs neg distribution)
#   * Median signal in pos vs neg + log2(median_pos / median_neg)
# Writes a multi-panel ECDF figure (one panel per input), a long-form TSV of
# per-bin signals, and a JSON summary.
#
# Usage:
#   python compare_pos_neg.py \
#     --bins-tsv  /work/.../downstream/bins/pos_neg_bins.tsv \
#     --out-dir   /work/.../downstream/pos_neg \
#     --input raw=/work/.../mm \
#     --input cisTopic=/work/.../cistopic_ctcf/impute/imputed_Prc_hdf5_all.h5 \
#     --input FITS=/work/.../FITS/work/ctcf/impute/imputed_Prc_hdf5_all.h5
#
# Each --input is LABEL=PATH. PATH is either:
#   * a directory with matrix.mtx.gz (interpreted as raw observed counts), or
#   * a .h5 / .hdf5 file with a Prc dataset (interpreted as imputed values).
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
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
import scipy.sparse as sp  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

log = logging.getLogger('compare_pos_neg')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


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


def load_bins_tsv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (pos_idx_orig, neg_idx_orig) as int64 arrays in mm row space."""
    pos: list[int] = []
    neg: list[int] = []
    with open(path) as fh:
        header = fh.readline().rstrip('\n').split('\t')
        try:
            i_orig = header.index('bin_idx_orig')
            i_lab  = header.index('label')
        except ValueError as e:
            raise SystemExit(f'Bad header in {path}: {e}')
        for ln in fh:
            row = ln.rstrip('\n').split('\t')
            if not row[0]:
                continue
            (pos if row[i_lab] == 'pos' else neg).append(int(row[i_orig]))
    return np.asarray(pos, dtype=np.int64), np.asarray(neg, dtype=np.int64)


def per_bin_total_from_mm(mm_dir: Path, bin_idx: np.ndarray) -> np.ndarray:
    """Sum across cells for the given bin indices, reading matrix.mtx.gz."""
    mtx_path = mm_dir / 'matrix.mtx.gz'
    if not mtx_path.exists():
        raise SystemExit(f'Missing {mtx_path}')
    log.info('  loading mm matrix %s', mtx_path)
    mat = sio.mmread(str(mtx_path)).tocsr()
    return np.asarray(mat[bin_idx].sum(axis=1)).ravel().astype(np.float64)


def per_bin_total_from_hdf5(h5_path: Path, bin_idx: np.ndarray,
                             chunk: int = 50_000) -> np.ndarray:
    """Sum across cells for the given bin indices, reading Prc from HDF5.

    h5py fancy indexing with sorted unique indices is much faster than scattered
    reads, so we sort, dedup, and remap on the way out.
    """
    log.info('  loading Prc from %s', h5_path)
    sort_order = np.argsort(bin_idx, kind='stable')
    sorted_idx = bin_idx[sort_order]
    uniq, inv = np.unique(sorted_idx, return_inverse=True)
    out_uniq = np.zeros(uniq.size, dtype=np.float64)
    with h5py.File(h5_path, 'r') as h5:
        ds = h5['Prc']
        if uniq.max() >= ds.shape[0]:
            raise SystemExit(
                f'Bin index {int(uniq.max())} out of range for HDF5 with '
                f'{ds.shape[0]} rows. Did the imputed HDF5 come from the same '
                f'mm/regions.tsv.gz used to build the bins?')
        for s in range(0, uniq.size, chunk):
            sl = uniq[s:s + chunk]
            block = ds[list(sl), :]
            out_uniq[s:s + chunk] = block.sum(axis=1, dtype=np.float64)
    out_sorted = out_uniq[inv]
    out = np.empty_like(out_sorted)
    out[sort_order] = out_sorted
    return out


def per_bin_total(label: str, path: Path, bin_idx: np.ndarray) -> np.ndarray:
    if path.is_dir():
        return per_bin_total_from_mm(path, bin_idx)
    if path.suffix.lower() in ('.h5', '.hdf5'):
        return per_bin_total_from_hdf5(path, bin_idx)
    raise SystemExit(
        f'--input {label}={path}: PATH must be a directory containing '
        f'matrix.mtx.gz or an .h5/.hdf5 file.')


def safe_log10(x: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(x, np.finfo(np.float64).tiny))


def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    if y.min() == y.max():
        return float('nan')
    return float(roc_auc_score(y, s))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__ or '',
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bins-tsv', type=Path, required=True,
                    help='Output of prepare_pos_neg_bins.R')
    ap.add_argument('--out-dir',  type=Path, required=True)
    ap.add_argument('--input',    type=parse_input_arg, action='append',
                    required=True, dest='inputs',
                    metavar='LABEL=PATH',
                    help='LABEL=PATH; PATH = mm/ directory or imputed HDF5. '
                         'May be passed multiple times.')
    ap.add_argument('--out-name', type=str, default='compare_pos_neg')
    ap.add_argument('--ecdf-log', action='store_true', default=True,
                    help='Plot ECDF on log1p x-axis (default).')
    ap.add_argument('--no-ecdf-log', dest='ecdf_log', action='store_false')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info('Loading bin indices from %s', args.bins_tsv)
    pos_idx, neg_idx = load_bins_tsv(args.bins_tsv)
    log.info('Loaded %d positives, %d negatives', pos_idx.size, neg_idx.size)

    bin_idx = np.concatenate([pos_idx, neg_idx])
    y_true = np.concatenate([np.ones(pos_idx.size,  dtype=np.int8),
                             np.zeros(neg_idx.size, dtype=np.int8)])

    per_input: dict[str, np.ndarray] = {}
    summary_inputs: list[dict] = []
    for label, path in args.inputs:
        log.info('Input %r -> %s', label, path)
        signal = per_bin_total(label, path, bin_idx)
        per_input[label] = signal
        pos_sig = signal[:pos_idx.size]
        neg_sig = signal[pos_idx.size:]
        ks = ks_2samp(pos_sig, neg_sig)
        summary_inputs.append({
            'label':           label,
            'path':            str(path),
            'auroc':           safe_auroc(y_true, signal),
            'ks_statistic':    float(ks.statistic),
            'ks_pvalue':       float(ks.pvalue),
            'pos_median':      float(np.median(pos_sig)),
            'neg_median':      float(np.median(neg_sig)),
            'pos_mean':        float(pos_sig.mean()),
            'neg_mean':        float(neg_sig.mean()),
            'pos_zero_frac':   float((pos_sig == 0).mean()),
            'neg_zero_frac':   float((neg_sig == 0).mean()),
            'log2_median_ratio': float(
                np.log2(max(np.median(pos_sig), 1e-30) / max(np.median(neg_sig), 1e-30))
            ),
        })

    # -- TSV: long-form per-bin signal across all inputs --------------------
    tsv_path = args.out_dir / f'{args.out_name}.tsv'
    log.info('Writing %s', tsv_path)
    with open(tsv_path, 'w') as fh:
        cols = ['input', 'group', 'bin_idx_orig'] + ['signal']
        fh.write('\t'.join(cols) + '\n')
        for label, sig in per_input.items():
            for k in range(bin_idx.size):
                grp = 'pos' if k < pos_idx.size else 'neg'
                fh.write(f'{label}\t{grp}\t{int(bin_idx[k])}\t{sig[k]:.6g}\n')

    # -- JSON summary -------------------------------------------------------
    json_path = args.out_dir / f'{args.out_name}.json'
    summary = {
        'bins_tsv': str(args.bins_tsv),
        'n_pos':    int(pos_idx.size),
        'n_neg':    int(neg_idx.size),
        'inputs':   summary_inputs,
    }
    json_path.write_text(json.dumps(summary, indent=2) + '\n')
    log.info('Wrote %s', json_path)

    for r in summary_inputs:
        log.info('  %-12s  AUROC=%.4f  KS=%.4f (p=%.2e)  median_pos=%.4g  median_neg=%.4g  log2_ratio=%+.3f',
                 r['label'], r['auroc'], r['ks_statistic'], r['ks_pvalue'],
                 r['pos_median'], r['neg_median'], r['log2_median_ratio'])

    # -- multi-panel ECDF figure --------------------------------------------
    n = len(per_input)
    cols_n = min(n, 3)
    rows_n = (n + cols_n - 1) // cols_n
    fig, axes = plt.subplots(rows_n, cols_n,
                              figsize=(5 * cols_n, 4 * rows_n),
                              squeeze=False)
    flat = axes.flatten()
    POS_COLOR = '#E41A1C'
    NEG_COLOR = '#377EB8'
    for ax_idx, (label, sig) in enumerate(per_input.items()):
        ax = flat[ax_idx]
        pos_sig = sig[:pos_idx.size]
        neg_sig = sig[pos_idx.size:]
        if args.ecdf_log:
            x_pos = np.log1p(np.sort(pos_sig))
            x_neg = np.log1p(np.sort(neg_sig))
            xlab  = 'log1p(per-bin signal)'
        else:
            x_pos = np.sort(pos_sig); x_neg = np.sort(neg_sig)
            xlab  = 'per-bin signal'
        y_pos = np.linspace(0, 1, len(x_pos), endpoint=False)
        y_neg = np.linspace(0, 1, len(x_neg), endpoint=False)
        ax.plot(x_pos, y_pos, color=POS_COLOR, lw=1.4, label='Positive (CTCF cCRE)')
        ax.plot(x_neg, y_neg, color=NEG_COLOR, lw=1.4, label='Negative (random)')
        info = next(d for d in summary_inputs if d['label'] == label)
        ax.set_title(f'{label}\nAUROC={info["auroc"]:.3f}  '
                     f'KS={info["ks_statistic"]:.3f} '
                     f'(log2 med ratio={info["log2_median_ratio"]:+.2f})',
                     fontsize=10)
        ax.set_xlabel(xlab); ax.set_ylabel('CDF')
        ax.set_ylim(0, 1.0)
        ax.legend(frameon=False, fontsize=8)
    for k in range(len(per_input), len(flat)):
        flat[k].axis('off')
    fig.suptitle('Per-bin signal at CTCF positive vs negative bins '
                 f'(n_pos={pos_idx.size}, n_neg={neg_idx.size})')
    fig.tight_layout()
    png_path = args.out_dir / f'{args.out_name}.png'
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    log.info('Wrote %s', png_path)

    # -- AUROC bar chart across inputs (single panel) ------------------------
    fig, ax = plt.subplots(figsize=(max(4, 0.7 * len(per_input) + 2), 4))
    labels = [r['label'] for r in summary_inputs]
    aurocs = [r['auroc'] for r in summary_inputs]
    bars = ax.bar(np.arange(len(labels)), aurocs,
                  color=['#777'] * len(labels))
    for b, v in zip(bars, aurocs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.3f}',
                ha='center', va='bottom', fontsize=9)
    ax.axhline(0.5, color='k', lw=0.5, ls=':')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylim(0, 1.0); ax.set_ylabel('AUROC (positive vs negative bins)')
    ax.set_title('Bin-level CTCF separability across pipelines')
    fig.tight_layout()
    bar_path = args.out_dir / f'{args.out_name}_auroc.png'
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    log.info('Wrote %s', bar_path)

    return 0


if __name__ == '__main__':
    sys.exit(main())
