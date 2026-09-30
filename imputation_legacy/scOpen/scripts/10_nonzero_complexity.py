#!/usr/bin/env python
from __future__ import annotations

import gzip
import json
import logging
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger('10_nonzero')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def read_lines_gz(path: Path) -> list[str]:
    with gzip.open(path, 'rt') as fh:
        return [x.rstrip('\n') for x in fh]


def main() -> int:
    ap = base_parser(__doc__ or '')
    ap.add_argument('--imputed-h5', type=str, default=None)
    ap.add_argument('--mm-dir', type=str, default=None)
    ap.add_argument('--thresholds', type=str,
                    default='0,1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4',
                    help='Comma-separated thresholds in probability space.')
    ap.add_argument('--chunk-rows', type=int, default=2000)
    ap.add_argument('--out-name', type=str, default='nonzero_complexity')
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    work = Path(cfg['paths']['work_dir'])
    eval_dir = work / 'eval'
    eval_dir.mkdir(parents=True, exist_ok=True)

    mm_dir = Path(args.mm_dir) if args.mm_dir else Path(cfg['paths']['mm'])
    mtx_path = mm_dir / 'matrix.mtx.gz'
    reg_path = mm_dir / 'regions.tsv.gz'
    bar_path = mm_dir / 'barcodes.tsv.gz'

    orig = sio.mmread(mtx_path).tocsr()  # full space regions x cells
    orig_regs = read_lines_gz(reg_path)
    orig_bars = read_lines_gz(bar_path)

    h5_path = Path(args.imputed_h5) if args.imputed_h5 else Path(cfg['paths']['impute']) / 'imputed_Prc_hdf5_all.h5'
    thresholds = np.array([float(x.strip()) for x in args.thresholds.split(',') if x.strip()], dtype=np.float64)
    thresholds.sort()

    with h5py.File(h5_path, 'r') as h5:
        imp_regs = [x.decode() if isinstance(x, bytes) else str(x) for x in h5['regions'][:]]
        imp_bars = [x.decode() if isinstance(x, bytes) else str(x) for x in h5['barcodes'][:]]
        ds = h5['Prc']

        # Detect stored scaling using full row coverage (chunked).
        # Using only the first rows underestimates scale and breaks thresholding.
        col_sums = np.zeros(ds.shape[1], dtype=np.float64)
        for i0 in range(0, ds.shape[0], args.chunk_rows):
            i1 = min(ds.shape[0], i0 + args.chunk_rows)
            col_sums += ds[i0:i1, :].sum(axis=0, dtype=np.float64)
        scale_in_h5 = float(col_sums.mean())
        if scale_in_h5 <= 0:
            scale_in_h5 = 1.0

        reg_to_orig = {r: i for i, r in enumerate(orig_regs)}
        bar_to_orig = {b: i for i, b in enumerate(orig_bars)}
        row_idx = np.array([reg_to_orig[r] for r in imp_regs], dtype=np.int64)
        col_idx = np.array([bar_to_orig[b] for b in imp_bars], dtype=np.int64)

        # Original in modeled space (same rows/cells as imputed)
        orig_modeled = (orig[row_idx][:, col_idx] > 0).tocsr()
        orig_nnz_modeled = int(orig_modeled.nnz)
        orig_nnz_full = int(orig.nnz)

        Rf, Cf = ds.shape
        R0, C0 = orig.shape
        total_modeled_entries = int(Rf * Cf)
        total_full_entries = int(R0 * C0)

        pred_nnz = np.zeros_like(thresholds, dtype=np.int64)
        tp_nnz = np.zeros_like(thresholds, dtype=np.int64)

        indptr = orig_modeled.indptr
        indices = orig_modeled.indices

        for i0 in range(0, Rf, args.chunk_rows):
            i1 = min(Rf, i0 + args.chunk_rows)
            block = ds[i0:i1, :].astype(np.float64, copy=False) / scale_in_h5

            # Predicted nnz at each threshold.
            for ti, t in enumerate(thresholds):
                pred_nnz[ti] += int(np.count_nonzero(block > t))

            # True-positive nnz over original nonzero coordinates only.
            for r_local, r in enumerate(range(i0, i1)):
                a = indptr[r]
                b = indptr[r + 1]
                if a == b:
                    continue
                cols = indices[a:b]
                vals = block[r_local, cols]
                for ti, t in enumerate(thresholds):
                    tp_nnz[ti] += int(np.count_nonzero(vals > t))

            if i0 % max(args.chunk_rows * 20, 1) == 0:
                log.info('Processed rows %d/%d', i1, Rf)

    # Metrics table
    rows = []
    for ti, t in enumerate(thresholds):
        p = int(pred_nnz[ti])
        tp = int(tp_nnz[ti])
        fp = p - tp
        fn = orig_nnz_modeled - tp
        precision = (tp / p) if p > 0 else float('nan')
        recall = (tp / orig_nnz_modeled) if orig_nnz_modeled > 0 else float('nan')
        f1 = (2 * precision * recall / (precision + recall)) if (precision > 0 and recall > 0) else 0.0
        rows.append({
            'threshold': float(t),
            'pred_nnz_modeled': p,
            'pred_density_modeled': p / total_modeled_entries,
            'pred_nnz_full_space': p,
            'pred_density_full_space': p / total_full_entries,
            'tp_nnz_modeled': tp,
            'fp_nnz_modeled': fp,
            'fn_nnz_modeled': fn,
            'precision_vs_original_nonzero': precision,
            'recall_vs_original_nonzero': recall,
            'f1': f1,
            'nnz_fold_vs_original_modeled': (p / orig_nnz_modeled) if orig_nnz_modeled > 0 else float('nan'),
            'nnz_fold_vs_original_full': (p / orig_nnz_full) if orig_nnz_full > 0 else float('nan'),
        })

    # Save TSV
    tsv_path = eval_dir / f'{args.out_name}.tsv'
    keys = list(rows[0].keys())
    with open(tsv_path, 'w') as fh:
        fh.write('\t'.join(keys) + '\n')
        for r in rows:
            fh.write('\t'.join(str(r[k]) for k in keys) + '\n')

    # Save JSON summary
    best_f1 = max(rows, key=lambda x: x['f1'])
    summary = {
        'imputed_h5': str(h5_path),
        'scale_in_h5_used': scale_in_h5,
        'original_shape_full': [int(R0), int(C0)],
        'imputed_shape_modeled': [int(Rf), int(Cf)],
        'coverage_regions_fraction': float(Rf / R0),
        'original_nnz_full': orig_nnz_full,
        'original_nnz_modeled': orig_nnz_modeled,
        'rows': rows,
        'best_f1_row': best_f1,
    }
    json_path = eval_dir / f'{args.out_name}.json'
    json_path.write_text(json.dumps(summary, indent=2) + '\n')

    # Plot curves
    x = np.array([r['threshold'] for r in rows], dtype=float)
    fold = np.array([r['nnz_fold_vs_original_modeled'] for r in rows], dtype=float)
    prec = np.array([r['precision_vs_original_nonzero'] for r in rows], dtype=float)
    rec = np.array([r['recall_vs_original_nonzero'] for r in rows], dtype=float)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(x, fold, marker='o')
    ax[0].set_xscale('log')
    ax[0].set_xlabel('threshold (probability space)')
    ax[0].set_ylabel('pred_nnz / original_nnz (modeled space)')
    ax[0].set_title('Nonzero inflation vs threshold')

    ax[1].plot(x, prec, marker='o', label='precision')
    ax[1].plot(x, rec, marker='o', label='recall')
    ax[1].set_xscale('log')
    ax[1].set_xlabel('threshold (probability space)')
    ax[1].set_ylabel('value')
    ax[1].set_ylim(0, 1.02)
    ax[1].set_title('Precision/Recall vs original nonzeros')
    ax[1].legend(frameon=False)

    fig.tight_layout()
    png_path = eval_dir / f'{args.out_name}.png'
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    log.info('Wrote %s', tsv_path)
    log.info('Wrote %s', json_path)
    log.info('Wrote %s', png_path)
    log.info('Best F1 threshold=%.3g, fold=%.3f, precision=%.3f, recall=%.3f',
             best_f1['threshold'], best_f1['nnz_fold_vs_original_modeled'],
             best_f1['precision_vs_original_nonzero'], best_f1['recall_vs_original_nonzero'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
