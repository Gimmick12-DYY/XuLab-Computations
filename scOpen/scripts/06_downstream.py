#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 06_downstream.py
#
# Optional post-imputation summaries. FITS produces no topic decomposition, so
# unlike cisTopic's step 06 there is no UMAP / topic-binarisation; this script
# instead writes a small JSON summary of the HDF5 (per-cell sum, density at a
# few thresholds, per-region mean) so users have a quick snapshot before the
# heavier eval steps.
#
# Extend this script as more downstream analyses are added.
# -----------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger('06_down')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def main() -> int:
    ap = base_parser(__doc__ or '')
    ap.add_argument('--imputed-h5', type=str, default=None)
    ap.add_argument('--chunk-rows', type=int, default=10000)
    ap.add_argument('--thresholds', type=str,
                    default='1e-7,1e-6,1e-5,1e-4',
                    help='Comma-separated thresholds in probability space.')
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    out_dir = Path(cfg['paths']['downstream'])

    h5_path = (Path(args.imputed_h5) if args.imputed_h5
               else Path(cfg['paths']['impute']) / 'imputed_Prc_hdf5_all.h5')
    if not h5_path.exists():
        log.error('Missing %s. Run 05_impute.py first.', h5_path)
        return 1

    thresholds = [float(t.strip()) for t in args.thresholds.split(',') if t.strip()]

    with h5py.File(h5_path, 'r') as h5:
        ds = h5['Prc']
        Rf, Cf = ds.shape
        attrs = {k: (v.item() if hasattr(v, 'item') else v) for k, v in h5.attrs.items()}

        log.info('Computing per-cell sum and per-region mean over %d x %d ...', Rf, Cf)
        col_sums = np.zeros(Cf, dtype=np.float64)
        row_means = np.zeros(Rf, dtype=np.float64)
        for i0 in range(0, Rf, args.chunk_rows):
            i1 = min(Rf, i0 + args.chunk_rows)
            block = ds[i0:i1, :]
            col_sums += block.sum(axis=0, dtype=np.float64)
            row_means[i0:i1] = block.mean(axis=1, dtype=np.float64)

        scale_in_h5 = float(col_sums.mean()) or 1.0
        thresh_h5 = [(t, t * scale_in_h5) for t in thresholds]

        log.info('Detected per-cell sum mean (HDF5 scale): %.4g', scale_in_h5)

        density_at_threshold = {}
        for t, t_h5 in thresh_h5:
            count = 0
            for i0 in range(0, Rf, args.chunk_rows):
                i1 = min(Rf, i0 + args.chunk_rows)
                count += int(np.count_nonzero(ds[i0:i1, :] > t_h5))
            density_at_threshold[f'{t:g}'] = count / float(Rf * Cf)

    def _qstats(v: np.ndarray) -> dict:
        v = np.asarray(v, dtype=np.float64).ravel()
        return {
            'mean':   float(v.mean()),
            'median': float(np.median(v)),
            'p10':    float(np.percentile(v, 10)),
            'p90':    float(np.percentile(v, 90)),
            'min':    float(v.min()),
            'max':    float(v.max()),
        }

    summary = {
        'imputed_h5':           str(h5_path),
        'shape':                [int(Rf), int(Cf)],
        'h5_attrs':             attrs,
        'scale_in_h5':          scale_in_h5,
        'col_sum_stats':        _qstats(col_sums),
        'row_mean_stats':       _qstats(row_means),
        'density_at_threshold': density_at_threshold,
    }

    out_json = out_dir / 'imputed_summary.json'
    out_json.write_text(json.dumps(summary, indent=2) + '\n')
    log.info('Wrote %s', out_json)
    return 0


if __name__ == '__main__':
    sys.exit(main())
