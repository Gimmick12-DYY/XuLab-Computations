#!/usr/bin/env python
from __future__ import annotations

import gzip
import json
import logging
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger('02_prepare_fits')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def _read_lines_gz(path: Path) -> list[str]:
    with gzip.open(path, 'rt') as fh:
        return [ln.rstrip('\n') for ln in fh]


def main() -> int:
    ap = base_parser(__doc__ or '')
    ap.add_argument('--n-regions', type=int, default=None)
    ap.add_argument('--region-selection', type=str, default=None, choices=['top_cells', 'top_counts', 'random'])
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    prep = cfg.setdefault('prepare', {})
    n_regions = int(args.n_regions or prep.get('n_regions', 2000))
    mode = args.region_selection or prep.get('region_selection', 'top_cells')
    seed = int(args.seed or prep.get('seed', 42))
    rng = np.random.default_rng(seed)

    mm_dir = Path(cfg['paths']['mm'])
    mtx_path = mm_dir / 'matrix.mtx.gz'
    reg_path = mm_dir / 'regions.tsv.gz'
    bar_path = mm_dir / 'barcodes.tsv.gz'

    if not mtx_path.exists() or not reg_path.exists() or not bar_path.exists():
        log.error('Missing mm inputs at %s. Run 01_export_rds_to_mm.R first.', mm_dir)
        return 1

    mat = sio.mmread(mtx_path).tocsr()  # regions x cells
    regions = np.asarray(_read_lines_gz(reg_path))
    barcodes = np.asarray(_read_lines_gz(bar_path))

    if mode == 'top_cells':
        score = np.diff(mat.indptr)
        pick = np.argsort(score)[::-1][:min(n_regions, mat.shape[0])]
    elif mode == 'top_counts':
        score = np.asarray(mat.sum(axis=1)).ravel()
        pick = np.argsort(score)[::-1][:min(n_regions, mat.shape[0])]
    else:
        pick = rng.choice(mat.shape[0], size=min(n_regions, mat.shape[0]), replace=False)

    sub = mat[pick]  # selected regions x cells
    # FITS expects rows=sites, cols=samples/cells as dense CSV without headers.
    dense = np.asarray(sub.toarray(), dtype=np.float32)

    out_dir = Path(cfg['paths']['fits_input'])
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'fits_input.csv'
    np.savetxt(csv_path, dense, delimiter=',', fmt='%.8g')

    # Save mapping files for evaluation
    (out_dir / 'regions.selected.tsv').write_text('\n'.join(regions[pick].tolist()) + '\n')
    (out_dir / 'barcodes.tsv').write_text('\n'.join(barcodes.tolist()) + '\n')

    meta = {
        'input_shape': [int(mat.shape[0]), int(mat.shape[1])],
        'selected_shape': [int(dense.shape[0]), int(dense.shape[1])],
        'region_selection': mode,
        'n_regions': int(dense.shape[0]),
        'seed': seed,
        'input_nnz': int(mat.nnz),
        'selected_nnz': int(np.count_nonzero(dense)),
        'csv_path': str(csv_path),
    }
    (out_dir / 'prepare_meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    log.info('Wrote FITS input CSV: %s', csv_path)
    log.info('Selected matrix shape %s nnz=%d', dense.shape, int(np.count_nonzero(dense)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
