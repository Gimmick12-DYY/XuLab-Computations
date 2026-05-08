#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 05_impute.py
#
# Persist the MAGIC imputed matrix in the compact, HDF5-free format that the
# universal downstream/compare_pos_neg.py consumes (matches FITS schema):
#
#   <work>/impute/
#     matrix.npy        float32 (n_modeled_regions, n_cells) dense matrix
#                       (transposed from MAGIC's (cells, features) output)
#     regions.tsv       one chr:start-end per row, in matrix row order
#     barcodes.tsv      one cell barcode per col, in matrix col order
#     meta.json         scale_factor, source paths, n_modeled, n_cells, ...
#
# We do NOT pad to the full pre-filter region universe; downstream tools look
# bins up by name in regions.tsv.
# -----------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger('05_impute')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def _read_lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def main() -> int:
    ap = base_parser(__doc__ or '')
    ap.add_argument('--scale-factor', type=float, default=None,
                    help='Per-cell target sum after rescale.')
    ap.add_argument('--no-rescale', action='store_true',
                    help='Skip the per-cell rescale step; store MAGIC values verbatim.')
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    imp_cfg = cfg.setdefault('impute', {})
    if args.scale_factor is not None:
        imp_cfg['scale_factor'] = args.scale_factor

    in_dir = Path(cfg['paths']['magic_input'])
    run_dir = Path(cfg['paths']['magic_run'])
    impute_dir = Path(cfg['paths']['impute'])
    impute_dir.mkdir(parents=True, exist_ok=True)

    raw_path = run_dir / 'imputed_cells_x_regions.npy'
    regions_in_path = in_dir / 'regions.tsv'
    barcodes_in_path = in_dir / 'barcodes.tsv'
    for p in (raw_path, regions_in_path, barcodes_in_path):
        if not p.exists():
            log.error('Missing %s. Run earlier steps first.', p)
            return 1

    log.info('Loading raw MAGIC output (cells x regions): %s', raw_path)
    cells_x_regions = np.load(raw_path, mmap_mode='r')
    log.info('  raw shape (cells x regions) = %s', cells_x_regions.shape)

    regions = _read_lines(regions_in_path)
    barcodes = _read_lines(barcodes_in_path)
    if cells_x_regions.shape != (len(barcodes), len(regions)):
        log.error('Shape mismatch: MAGIC %s vs (cells=%d, regions=%d).',
                  cells_x_regions.shape, len(barcodes), len(regions))
        return 1

    # Transpose to (regions x cells) for the standardised schema. Force a copy
    # so the .npy on disk is C-contiguous and downstream tools can mmap rows.
    log.info('Transposing to regions x cells ...')
    matrix = np.ascontiguousarray(np.asarray(cells_x_regions, dtype=np.float32).T)
    log.info('  imputed (regions x cells) = %s, mean=%.4g, max=%.4g',
             matrix.shape, float(matrix.mean()), float(matrix.max()))

    desired_scale = float(imp_cfg.get('scale_factor', 1_000_000))
    if args.no_rescale:
        log.info('Storing MAGIC values verbatim (--no-rescale).')
        rescaled_flag = False
    else:
        sum_per_cell = np.asarray(matrix.sum(axis=0)).ravel()
        sum_per_cell_mean = float(sum_per_cell.mean()) if sum_per_cell.size else 0.0
        if sum_per_cell_mean <= 0:
            log.warning('MAGIC per-cell sum mean is non-positive (%.4g); storing verbatim.',
                        sum_per_cell_mean)
            rescaled_flag = False
        else:
            factor = desired_scale / sum_per_cell_mean
            log.info('Rescaling by %.4g so per-cell sum mean -> %.0f.', factor, desired_scale)
            matrix = (matrix * factor).astype(np.float32, copy=False)
            rescaled_flag = True

    matrix_path  = impute_dir / 'matrix.npy'
    regions_out  = impute_dir / 'regions.tsv'
    barcodes_out = impute_dir / 'barcodes.tsv'
    meta_out     = impute_dir / 'meta.json'

    log.info('Writing %s (shape %s, dtype=float32)', matrix_path, matrix.shape)
    np.save(matrix_path, matrix)
    regions_out.write_text('\n'.join(regions) + '\n')
    barcodes_out.write_text('\n'.join(barcodes) + '\n')

    summary = {
        'pipeline':       'MAGIC',
        'format':         'dense_npy',
        'matrix_path':    str(matrix_path),
        'regions_path':   str(regions_out),
        'barcodes_path':  str(barcodes_out),
        'source_raw':     str(raw_path),
        'shape':          [int(matrix.shape[0]), int(matrix.shape[1])],
        'n_modeled':      int(matrix.shape[0]),
        'n_cells':        int(matrix.shape[1]),
        'scale_factor':   desired_scale,
        'rescaled':       rescaled_flag,
    }
    meta_out.write_text(json.dumps(summary, indent=2) + '\n')
    log.info('Done: %s', summary)
    return 0


if __name__ == '__main__':
    sys.exit(main())
