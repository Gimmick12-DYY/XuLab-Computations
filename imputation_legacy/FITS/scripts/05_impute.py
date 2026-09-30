#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 05_impute.py
#
# Persist the FITS Phase2 imputed matrix to a compact, HDF5-free format that
# the universal downstream/compare_pos_neg.py can consume:
#
#   <work>/impute/
#     matrix.npy        float32 (n_modeled, n_cells) dense reconstruction
#     regions.tsv       one chr:start-end per row, in the same row order as matrix.npy
#     barcodes.tsv      one cell barcode per col
#     meta.json         scale_factor, source paths, n_modeled, n_cells, ...
#
# We do NOT pad to the full pre-filter region universe; downstream tools must
# look up bins by name in regions.tsv.
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


def _load_imputed_matrix(path: Path) -> np.ndarray:
    """Load FITS Phase2 output. Accepts CSV / TSV / TXT, comma- or whitespace-delimited."""
    suffix = path.suffix.lower()
    if suffix == '.csv':
        return np.loadtxt(path, delimiter=',', dtype=np.float32)
    if suffix == '.tsv':
        return np.loadtxt(path, delimiter='\t', dtype=np.float32)
    try:
        return np.loadtxt(path, delimiter=',', dtype=np.float32)
    except ValueError:
        return np.loadtxt(path, dtype=np.float32)


def main() -> int:
    ap = base_parser(__doc__ or '')
    ap.add_argument('--imputed-file', type=str, default=None,
                    help='FITS Phase2 output CSV (default: read from <fits_run>/fits_imputed_path.txt).')
    ap.add_argument('--scale-factor', type=float, default=None,
                    help='Per-cell target sum after rescale.')
    ap.add_argument('--no-rescale', action='store_true',
                    help='Skip the per-cell rescale step; store FITS values verbatim.')
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    imp_cfg = cfg.setdefault('impute', {})
    if args.scale_factor is not None:
        imp_cfg['scale_factor'] = args.scale_factor

    fits_input = Path(cfg['paths']['fits_input'])
    fits_run   = Path(cfg['paths']['fits_run'])
    impute_dir = Path(cfg['paths']['impute'])
    impute_dir.mkdir(parents=True, exist_ok=True)

    if args.imputed_file:
        imp_path = Path(args.imputed_file)
    else:
        marker = fits_run / 'fits_imputed_path.txt'
        if not marker.exists():
            log.error('Missing %s. Run 04_run_fits_phase2.py first or pass --imputed-file.', marker)
            return 1
        imp_path = Path(marker.read_text().strip())
    if not imp_path.exists():
        log.error('Imputed file not found: %s', imp_path)
        return 1
    log.info('Reading FITS imputed matrix: %s', imp_path)
    imputed = _load_imputed_matrix(imp_path)
    log.info('Imputed shape %s, mean=%.4g, max=%.4g', imputed.shape,
             float(imputed.mean()), float(imputed.max()))

    selected_path = fits_input / 'regions.selected.tsv'
    barcodes_path = fits_input / 'barcodes.tsv'
    if not selected_path.exists() or not barcodes_path.exists():
        log.error('Missing prepare metadata (%s / %s). Run 02_prepare_fits_input.py first.',
                  selected_path, barcodes_path)
        return 1
    selected_regions = _read_lines(selected_path)
    barcodes         = _read_lines(barcodes_path)

    if imputed.shape != (len(selected_regions), len(barcodes)):
        log.error('Shape mismatch: imputed %s vs (selected_regions=%d, barcodes=%d).',
                  imputed.shape, len(selected_regions), len(barcodes))
        return 1

    desired_scale = float(imp_cfg.get('scale_factor', 1_000_000))
    if args.no_rescale:
        log.info('Storing FITS values verbatim (--no-rescale).')
        rescaled = imputed
        rescaled_flag = False
    else:
        sum_per_cell = np.asarray(imputed.sum(axis=0)).ravel()
        sum_per_cell_mean = float(sum_per_cell.mean()) if sum_per_cell.size else 0.0
        if sum_per_cell_mean <= 0:
            log.warning('FITS per-cell sum mean is non-positive (%.4g); storing verbatim.',
                        sum_per_cell_mean)
            rescaled = imputed
            rescaled_flag = False
        else:
            factor = desired_scale / sum_per_cell_mean
            log.info('Rescaling by %.4g so per-cell sum mean -> %.0f.', factor, desired_scale)
            rescaled = (imputed * factor).astype(np.float32)
            rescaled_flag = True

    matrix_path   = impute_dir / 'matrix.npy'
    regions_out   = impute_dir / 'regions.tsv'
    barcodes_out  = impute_dir / 'barcodes.tsv'
    meta_out      = impute_dir / 'meta.json'

    log.info('Writing %s (shape %s, dtype=float32)', matrix_path, rescaled.shape)
    np.save(matrix_path, rescaled.astype(np.float32, copy=False))
    regions_out.write_text('\n'.join(selected_regions) + '\n')
    barcodes_out.write_text('\n'.join(barcodes) + '\n')

    summary = {
        'pipeline':        'FITS',
        'format':          'dense_npy',
        'matrix_path':     str(matrix_path),
        'regions_path':    str(regions_out),
        'barcodes_path':   str(barcodes_out),
        'source_imputed':  str(imp_path),
        'shape':           [int(rescaled.shape[0]), int(rescaled.shape[1])],
        'n_modeled':       int(rescaled.shape[0]),
        'n_cells':         int(rescaled.shape[1]),
        'scale_factor':    desired_scale,
        'rescaled':        rescaled_flag,
    }
    meta_out.write_text(json.dumps(summary, indent=2) + '\n')
    log.info('Done: %s', summary)
    return 0


if __name__ == '__main__':
    sys.exit(main())
