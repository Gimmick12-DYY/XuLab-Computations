#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 05_impute.py
#
# Wrap the FITS Phase2 imputed CSV in the same HDF5 schema cisTopic writes:
#
#   imputed_Prc_hdf5_all.h5
#     /Prc        (R_full, C)  float32  imputed values padded to full region universe
#     /regions    (R_full,)    bytes    full pre-filter region IDs
#     /barcodes   (C,)         bytes    cell barcodes (post-cell-filter)
#     attrs:
#       scale_factor       configured target sum per cell
#       mode               "fits_hdf5_all"
#       n_modeled_regions  number of FITS-imputed rows
#       n_full_regions     R_full
#
# Why pad: cisTopic's eval scripts (08-11) expect the HDF5 row index to map 1:1
# to the original genome coordinates. Rows that were not modeled (filter drop
# OR FITS subset drop) sit as zeros and compress to almost nothing under gzip.
# This makes A/B comparison with cisTopic apples-to-apples.
# -----------------------------------------------------------------------------
from __future__ import annotations

import gzip
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger('05_impute')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def _read_lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def _read_lines_gz(path: Path) -> list[str]:
    with gzip.open(path, 'rt') as fh:
        return [ln.rstrip('\n') for ln in fh]


def _load_imputed_matrix(path: Path) -> np.ndarray:
    """Load FITS Phase2 output. Accepts CSV / TSV / TXT, comma- or whitespace-delimited."""
    suffix = path.suffix.lower()
    if suffix in ('.csv',):
        return np.loadtxt(path, delimiter=',', dtype=np.float32)
    if suffix in ('.tsv',):
        return np.loadtxt(path, delimiter='\t', dtype=np.float32)
    # txt or unknown: try comma first, then whitespace
    try:
        return np.loadtxt(path, delimiter=',', dtype=np.float32)
    except ValueError:
        return np.loadtxt(path, dtype=np.float32)


def main() -> int:
    ap = base_parser(__doc__ or '')
    ap.add_argument('--imputed-file', type=str, default=None,
                    help='FITS Phase2 output CSV (default: read from <fits_run>/fits_imputed_path.txt).')
    ap.add_argument('--scale-factor', type=float, default=None,
                    help='Per-cell target sum encoded in HDF5 values. Eval scripts will divide '
                         'by the observed per-cell sum mean to recover P(r|c).')
    ap.add_argument('--no-rescale', action='store_true',
                    help='Skip the per-cell rescale step; store FITS values verbatim.')
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    imp_cfg = cfg.setdefault('impute', {})
    if args.scale_factor is not None:
        imp_cfg['scale_factor'] = args.scale_factor

    fits_input = Path(cfg['paths']['fits_input'])
    fits_run = Path(cfg['paths']['fits_run'])
    impute_dir = Path(cfg['paths']['impute'])
    impute_dir.mkdir(parents=True, exist_ok=True)

    # -- locate FITS imputed output ------------------------------------------
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

    # -- alignment metadata --------------------------------------------------
    selected_path = fits_input / 'regions.selected.tsv'
    barcodes_path = fits_input / 'barcodes.tsv'
    if not selected_path.exists() or not barcodes_path.exists():
        log.error('Missing prepare metadata (%s / %s). Run 02_prepare_fits_input.py first.',
                  selected_path, barcodes_path)
        return 1
    selected_regions = _read_lines(selected_path)
    barcodes = _read_lines(barcodes_path)

    if imputed.shape != (len(selected_regions), len(barcodes)):
        log.error('Shape mismatch: imputed %s vs (selected_regions=%d, barcodes=%d).',
                  imputed.shape, len(selected_regions), len(barcodes))
        return 1

    full_regions_path = Path(cfg['paths']['mm']) / 'regions.tsv.gz'
    if not full_regions_path.exists():
        log.error('Cannot pad to full shape: %s missing.', full_regions_path)
        return 1
    full_regions = _read_lines_gz(full_regions_path)
    R_full = len(full_regions)
    C = len(barcodes)
    log.info('Full region universe: %d; FITS-modeled rows: %d (%.3f%% of universe)',
             R_full, len(selected_regions), 100.0 * len(selected_regions) / R_full)

    # -- optional per-cell rescale -------------------------------------------
    desired_scale = float(imp_cfg.get('scale_factor', 1_000_000))
    if args.no_rescale:
        log.info('Storing FITS values verbatim (--no-rescale).')
        rescaled = imputed
    else:
        sum_per_cell = np.asarray(imputed.sum(axis=0)).ravel()
        sum_per_cell_mean = float(sum_per_cell.mean()) if sum_per_cell.size else 0.0
        if sum_per_cell_mean <= 0:
            log.warning('FITS per-cell sum mean is non-positive (%.4g); '
                        'storing values verbatim.', sum_per_cell_mean)
            rescaled = imputed
        else:
            factor = desired_scale / sum_per_cell_mean
            log.info('Rescaling by %.4g so per-cell sum mean -> %.0f.', factor, desired_scale)
            rescaled = (imputed * factor).astype(np.float32)

    # -- pad to full region space --------------------------------------------
    selected_to_pos = {r: i for i, r in enumerate(selected_regions)}
    full_to_modeled = np.fromiter(
        (selected_to_pos.get(r, -1) for r in full_regions),
        dtype=np.int64, count=R_full,
    )
    n_present = int((full_to_modeled >= 0).sum())
    if n_present != len(selected_regions):
        log.warning('Modeled regions not all matched in full list: %d/%d found.',
                    n_present, len(selected_regions))

    h5_path = impute_dir / 'imputed_Prc_hdf5_all.h5'
    log.info('Writing HDF5 -> %s (shape %d x %d)', h5_path, R_full, C)
    with h5py.File(h5_path, 'w') as h5:
        dset = h5.create_dataset(
            'Prc',
            shape=(R_full, C),
            dtype=np.float32,
            compression='gzip', compression_opts=4,
            chunks=(min(4096, R_full), min(512, C)),
            fillvalue=np.float32(0.0),
        )
        chunk_rows = 50_000
        for start in range(0, R_full, chunk_rows):
            end = min(start + chunk_rows, R_full)
            idx = full_to_modeled[start:end]
            present = idx >= 0
            if not present.any():
                continue
            block = np.zeros((end - start, C), dtype=np.float32)
            block[present] = rescaled[idx[present]]
            dset[start:end] = block
        h5.create_dataset('regions',  data=np.asarray(full_regions, dtype='S'))
        h5.create_dataset('barcodes', data=np.asarray(barcodes,     dtype='S'))
        h5.attrs['scale_factor']      = float(desired_scale)
        h5.attrs['mode']              = 'fits_hdf5_all'
        h5.attrs['n_modeled_regions'] = int(len(selected_regions))
        h5.attrs['n_full_regions']    = int(R_full)
        h5.attrs['source_imputed']    = str(imp_path)

    summary = {
        'imputed_h5':       str(h5_path),
        'source_imputed':   str(imp_path),
        'shape':            [R_full, C],
        'n_modeled_regions': int(len(selected_regions)),
        'n_full_regions':    R_full,
        'n_cells':           C,
        'scale_factor':      desired_scale,
        'rescaled':          not args.no_rescale,
    }
    (impute_dir / 'imputed_Prc_hdf5_all.meta.json').write_text(
        json.dumps(summary, indent=2) + '\n'
    )
    log.info('Done: %s', summary)
    return 0


if __name__ == '__main__':
    sys.exit(main())
