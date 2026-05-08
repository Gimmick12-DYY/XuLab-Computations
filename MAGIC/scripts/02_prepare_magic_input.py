#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 02_prepare_magic_input.py
#
# Convert the Matrix Market output of 01_export_rds_to_mm.R into a sparse
# (regions x cells) input that 03_run_magic.py loads, transposes to
# (cells x features), and feeds into MAGIC.fit_transform.
#
# Outputs (under <work>/magic_input/):
#   matrix.npz       scipy.sparse.csr_matrix saved with sp.save_npz
#                    (rows = regions, cols = cells; transposed by step 03)
#   regions.tsv      one chr:start-end per row, in matrix row order
#   barcodes.tsv     one cell barcode per col, in matrix col order
#   prepare_meta.json
#
# We binarise (matching scOpen / cisTopic) and drop all-zero rows because they
# contribute zero to the diffusion. We do NOT apply the cisTopic-style cell or
# min_cells_per_region quality filter -- MAGIC's diffusion handles sparsity by
# design.
# -----------------------------------------------------------------------------
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

log = logging.getLogger('02_prepare')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def _read_lines_gz(path: Path) -> list[str]:
    with gzip.open(path, 'rt') as fh:
        return [ln.rstrip('\n') for ln in fh]


def _write_lines(lines: list[str], path: Path) -> None:
    path.write_text('\n'.join(lines) + '\n')


def main() -> int:
    ap = base_parser(__doc__ or '')
    ap.add_argument('--no-binarize', action='store_true')
    ap.add_argument('--no-drop-zero-rows', dest='drop_zero_rows',
                    action='store_false', default=None,
                    help='Keep all-zero rows. Default: drop them (uninformative).')
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    flt = cfg.setdefault('filter', {})
    if args.no_binarize:
        flt['binarize_input'] = False
    if args.drop_zero_rows is False:
        flt['drop_zero_rows'] = False
    binarize = bool(flt.get('binarize_input', True))
    drop_zero = bool(flt.get('drop_zero_rows', True))

    mm_dir = Path(cfg['paths']['mm'])
    mtx_path = mm_dir / 'matrix.mtx.gz'
    reg_path = mm_dir / 'regions.tsv.gz'
    bar_path = mm_dir / 'barcodes.tsv.gz'
    for p in (mtx_path, reg_path, bar_path):
        if not p.exists():
            log.error('Missing mm input: %s. Run 01_export_rds_to_mm.R first.', p)
            return 1

    log.info('Reading %s', mtx_path)
    mat = sio.mmread(mtx_path).tocsr()
    regions = np.asarray(_read_lines_gz(reg_path))
    barcodes = np.asarray(_read_lines_gz(bar_path))
    if mat.shape != (len(regions), len(barcodes)):
        log.error('Shape mismatch: matrix %s vs regions %d / barcodes %d',
                  mat.shape, len(regions), len(barcodes))
        return 1
    log.info('Loaded %d regions x %d cells, nnz=%d (density=%.3g)',
             *mat.shape, mat.nnz, mat.nnz / float(mat.shape[0] * mat.shape[1]))

    if binarize:
        log.info('Binarising matrix (count > 0 -> 1).')
        mat = mat.astype(bool).astype(np.int8)

    if drop_zero:
        row_nnz = np.diff(mat.indptr)
        keep_rows = row_nnz > 0
        n_drop = int((~keep_rows).sum())
        if n_drop > 0:
            log.info('Dropping %d / %d all-zero rows; keeping %d.',
                     n_drop, len(keep_rows), int(keep_rows.sum()))
            mat = mat[keep_rows]
            regions = regions[keep_rows]

    log.info('MAGIC input matrix: %d regions x %d cells, nnz=%d',
             *mat.shape, mat.nnz)

    out_dir = Path(cfg['paths']['magic_input'])
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path  = out_dir / 'matrix.npz'
    regions_path = out_dir / 'regions.tsv'
    barcode_path = out_dir / 'barcodes.tsv'

    log.info('Writing %s', matrix_path)
    sp.save_npz(matrix_path, mat.tocsr())

    _write_lines(list(regions),  regions_path)
    _write_lines(list(barcodes), barcode_path)

    meta = {
        'shape':           [int(mat.shape[0]), int(mat.shape[1])],
        'n_regions':       int(mat.shape[0]),
        'n_cells':         int(mat.shape[1]),
        'binarize_input':  binarize,
        'drop_zero_rows':  drop_zero,
        'nnz':             int(mat.nnz),
        'matrix_path':     str(matrix_path),
        'regions_path':    str(regions_path),
        'barcodes_path':   str(barcode_path),
    }
    (out_dir / 'prepare_meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    log.info('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
