#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 02_prepare_scopen_input.py
#
# Convert the Matrix Market output of 01_export_rds_to_mm.R into the 10X-format
# directory scOpen ingests:
#
#   <work>/scopen_input/
#     matrix.mtx          un-gzipped, integer-valued, regions x cells
#     barcodes.tsv        one barcode per line
#     peaks.bed           tab-separated chrom\tstart\tend
#     regions.tsv         original chr:start-end strings, in row order
#
# scOpen's `--input_format 10X` loader reads the mtx via scipy.io.mmread and
# rebuilds peak IDs as `<chrom>:<start>-<end>`, so we keep the BED columns in
# sync with our region naming convention.
#
# We do NOT apply any cisTopic-style cell/region filter -- scOpen handles
# sparse input natively (TF-IDF + sparse NMF) and does not need that filter
# to converge.
# -----------------------------------------------------------------------------
from __future__ import annotations

import gzip
import json
import logging
import re
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


_REGION_RE = re.compile(r'^([^:]+):(\d+)-(\d+)$')


def _region_to_bed_row(name: str) -> tuple[str, str, str]:
    m = _REGION_RE.match(name)
    if not m:
        raise SystemExit(f'Cannot parse region {name!r} as chr:start-end. '
                         f'Re-run 01_export_rds_to_mm.R or pass plain coords.')
    return m.group(1), m.group(2), m.group(3)


def main() -> int:
    ap = base_parser(__doc__ or '')
    ap.add_argument('--no-binarize', action='store_true')
    ap.add_argument('--drop-zero-rows', action='store_true', default=True,
                    help='Drop regions with zero counts across all cells. '
                         'These are uninformative and waste memory in scOpen NMF. '
                         'Default: True.')
    ap.add_argument('--no-drop-zero-rows', dest='drop_zero_rows',
                    action='store_false')
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    flt = cfg.setdefault('filter', {})

    if args.no_binarize:
        flt['binarize_input'] = False
    binarize = bool(flt.get('binarize_input', True))

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

    # -- binarise -------------------------------------------------------------
    if binarize:
        log.info('Binarising matrix (count > 0 -> 1).')
        mat = mat.astype(bool).astype(np.int8)

    # -- drop all-zero rows ---------------------------------------------------
    # NMF cannot do anything useful with all-zero rows, so we drop them. This
    # is NOT the cisTopic-style quality filter -- it just removes rows that
    # contribute zero information to the factorisation.
    if args.drop_zero_rows:
        row_nnz = np.diff(mat.indptr)
        keep_rows = row_nnz > 0
        n_drop = int((~keep_rows).sum())
        if n_drop > 0:
            log.info('Dropping %d / %d all-zero rows; keeping %d.',
                     n_drop, len(keep_rows), int(keep_rows.sum()))
            mat = mat[keep_rows]
            regions = regions[keep_rows]

    log.info('scOpen input matrix: %d regions x %d cells, nnz=%d',
             *mat.shape, mat.nnz)

    # -- write 10X-format input directory ------------------------------------
    out_dir = Path(cfg['paths']['scopen_input'])
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_path  = out_dir / 'matrix.mtx'
    barcode_path = out_dir / 'barcodes.tsv'
    peak_path    = out_dir / 'peaks.bed'
    regions_path = out_dir / 'regions.tsv'

    log.info('Writing %s', matrix_path)
    sio.mmwrite(str(matrix_path), sp.coo_matrix(mat),
                field='integer', symmetry='general')

    log.info('Writing %s', barcode_path)
    _write_lines(list(barcodes), barcode_path)

    log.info('Writing %s', peak_path)
    with open(peak_path, 'w') as fh:
        for r in regions:
            chrom, s, e = _region_to_bed_row(str(r))
            fh.write(f'{chrom}\t{s}\t{e}\n')

    _write_lines(list(regions), regions_path)

    meta = {
        'shape':              [int(mat.shape[0]), int(mat.shape[1])],
        'n_regions':          int(mat.shape[0]),
        'n_cells':            int(mat.shape[1]),
        'binarize_input':     binarize,
        'drop_zero_rows':     bool(args.drop_zero_rows),
        'nnz':                int(mat.nnz),
        'matrix_path':        str(matrix_path),
        'barcodes_path':      str(barcode_path),
        'peaks_path':         str(peak_path),
        'regions_path':       str(regions_path),
    }
    (out_dir / 'prepare_meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    log.info('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
