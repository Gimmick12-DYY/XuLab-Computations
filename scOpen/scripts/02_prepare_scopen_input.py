#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 02_prepare_scopen_input.py
#
# Apply cisTopic-style region/cell filtering and (optional) binarisation to the
# Matrix Market output of 01_export_rds_to_mm.R, then write a 10X-format input
# directory that scOpen can ingest:
#
#   <work>/scopen_input/
#     matrix.mtx     un-gzipped, integer-valued, regions x cells
#     barcodes.tsv   one barcode per line
#     peaks.bed      tab-separated chrom\tstart\tend (parsed from chr:start-end)
#
# scOpen's `--input_format 10X` loader reads the mtx via scipy.io.mmread and
# rebuilds peak IDs as `<chrom>:<start>-<end>`, so we keep the BED columns in
# sync with our region naming convention.
#
# Unlike FITS, scOpen accepts sparse input natively, so there is NO top-N
# region subset — we run scOpen on the entire post-filter universe.
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
    ap.add_argument('--min-counts-per-region', type=int, default=None)
    ap.add_argument('--min-cells-per-region',  type=int, default=None)
    ap.add_argument('--min-regions-per-cell',  type=int, default=None)
    ap.add_argument('--no-binarize', action='store_true')
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    flt = cfg.setdefault('filter', {})

    for k, v in (
        ('min_counts_per_region', args.min_counts_per_region),
        ('min_cells_per_region',  args.min_cells_per_region),
        ('min_regions_per_cell',  args.min_regions_per_cell),
    ):
        if v is not None:
            flt[k] = v
    if args.no_binarize:
        flt['binarize_input'] = False

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

    # -- region filter --------------------------------------------------------
    min_counts = int(flt.get('min_counts_per_region', 0))
    min_cells  = int(flt.get('min_cells_per_region', 0))
    if min_counts > 0 or min_cells > 0:
        row_sum = np.asarray(mat.sum(axis=1)).ravel()
        row_nnz = np.diff(mat.indptr)
        keep_rows = (row_sum >= min_counts) & (row_nnz >= min_cells)
        log.info(
            'Region filter (counts>=%d, cells>=%d): keeping %d / %d (%.2f%%)',
            min_counts, min_cells, int(keep_rows.sum()), len(keep_rows),
            100.0 * keep_rows.mean(),
        )
        mat = mat[keep_rows]
        regions = regions[keep_rows]

    # -- binarise -------------------------------------------------------------
    if flt.get('binarize_input', True):
        log.info('Binarising matrix (count > 0 -> 1).')
        mat = mat.astype(bool).astype(np.int8)

    # -- cell filter ----------------------------------------------------------
    min_regs = int(flt.get('min_regions_per_cell', 0))
    if min_regs > 0:
        col_nnz = np.diff(mat.tocsc().indptr)
        keep_cols = col_nnz >= min_regs
        log.info('Cell filter (regions>=%d): keeping %d / %d cells.',
                 min_regs, int(keep_cols.sum()), len(keep_cols))
        mat = mat[:, keep_cols]
        barcodes = barcodes[keep_cols]

    log.info('Post-filter matrix: %d regions x %d cells, nnz=%d',
             *mat.shape, mat.nnz)

    # -- write 10X-format input directory ------------------------------------
    out_dir = Path(cfg['paths']['scopen_input'])
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_path  = out_dir / 'matrix.mtx'
    barcode_path = out_dir / 'barcodes.tsv'
    peak_path    = out_dir / 'peaks.bed'

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

    # Persist filtered region list (in original chr:start-end format) so step
    # 05 can pad the imputed matrix back into the full pre-filter universe.
    _write_lines(list(regions), out_dir / 'regions.filtered.tsv')

    meta = {
        'filtered_shape':        [int(mat.shape[0]), int(mat.shape[1])],
        'n_regions_in_universe': int(len(regions)),
        'n_cells':               int(len(barcodes)),
        'filter': {
            'min_counts_per_region': min_counts,
            'min_cells_per_region':  min_cells,
            'min_regions_per_cell':  min_regs,
            'binarize_input':        bool(flt.get('binarize_input', True)),
        },
        'nnz':                   int(mat.nnz),
        'matrix_path':           str(matrix_path),
        'barcodes_path':         str(barcode_path),
        'peaks_path':            str(peak_path),
        'regions_filtered_path': str(out_dir / 'regions.filtered.tsv'),
    }
    (out_dir / 'prepare_meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    log.info('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
