#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 02_prepare_fits_input.py
#
# Apply cisTopic-style region/cell filtering and (optional) binarisation to the
# Matrix Market output of 01_export_rds_to_mm.R, then write a dense CSV that
# FITS can consume. The filtered universe (post region+cell filter) is recorded
# alongside the FITS-subset selection so 05_impute.py can pad imputed values
# back into the full pre-filter region space.
#
# FITS requires a dense matrix and FITSPhase1L's tree growth scales poorly past
# a few thousand rows, so a top-N subset of the filtered regions is taken by
# default. Setting prepare.n_regions: null disables subsetting and writes the
# entire filtered matrix (likely intractable at CTCF scale).
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
    ap.add_argument('--min-counts-per-region', type=int, default=None)
    ap.add_argument('--min-cells-per-region',  type=int, default=None)
    ap.add_argument('--min-regions-per-cell',  type=int, default=None)
    ap.add_argument('--no-binarize', action='store_true')
    ap.add_argument('--n-regions', type=int, default=None,
                    help='Top-N region subset for FITS. Pass 0 (or set in config to null) to disable.')
    ap.add_argument('--region-selection', type=str, default=None,
                    choices=['top_cells', 'top_counts', 'random'])
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    flt = cfg.setdefault('filter', {})
    prep = cfg.setdefault('prepare', {})

    for k, v in (
        ('min_counts_per_region', args.min_counts_per_region),
        ('min_cells_per_region',  args.min_cells_per_region),
        ('min_regions_per_cell',  args.min_regions_per_cell),
    ):
        if v is not None:
            flt[k] = v
    if args.no_binarize:
        flt['binarize_input'] = False

    selection_mode = args.region_selection or prep.get('region_selection', 'top_cells')
    seed = int(args.seed if args.seed is not None else prep.get('seed', 42))
    rng = np.random.default_rng(seed)

    # Resolve n_regions: CLI > config; CLI 0 => disable subsetting.
    if args.n_regions is not None:
        n_regions_cfg: int | None = args.n_regions if args.n_regions > 0 else None
    else:
        raw = prep.get('n_regions', None)
        n_regions_cfg = int(raw) if raw not in (None, 0, False) else None

    mm_dir = Path(cfg['paths']['mm'])
    mtx_path = mm_dir / 'matrix.mtx.gz'
    reg_path = mm_dir / 'regions.tsv.gz'
    bar_path = mm_dir / 'barcodes.tsv.gz'

    for p in (mtx_path, reg_path, bar_path):
        if not p.exists():
            log.error('Missing mm input: %s. Run 01_export_rds_to_mm.R first.', p)
            return 1

    log.info('Reading %s', mtx_path)
    mat = sio.mmread(mtx_path).tocsr()  # regions x cells
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

    out_dir = Path(cfg['paths']['fits_input'])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist the post-filter region universe and cell list. 05_impute.py uses
    # these (plus mm/regions.tsv.gz for the full universe) to align FITS output
    # back to the original genome coordinates.
    _write_lines(list(regions), out_dir / 'regions.filtered.tsv')
    _write_lines(list(barcodes), out_dir / 'barcodes.tsv')

    # -- subset for FITS ------------------------------------------------------
    if n_regions_cfg is None or n_regions_cfg >= mat.shape[0]:
        log.info('No region subsetting (n_regions=%s, filtered rows=%d).',
                 n_regions_cfg, mat.shape[0])
        sub_idx = np.arange(mat.shape[0])
    else:
        if selection_mode == 'top_cells':
            score = np.diff(mat.indptr)
            sub_idx = np.argsort(score)[::-1][:n_regions_cfg]
        elif selection_mode == 'top_counts':
            score = np.asarray(mat.sum(axis=1)).ravel()
            sub_idx = np.argsort(score)[::-1][:n_regions_cfg]
        elif selection_mode == 'random':
            sub_idx = rng.choice(mat.shape[0], size=n_regions_cfg, replace=False)
        else:
            log.error('Unknown region_selection: %s', selection_mode)
            return 1
        # Sort sub_idx so the row order matches a sensible region ordering on disk;
        # FITS itself doesn't care about order.
        sub_idx = np.sort(sub_idx)
        log.info('FITS subset: %s top-%d of %d filtered regions.',
                 selection_mode, n_regions_cfg, mat.shape[0])

    sub = mat[sub_idx]
    selected_regions = regions[sub_idx].tolist()
    _write_lines(selected_regions, out_dir / 'regions.selected.tsv')

    # FITS expects a dense numeric CSV. With binarised input most entries are 0
    # so '%g' keeps the file small.
    dense = np.asarray(sub.toarray(), dtype=np.float32)
    csv_path = out_dir / 'fits_input.csv'
    log.info('Writing FITS dense input -> %s (shape %s)', csv_path, dense.shape)
    np.savetxt(csv_path, dense, delimiter=',', fmt='%.8g')

    meta = {
        'input_shape': [int(mat.shape[0]), int(mat.shape[1])],
        'selected_shape': [int(dense.shape[0]), int(dense.shape[1])],
        'n_regions_in_universe': int(len(regions)),
        'n_cells_in_universe':   int(len(barcodes)),
        'region_selection': selection_mode,
        'n_regions_subset': int(dense.shape[0]),
        'seed': seed,
        'filter': {
            'min_counts_per_region': min_counts,
            'min_cells_per_region':  min_cells,
            'min_regions_per_cell':  min_regs,
            'binarize_input':        bool(flt.get('binarize_input', True)),
        },
        'input_nnz_filtered': int(mat.nnz),
        'selected_nnz':       int(np.count_nonzero(dense)),
        'csv_path':              str(csv_path),
        'regions_filtered_path': str(out_dir / 'regions.filtered.tsv'),
        'regions_selected_path': str(out_dir / 'regions.selected.tsv'),
        'barcodes_path':         str(out_dir / 'barcodes.tsv'),
    }
    (out_dir / 'prepare_meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    log.info('Selected matrix shape %s nnz=%d (density=%.3g)',
             dense.shape, meta['selected_nnz'],
             meta['selected_nnz'] / float(dense.size or 1))
    log.info('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
