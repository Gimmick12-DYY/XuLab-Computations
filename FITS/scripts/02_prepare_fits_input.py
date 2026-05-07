#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 02_prepare_fits_input.py
#
# Convert the Matrix Market output of 01_export_rds_to_mm.R into the dense CSV
# FITS Phase1 ingests. We drop the cisTopic-style cell/region quality filter
# entirely -- it was useful only because cisTopic's MALLET LDA degenerates
# without it; FITS does not need or want that filter.
#
# What we DO still need is a region subset, because FITS Phase1L grows trees
# over a dense matrix and can't ingest 3M rows. The default selection picks the
# top-N regions by nonzero-cell count from the full input matrix.
# -----------------------------------------------------------------------------
from __future__ import annotations

import gzip
import json
import logging
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

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
    ap.add_argument('--n-regions', type=int, default=None,
                    help='Top-N region subset for FITS. Pass 0 to disable subsetting '
                         '(likely intractable at CTCF scale).')
    ap.add_argument('--region-selection', type=str, default=None,
                    choices=['top_cells', 'top_counts', 'random'])
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    flt = cfg.setdefault('filter', {})
    prep = cfg.setdefault('prepare', {})

    if args.no_binarize:
        flt['binarize_input'] = False
    binarize = bool(flt.get('binarize_input', True))

    selection_mode = args.region_selection or prep.get('region_selection', 'top_cells')
    seed = int(args.seed if args.seed is not None else prep.get('seed', 42))
    rng = np.random.default_rng(seed)

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

    # -- binarise (no quality filter) -----------------------------------------
    # We keep the binarisation step because FITS needs a small dense CSV; with
    # the raw integer matrix the file would explode in size.
    if binarize:
        log.info('Binarising matrix (count > 0 -> 1).')
        mat = mat.astype(bool).astype(np.int8)

    # -- subset for FITS ------------------------------------------------------
    out_dir = Path(cfg['paths']['fits_input'])
    out_dir.mkdir(parents=True, exist_ok=True)

    if n_regions_cfg is None or n_regions_cfg >= mat.shape[0]:
        log.info('No region subsetting (n_regions=%s, available rows=%d).',
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
        sub_idx = np.sort(sub_idx)
        log.info('FITS subset: %s top-%d of %d regions.',
                 selection_mode, n_regions_cfg, mat.shape[0])

    sub = mat[sub_idx]
    selected_regions = regions[sub_idx].tolist()
    _write_lines(selected_regions, out_dir / 'regions.selected.tsv')
    _write_lines(list(barcodes),    out_dir / 'barcodes.tsv')

    dense = np.asarray(sub.toarray(), dtype=np.float32)
    csv_path = out_dir / 'fits_input.csv'
    log.info('Writing FITS dense input -> %s (shape %s)', csv_path, dense.shape)
    np.savetxt(csv_path, dense, delimiter=',', fmt='%.8g')

    meta = {
        'input_shape':           [int(mat.shape[0]), int(mat.shape[1])],
        'selected_shape':        [int(dense.shape[0]), int(dense.shape[1])],
        'n_regions_in_universe': int(len(regions)),
        'n_cells_in_universe':   int(len(barcodes)),
        'region_selection':      selection_mode,
        'n_regions_subset':      int(dense.shape[0]),
        'seed':                  seed,
        'binarize_input':        binarize,
        'input_nnz':             int(mat.nnz),
        'selected_nnz':          int(np.count_nonzero(dense)),
        'csv_path':              str(csv_path),
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
