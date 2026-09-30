#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 05_impute.py
#
# Persist the scOpen factorisation in a compact, HDF5-free format that the
# universal downstream/compare_pos_neg.py consumes:
#
#   <work>/impute/
#     factors.npz       W (n_modeled, k) + H (k, n_cells) + per_cell_factor
#     regions.tsv       one chr:start-end per row, in the order matching W's rows
#     barcodes.tsv      one cell barcode per col, matching H's columns
#     meta.json         k, scale_factor, source paths, n_modeled, n_cells, ...
#
# Downstream tools reconstruct any rows they need on-demand via
#   W[idx] @ H * per_cell_factor
# without ever materialising the full dense (R x C) matrix.
# -----------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger('05_impute')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def _read_lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def main() -> int:
    ap = base_parser(__doc__ or '')
    ap.add_argument('--output-prefix', type=str, default='scOpen')
    ap.add_argument('--scale-factor', type=float, default=None)
    ap.add_argument('--no-rescale', action='store_true')
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    imp_cfg = cfg.setdefault('impute', {})
    if args.scale_factor is not None:
        imp_cfg['scale_factor'] = args.scale_factor

    run_dir    = Path(cfg['paths']['scopen_run'])
    input_dir  = Path(cfg['paths']['scopen_input'])
    impute_dir = Path(cfg['paths']['impute'])
    impute_dir.mkdir(parents=True, exist_ok=True)

    w_path = run_dir / f'{args.output_prefix}_peaks.txt'
    h_path = run_dir / f'{args.output_prefix}_barcodes.txt'
    if not w_path.exists() or not h_path.exists():
        log.error('Missing factor files (W=%s, H=%s). Run 03_run_scopen.py.',
                  w_path, h_path)
        return 1

    log.info('Reading W from %s', w_path)
    W_df = pd.read_csv(w_path, sep='\t', index_col=0)
    log.info('  W shape %s', W_df.shape)

    log.info('Reading H from %s', h_path)
    H_df = pd.read_csv(h_path, sep='\t', index_col=0)
    log.info('  H shape %s', H_df.shape)

    if W_df.shape[1] != H_df.shape[0]:
        log.error('Rank mismatch: W has %d cols, H has %d rows.',
                  W_df.shape[1], H_df.shape[0])
        return 1

    selected_regions = list(W_df.index.astype(str))
    barcodes = list(H_df.columns.astype(str))
    W = W_df.to_numpy(dtype=np.float32, copy=False)
    H = H_df.to_numpy(dtype=np.float32, copy=False)
    del W_df, H_df

    fil_path = input_dir / 'regions.tsv'
    if fil_path.exists():
        fil_regions = _read_lines(fil_path)
        if fil_regions != selected_regions:
            log.warning('regions.tsv (%d rows) does not exactly match W index '
                        '(%d rows). Trusting W; ensure 02 and 03 ran against '
                        'the same input.',
                        len(fil_regions), len(selected_regions))

    desired_scale = float(imp_cfg.get('scale_factor', 1_000_000))
    if args.no_rescale:
        log.info('Storing W @ H verbatim (--no-rescale).')
        per_cell_factor = float(1.0)
        rescaled = False
    else:
        col_sums = (W.sum(axis=0) @ H).astype(np.float64)
        sum_per_cell_mean = float(col_sums.mean()) if col_sums.size else 0.0
        if sum_per_cell_mean <= 0:
            log.warning('W @ H per-cell sum mean <= 0; storing factors verbatim.')
            per_cell_factor = float(1.0)
            rescaled = False
        else:
            per_cell_factor = float(desired_scale / sum_per_cell_mean)
            log.info('Rescaling factor=%.4g so per-cell sum mean -> %.0f.',
                     per_cell_factor, desired_scale)
            rescaled = True

    factors_path = impute_dir / 'factors.npz'
    regions_out  = impute_dir / 'regions.tsv'
    barcodes_out = impute_dir / 'barcodes.tsv'
    meta_out     = impute_dir / 'meta.json'

    log.info('Writing %s (W=%s, H=%s, per_cell_factor=%.4g)',
             factors_path, W.shape, H.shape, per_cell_factor)
    np.savez_compressed(
        factors_path,
        W=W.astype(np.float32, copy=False),
        H=H.astype(np.float32, copy=False),
        per_cell_factor=np.float32(per_cell_factor),
    )
    regions_out.write_text('\n'.join(selected_regions) + '\n')
    barcodes_out.write_text('\n'.join(barcodes) + '\n')

    summary = {
        'pipeline':        'scOpen',
        'format':          'factored_npz',
        'factors_path':    str(factors_path),
        'regions_path':    str(regions_out),
        'barcodes_path':   str(barcodes_out),
        'shape':           [int(W.shape[0]), int(H.shape[1])],
        'n_modeled':       int(W.shape[0]),
        'n_cells':         int(H.shape[1]),
        'rank_k':          int(W.shape[1]),
        'scale_factor':    desired_scale,
        'per_cell_factor': float(per_cell_factor),
        'rescaled':        rescaled,
        'source_w':        str(w_path),
        'source_h':        str(h_path),
    }
    meta_out.write_text(json.dumps(summary, indent=2) + '\n')
    log.info('Done: %s', summary)
    return 0


if __name__ == '__main__':
    sys.exit(main())
