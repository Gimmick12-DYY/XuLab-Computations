#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 05_impute.py
#
# Reconstruct the imputed peak-by-cell matrix from scOpen's saved factors
# (W = peaks x k, H = k x cells) and persist it to the same HDF5 schema cisTopic
# and FITS use, padded to the full pre-filter region universe.
#
# We never materialise the full dense (R_full x C) matrix in memory: rows are
# computed in chunks of W @ H and streamed straight into the gzip-compressed
# HDF5 dataset. Zero-padded rows for filtered-out regions compress to almost
# nothing.
# -----------------------------------------------------------------------------
from __future__ import annotations

import gzip
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger('05_impute')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def _read_lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def _read_lines_gz(path: Path) -> list[str]:
    with gzip.open(path, 'rt') as fh:
        return [ln.rstrip('\n') for ln in fh]


def main() -> int:
    ap = base_parser(__doc__ or '')
    ap.add_argument('--output-prefix', type=str, default='scOpen')
    ap.add_argument('--scale-factor', type=float, default=None)
    ap.add_argument('--no-rescale', action='store_true')
    ap.add_argument('--chunk-rows', type=int, default=20_000,
                    help='Row block size when streaming W @ H into HDF5.')
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
    # W: index = peak names, columns = 0..k-1
    W_df = pd.read_csv(w_path, sep='\t', index_col=0)
    log.info('  W shape %s', W_df.shape)

    log.info('Reading H from %s', h_path)
    # H: rows = 0..k-1 (default integer index), columns = barcodes
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

    # Cross-check against the filter universe written by step 02 so we catch
    # accidental config drift.
    fil_path = input_dir / 'regions.filtered.tsv'
    if fil_path.exists():
        fil_regions = _read_lines(fil_path)
        if fil_regions != selected_regions:
            log.warning('regions.filtered.tsv (%d rows) does not exactly match '
                        'W index (%d rows). Trusting W; ensure 02 and 03 ran '
                        'against the same input.',
                        len(fil_regions), len(selected_regions))

    full_regions_path = Path(cfg['paths']['mm']) / 'regions.tsv.gz'
    if not full_regions_path.exists():
        log.error('Cannot pad to full shape: %s missing.', full_regions_path)
        return 1
    full_regions = _read_lines_gz(full_regions_path)
    R_full = len(full_regions)
    C = H.shape[1]
    log.info('Full region universe: %d; modeled rows: %d (%.2f%% of universe)',
             R_full, len(selected_regions),
             100.0 * len(selected_regions) / R_full)

    # -- per-cell rescale ----------------------------------------------------
    desired_scale = float(imp_cfg.get('scale_factor', 1_000_000))
    if args.no_rescale:
        log.info('Storing W @ H verbatim (--no-rescale).')
        scale_factor_attr = float(desired_scale)
        per_cell_factor = None
    else:
        # Compute per-cell sum of W @ H without materialising the full matrix:
        # sum_r (W @ H)[r,:]  =  (sum_r W) @ H
        col_sums = (W.sum(axis=0) @ H).astype(np.float64)
        sum_per_cell_mean = float(col_sums.mean()) if col_sums.size else 0.0
        if sum_per_cell_mean <= 0:
            log.warning('W @ H per-cell sum mean <= 0; storing values verbatim.')
            per_cell_factor = None
            scale_factor_attr = float(desired_scale)
        else:
            factor = desired_scale / sum_per_cell_mean
            log.info('Rescaling by %.4g so per-cell sum mean -> %.0f.',
                     factor, desired_scale)
            per_cell_factor = float(factor)
            scale_factor_attr = float(desired_scale)

    # -- index map: original full row -> W row (-1 if not modeled) ----------
    selected_to_pos = {r: i for i, r in enumerate(selected_regions)}
    full_to_modeled = np.fromiter(
        (selected_to_pos.get(r, -1) for r in full_regions),
        dtype=np.int64, count=R_full,
    )
    n_present = int((full_to_modeled >= 0).sum())
    if n_present != len(selected_regions):
        log.warning('Modeled regions not all matched in full list: %d/%d found.',
                    n_present, len(selected_regions))

    # -- stream W @ H row-blocks into HDF5 ----------------------------------
    h5_path = impute_dir / 'imputed_Prc_hdf5_all.h5'
    log.info('Writing HDF5 -> %s (shape %d x %d)', h5_path, R_full, C)
    chunk = max(1, int(args.chunk_rows))
    with h5py.File(h5_path, 'w') as h5:
        dset = h5.create_dataset(
            'Prc',
            shape=(R_full, C),
            dtype=np.float32,
            compression='gzip', compression_opts=4,
            chunks=(min(4096, R_full), min(512, C)),
            fillvalue=np.float32(0.0),
        )
        for start in range(0, R_full, chunk):
            end = min(start + chunk, R_full)
            idx = full_to_modeled[start:end]
            present = idx >= 0
            if not present.any():
                continue
            block = np.zeros((end - start, C), dtype=np.float32)
            # Compute (W[idx[present]] @ H) just for the rows that have a model.
            block[present] = W[idx[present]] @ H
            if per_cell_factor is not None:
                block[present] *= per_cell_factor
            dset[start:end] = block
        h5.create_dataset('regions',  data=np.asarray(full_regions, dtype='S'))
        h5.create_dataset('barcodes', data=np.asarray(barcodes,     dtype='S'))
        h5.attrs['scale_factor']      = scale_factor_attr
        h5.attrs['mode']              = 'scopen_hdf5_all'
        h5.attrs['n_modeled_regions'] = int(len(selected_regions))
        h5.attrs['n_full_regions']    = int(R_full)
        h5.attrs['rank_k']            = int(W.shape[1])
        h5.attrs['source_w']          = str(w_path)
        h5.attrs['source_h']          = str(h_path)

    summary = {
        'imputed_h5':        str(h5_path),
        'shape':             [R_full, C],
        'n_modeled_regions': int(len(selected_regions)),
        'n_full_regions':    R_full,
        'n_cells':           C,
        'rank_k':            int(W.shape[1]),
        'scale_factor':      scale_factor_attr,
        'rescaled':          per_cell_factor is not None,
    }
    (impute_dir / 'imputed_Prc_hdf5_all.meta.json').write_text(
        json.dumps(summary, indent=2) + '\n'
    )
    log.info('Done: %s', summary)
    return 0


if __name__ == '__main__':
    sys.exit(main())
