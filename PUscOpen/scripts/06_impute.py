#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 06_impute.py
#
# Materialise the final imputed (regions x cells) matrix:
#
#     M[r, c]  =  damp[r] * (W @ H)[r, c]  * rescale_factor
#
# where damp comes from step 05 (neighborhood support * optional PU factor)
# and rescale_factor optionally normalises so per-cell sums average
# impute.scale_factor (CPM-like; mostly a convention so all five pipelines
# end up on the same scale).
#
# Outputs (under <work>/impute/):
#   Default impute.align_to_mm: true
#     regions.tsv     full mm row order (copied from mm/regions.tsv.gz)
#     matrix_csr.npz  scipy CSR float32, shape (n_mm_bins, n_cells), nnz only
#                     on refined-mask rows (fair vs full-matrix pipelines)
#     barcodes.tsv    one barcode per col
#     meta.json
#   Legacy impute.align_to_mm: false
#     matrix.npy      float32 (R_refined, C)
#     regions.tsv     refined rows only
#     barcodes.tsv
#     meta.json
# -----------------------------------------------------------------------------
from __future__ import annotations

import gzip
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger('06_impute')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def _read_mm_region_lines(mm_dir: Path) -> list[str]:
    reg_gz = mm_dir / 'regions.tsv.gz'
    reg_plain = mm_dir / 'regions.tsv'
    if reg_gz.is_file():
        with gzip.open(reg_gz, 'rt') as fh:
            return [ln.rstrip('\n') for ln in fh if ln.strip()]
    if reg_plain.is_file():
        return [ln for ln in reg_plain.read_text().splitlines() if ln.strip()]
    raise SystemExit(f'Missing {reg_gz} and {reg_plain}; cannot align to mm.')


def main() -> int:
    ap = base_parser(__doc__ or '')
    ap.add_argument('--output-prefix', type=str, default='scOpen')
    ap.add_argument('--scale-factor', type=float, default=None)
    ap.add_argument('--no-rescale', action='store_true')
    ap.add_argument('--chunk-rows', type=int, default=20_000)
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    imp_cfg = cfg.setdefault('impute', {})
    if args.scale_factor is not None:
        imp_cfg['scale_factor'] = args.scale_factor
    align_to_mm = bool(imp_cfg.get('align_to_mm', True))

    run_dir = Path(cfg['paths']['scopen_run'])
    pf_dir  = Path(cfg['paths']['postfilter'])
    imp_dir = Path(cfg['paths']['impute'])
    mm_dir  = Path(cfg['paths']['mm'])
    imp_dir.mkdir(parents=True, exist_ok=True)

    w_path = run_dir / f'{args.output_prefix}_peaks.txt'
    h_path = run_dir / f'{args.output_prefix}_barcodes.txt'
    damp_path = pf_dir / 'damping.npy'
    for p in (w_path, h_path, damp_path):
        if not p.exists():
            log.error('Missing %s; run earlier steps first.', p)
            return 1

    log.info('Loading W and H ...')
    W_df = pd.read_csv(w_path, sep='\t', index_col=0)
    H_df = pd.read_csv(h_path, sep='\t', index_col=0)
    if W_df.shape[1] != H_df.shape[0]:
        log.error('Rank mismatch: W has %d cols, H has %d rows.',
                  W_df.shape[1], H_df.shape[0])
        return 1
    regions = list(W_df.index.astype(str))
    barcodes = list(H_df.columns.astype(str))
    W = W_df.to_numpy(dtype=np.float32, copy=False)
    H = H_df.to_numpy(dtype=np.float32, copy=False)
    del W_df, H_df
    Rm, C = W.shape[0], H.shape[1]
    log.info('  W=%s, H=%s', W.shape, H.shape)

    log.info('Loading damping vector %s', damp_path)
    damp = np.load(damp_path).astype(np.float32)
    if damp.shape[0] != Rm:
        log.error('Damping length %d != W rows %d.', damp.shape[0], Rm)
        return 1
    log.info('  damp stats: mean=%.4g, p10=%.4g, p90=%.4g',
             float(damp.mean()),
             float(np.quantile(damp, 0.10)),
             float(np.quantile(damp, 0.90)))

    desired_scale = float(imp_cfg.get('scale_factor', 1_000_000))
    rescale_factor = 1.0
    rescaled_flag = False
    if not args.no_rescale:
        col_sums = (W * damp[:, None]).sum(axis=0, dtype=np.float64) @ H.astype(np.float64)
        mean_sum = float(col_sums.mean()) if col_sums.size else 0.0
        if mean_sum > 0:
            rescale_factor = float(desired_scale / mean_sum)
            rescaled_flag = True
            log.info('Per-cell sum mean = %.4g -> rescale by %.4g (target %.0f)',
                     mean_sum, rescale_factor, desired_scale)
        else:
            log.warning('Per-cell sum mean <= 0; storing values verbatim.')

    barcodes_out = imp_dir / 'barcodes.tsv'
    meta_out     = imp_dir / 'meta.json'
    barcodes_out.write_text('\n'.join(barcodes) + '\n')

    if align_to_mm:
        full_regions = _read_mm_region_lines(mm_dir)
        N = len(full_regions)
        name_to_row = {nm: i for i, nm in enumerate(full_regions)}
        try:
            full_rows = np.array([name_to_row[nm] for nm in regions], dtype=np.int64)
        except KeyError as e:
            raise SystemExit(
                f'Refined region {e!r} missing from mm regions list; '
                f'cannot align output.'
            ) from e
        if len(full_rows) != len(regions):
            raise SystemExit('internal: refined name list length mismatch')
        if N == 0 or C == 0:
            raise SystemExit('empty mm regions or zero cells')

        contrib = np.zeros(N, dtype=np.int64)
        contrib[full_rows] = C
        indptr = np.zeros(N + 1, dtype=np.int64)
        np.cumsum(contrib, out=indptr[1:])
        nnz = int(indptr[-1])
        log.info('Building CSR aligned to mm: shape=(%d, %d), nnz=%d (%.3g MB data)',
                 N, C, nnz, nnz * 4 / (1024 ** 2))

        data = np.empty(nnz, dtype=np.float32)
        indices = np.empty(nnz, dtype=np.int32)
        write_off = indptr[:-1].copy()

        chunk = max(1, int(args.chunk_rows))
        for s in range(0, Rm, chunk):
            e = min(s + chunk, Rm)
            block = (W[s:e] @ H).astype(np.float32, copy=False)
            block *= damp[s:e, None]
            if rescaled_flag:
                block *= np.float32(rescale_factor)
            for off, loc in enumerate(range(s, e)):
                g = int(full_rows[loc])
                start = int(write_off[g])
                data[start:start + C] = block[off]
                indices[start:start + C] = np.arange(C, dtype=np.int32)
                write_off[g] = start + C

        csr = sparse.csr_matrix((data, indices, indptr), shape=(N, C), dtype=np.float32)
        csr.check_format(full_check=False)

        csr_path = imp_dir / 'matrix_csr.npz'
        sparse.save_npz(csr_path, csr)
        log.info('Wrote %s', csr_path)

        regions_out = imp_dir / 'regions.tsv'
        log.info('Writing full regions list %s (%d lines) ...', regions_out, N)
        with open(regions_out, 'w') as fh:
            for line in full_regions:
                fh.write(line + '\n')

        # Remove legacy compact artifact if present from an older run.
        legacy = imp_dir / 'matrix.npy'
        if legacy.exists():
            legacy.unlink()
            log.info('Removed stale %s', legacy)

        summary = {
            'pipeline':       'PUscOpen',
            'format':         'csr_full_mm',
            'matrix_path':    str(csr_path),
            'regions_path':   str(regions_out),
            'barcodes_path':  str(barcodes_out),
            'shape':          [int(N), int(C)],
            'nnz':            int(csr.nnz),
            'n_modeled_rows': int(Rm),
            'n_cells':        int(C),
            'rank_k':         int(W.shape[1]),
            'scale_factor':   desired_scale,
            'rescale_factor': float(rescale_factor),
            'rescaled':       rescaled_flag,
            'align_to_mm':    True,
            'source_w':       str(w_path),
            'source_h':       str(h_path),
            'damping':        str(damp_path),
        }
        meta_out.write_text(json.dumps(summary, indent=2) + '\n')
        log.info('Done: %s', summary)
        return 0

    # ----- Legacy compact dense output -----
    matrix_path  = imp_dir / 'matrix.npy'
    regions_out  = imp_dir / 'regions.tsv'
    log.info('Materialising %s (shape=%s, dtype=float32) ...', matrix_path, (Rm, C))
    out = np.lib.format.open_memmap(matrix_path, mode='w+', dtype=np.float32,
                                     shape=(Rm, C))
    chunk = max(1, int(args.chunk_rows))
    for s in range(0, Rm, chunk):
        e = min(s + chunk, Rm)
        block = (W[s:e] @ H).astype(np.float32, copy=False)
        block *= damp[s:e, None]
        if rescaled_flag:
            block *= rescale_factor
        out[s:e] = block
    out.flush(); del out

    regions_out.write_text('\n'.join(regions) + '\n')

    csr_path = imp_dir / 'matrix_csr.npz'
    if csr_path.exists():
        csr_path.unlink()

    summary = {
        'pipeline':       'PUscOpen',
        'format':         'dense_npy',
        'matrix_path':    str(matrix_path),
        'regions_path':   str(regions_out),
        'barcodes_path':  str(barcodes_out),
        'shape':          [int(Rm), int(C)],
        'n_modeled':      int(Rm),
        'n_cells':        int(C),
        'rank_k':         int(W.shape[1]),
        'scale_factor':   desired_scale,
        'rescale_factor': float(rescale_factor),
        'rescaled':       rescaled_flag,
        'align_to_mm':    False,
        'source_w':       str(w_path),
        'source_h':       str(h_path),
        'damping':        str(damp_path),
    }
    meta_out.write_text(json.dumps(summary, indent=2) + '\n')
    log.info('Done: %s', summary)
    return 0


if __name__ == '__main__':
    sys.exit(main())
