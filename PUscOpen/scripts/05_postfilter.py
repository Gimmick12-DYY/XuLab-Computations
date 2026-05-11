#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 05_postfilter.py
#
# Reconstruct the per-(bin, cell) scOpen output from W @ H, apply two
# CTCF-aware refinements, and persist a per-bin "keep" mask:
#
#   1. Neighborhood-consistency rule. CTCF binding is genomically structured;
#      isolated imputed positives in long signal-poor stretches are usually
#      noise. For each refined-mask bin we count co-located bins (within
#      +/- neighborhood_bp on the same chromosome) whose per-bin imputed
#      total exceeds `neighbor_threshold`. Bins with fewer than
#      `min_neighbors` qualifying neighbors are damped.
#
#   2. (Optional) Multiply per-(bin, cell) values by the bin's PU score so
#      the final score reflects both "imputer thinks this is high signal"
#      AND "prior says this bin is plausibly positive".
#
# This step writes only the *factored* output + a per-bin damping vector;
# the final dense matrix is materialised in 06_impute.py to keep memory
# bounded.
#
# Outputs (under <work>/postfilter/):
#   pu_aligned.tsv            bin_idx_orig, bin_name, pu_score    (refined-mask order)
#   bin_signal.tsv            per-bin scOpen totals, neighborhood counts, damp
#   damping.npy               (R_refined,) float32 vector applied per bin
#   postfilter_meta.json
# -----------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger('05_postfilter')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

_REGION_RE = re.compile(r'^([^:]+):(\d+)-(\d+)$')


def _parse_region(name: str) -> tuple[str, int, int]:
    m = _REGION_RE.match(name)
    if not m:
        raise SystemExit(f'Cannot parse region {name!r} as chr:start-end.')
    return m.group(1), int(m.group(2)), int(m.group(3))


def _neighborhood_counts(regions: list[str], signal: np.ndarray,
                         neighborhood_bp: int, neighbor_threshold: float
                         ) -> np.ndarray:
    """For each region, count how many *other* regions on the same chromosome
    have signal > neighbor_threshold and lie within +/- neighborhood_bp."""
    n = len(regions)
    counts = np.zeros(n, dtype=np.int32)
    # Group by chromosome
    by_chrom: dict[str, list[tuple[int, int, int]]] = {}
    for i, nm in enumerate(regions):
        c, s, _ = _parse_region(nm)
        by_chrom.setdefault(c, []).append((s, i, 0))
    for chrom, entries in by_chrom.items():
        entries.sort()  # by start coord
        starts = np.fromiter((e[0] for e in entries), dtype=np.int64,
                             count=len(entries))
        idx_arr = np.fromiter((e[1] for e in entries), dtype=np.int64,
                              count=len(entries))
        sig_arr = signal[idx_arr]
        hot = sig_arr > neighbor_threshold
        for k, (s, gi, _) in enumerate(entries):
            lo = int(np.searchsorted(starts, s - neighborhood_bp, side='left'))
            hi = int(np.searchsorted(starts, s + neighborhood_bp, side='right'))
            # Count hot bins in [lo, hi), minus self if hot.
            c = int(hot[lo:hi].sum())
            if hot[k]:
                c -= 1
            counts[gi] = max(c, 0)
    return counts


def main() -> int:
    ap = base_parser(__doc__ or '')
    ap.add_argument('--output-prefix', type=str, default='scOpen')
    ap.add_argument('--neighborhood-bp', type=int, default=None)
    ap.add_argument('--min-neighbors', type=int, default=None)
    ap.add_argument('--neighbor-threshold', type=str, default=None,
                    help='Numeric quantile threshold or "auto".')
    ap.add_argument('--no-multiply-by-pu', dest='multiply_by_pu',
                    action='store_false', default=None,
                    help='Override postfilter.multiply_by_pu (default from config).')
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    pf = cfg.setdefault('postfilter', {})
    if args.neighborhood_bp is not None:
        pf['neighborhood_bp'] = args.neighborhood_bp
    if args.min_neighbors is not None:
        pf['min_neighbors'] = args.min_neighbors
    if args.neighbor_threshold is not None:
        pf['neighbor_threshold'] = args.neighbor_threshold
    if args.multiply_by_pu is False:
        pf['multiply_by_pu'] = False

    run_dir = Path(cfg['paths']['scopen_run'])
    pf_dir  = Path(cfg['paths']['postfilter'])
    pu_dir  = Path(cfg['paths']['pu'])

    w_path = run_dir / f'{args.output_prefix}_peaks.txt'
    h_path = run_dir / f'{args.output_prefix}_barcodes.txt'
    if not w_path.exists() or not h_path.exists():
        log.error('Missing factor files (W=%s, H=%s). Run 04_run_scopen.py.',
                  w_path, h_path)
        return 1

    log.info('Reading W from %s', w_path)
    W_df = pd.read_csv(w_path, sep='\t', index_col=0)
    log.info('Reading H from %s', h_path)
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
    log.info('  W=%s, H=%s, regions=%d, cells=%d', W.shape, H.shape,
             len(regions), len(barcodes))

    # Per-bin signal = sum over cells of W @ H = W @ H.sum(axis=1).
    H_sum = H.sum(axis=1).astype(np.float64)
    signal = (W.astype(np.float64) @ H_sum)

    # ------------------------------------------------------------------
    # PU score alignment by bin name.
    # ------------------------------------------------------------------
    pu_scores_path = pu_dir / 'pu_scores.tsv'
    if not pu_scores_path.exists():
        log.error('Missing %s. Run 03_pu_classifier.py first.', pu_scores_path)
        return 1
    pu_df = pd.read_csv(pu_scores_path, sep='\t')
    pu_by_name = dict(zip(pu_df['bin_name'].astype(str), pu_df['pu_score']))
    pu_aligned = np.array([float(pu_by_name.get(r, 0.0)) for r in regions],
                          dtype=np.float64)
    log.info('PU score lookup: %d / %d regions had a hit (others -> 0)',
             int(np.isfinite(pu_aligned).sum() - (pu_aligned == 0.0).sum()) + int((pu_aligned == 0.0).sum()),
             len(regions))

    # ------------------------------------------------------------------
    # Neighborhood-consistency rule
    # ------------------------------------------------------------------
    nb_bp = int(pf.get('neighborhood_bp', 10_000))
    nb_min = int(pf.get('min_neighbors', 1))
    nb_thr_cfg = pf.get('neighbor_threshold', 'auto')
    if nb_thr_cfg in (None, 'auto', 'AUTO'):
        # Use the median signal over positive-seed bins from the PU table
        # as the "this is genuine signal" reference threshold. Falls back to
        # the global median if the seed positives aren't represented.
        if (pu_df['in_train_pos'] == 1).any():
            seed_names = set(pu_df.loc[pu_df['in_train_pos'] == 1, 'bin_name'].astype(str))
            seed_sig = np.array([signal[i] for i, r in enumerate(regions)
                                 if r in seed_names], dtype=np.float64)
            if seed_sig.size:
                nb_thr = float(np.median(seed_sig))
            else:
                nb_thr = float(np.median(signal))
        else:
            nb_thr = float(np.median(signal))
        nb_thr_origin = f'auto:median_seed_signal={nb_thr:.4g}'
    else:
        try:
            nb_thr = float(nb_thr_cfg)
            nb_thr_origin = f'fixed:{nb_thr:.4g}'
        except (TypeError, ValueError):
            nb_thr = float(np.median(signal))
            nb_thr_origin = f'fallback_median={nb_thr:.4g}'

    log.info('Neighborhood rule: +/- %d bp, min_neighbors=%d, threshold=%s',
             nb_bp, nb_min, nb_thr_origin)
    nb_counts = _neighborhood_counts(regions, signal, nb_bp, nb_thr)
    has_support = (nb_counts >= nb_min) | (signal > 4.0 * nb_thr)

    # Damping factor: 1.0 if supported, 0.25 otherwise. A soft penalty keeps
    # downstream metrics meaningful (the bins still appear in the matrix
    # rather than being silently zeroed).
    damp = np.where(has_support, 1.0, 0.25).astype(np.float32)
    n_damped = int((~has_support).sum())
    log.info('Neighborhood post-filter: damped %d / %d bins (%.2f%%)',
             n_damped, len(regions), 100.0 * n_damped / max(len(regions), 1))

    # ------------------------------------------------------------------
    # Optional multiplicative PU prior
    # ------------------------------------------------------------------
    multiply_by_pu = bool(pf.get('multiply_by_pu', True))
    if multiply_by_pu:
        damp = (damp * pu_aligned.astype(np.float32))
        log.info('damp *= pu_score  (mean factor = %.4g, p90 = %.4g)',
                 float(damp.mean()), float(np.quantile(damp, 0.9)))

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    pf_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = pf_dir / 'pu_aligned.tsv'
    pd.DataFrame({
        'bin_idx':  np.arange(len(regions), dtype=np.int64),
        'bin_name': regions,
        'pu_score': pu_aligned,
    }).to_csv(aligned_path, sep='\t', index=False, float_format='%.6g')

    bin_sig_path = pf_dir / 'bin_signal.tsv'
    pd.DataFrame({
        'bin_idx':           np.arange(len(regions), dtype=np.int64),
        'bin_name':          regions,
        'signal_total':      signal,
        'neighbor_count':    nb_counts,
        'has_support':       has_support.astype(np.int8),
        'damp':              damp,
    }).to_csv(bin_sig_path, sep='\t', index=False, float_format='%.6g')

    damp_npy = pf_dir / 'damping.npy'
    np.save(damp_npy, damp.astype(np.float32))

    meta = {
        'n_regions':         int(len(regions)),
        'n_cells':           int(len(barcodes)),
        'rank_k':            int(W.shape[1]),
        'neighborhood_bp':   nb_bp,
        'min_neighbors':     nb_min,
        'neighbor_threshold': {
            'value':  float(nb_thr),
            'origin': nb_thr_origin,
        },
        'n_damped':          n_damped,
        'multiply_by_pu':    multiply_by_pu,
        'pu_mean':           float(pu_aligned.mean()),
        'pu_median':         float(np.median(pu_aligned)),
        'outputs': {
            'pu_aligned':  str(aligned_path),
            'bin_signal':  str(bin_sig_path),
            'damping_npy': str(damp_npy),
        },
        'source': {
            'W': str(w_path),
            'H': str(h_path),
        },
    }
    (pf_dir / 'postfilter_meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    log.info('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
