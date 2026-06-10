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
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger('06_impute')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

_REGION_RE = re.compile(r'^([^:]+):(\d+)-(\d+)$')


def _read_mm_region_lines(mm_dir: Path) -> list[str]:
    reg_gz = mm_dir / 'regions.tsv.gz'
    reg_plain = mm_dir / 'regions.tsv'
    if reg_gz.is_file():
        with gzip.open(reg_gz, 'rt') as fh:
            return [ln.rstrip('\n') for ln in fh if ln.strip()]
    if reg_plain.is_file():
        return [ln for ln in reg_plain.read_text().splitlines() if ln.strip()]
    raise SystemExit(f'Missing {reg_gz} and {reg_plain}; cannot align to mm.')


# -----------------------------------------------------------------------------
# Helpers shared with cisTopic 05_impute.py (kept inline to avoid import
# coupling). See cisTopic/scripts/05_impute.py for the canonical implementation.
# -----------------------------------------------------------------------------
def _parse_regions(regions: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(regions)
    chroms = np.empty(n, dtype=object)
    starts = np.empty(n, dtype=np.int64)
    ends   = np.empty(n, dtype=np.int64)
    for i, r in enumerate(regions):
        m = _REGION_RE.match(r)
        if m is None:
            raise SystemExit(f'Cannot parse region {r!r} as chr:start-end')
        chroms[i] = m.group(1)
        starts[i] = int(m.group(2))
        ends[i]   = int(m.group(3))
    return chroms, starts, ends


def _load_target_bed_mask(bed_path,
                          chroms: np.ndarray, starts: np.ndarray,
                          ends: np.ndarray) -> np.ndarray | None:
    if not bed_path:
        return None
    if isinstance(bed_path, str) and bed_path.strip().lower() == 'off':
        log.info('propagate.target_bed = "off": no BED restriction applied.')
        return None
    p = Path(bed_path)
    if not p.exists():
        log.warning('propagate.target_bed not found: %s; ignoring restriction.', p)
        return None
    log.info('Loading propagate.target_bed from %s', p)
    bed_by_chrom: dict[str, list[tuple[int, int]]] = {}
    opener = gzip.open if str(p).endswith('.gz') else open
    with opener(str(p), 'rt') as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith(('#', 'track', 'browser')):
                continue
            parts = ln.split('\t')
            if len(parts) < 3:
                continue
            try:
                bed_by_chrom.setdefault(parts[0], []).append(
                    (int(parts[1]), int(parts[2]))
                )
            except ValueError:
                continue

    mask = np.zeros(len(chroms), dtype=bool)
    for chrom in np.unique(chroms):
        ivs = bed_by_chrom.get(str(chrom))
        if not ivs:
            continue
        ivs.sort()
        starts_iv = np.fromiter((iv[0] for iv in ivs), dtype=np.int64, count=len(ivs))
        ends_iv   = np.fromiter((iv[1] for iv in ivs), dtype=np.int64, count=len(ivs))
        bin_idx   = np.where(chroms == chrom)[0]
        bs        = starts[bin_idx]
        be        = ends[bin_idx]
        hi_arr    = np.searchsorted(starts_iv, be, side='left')
        for k, (b_s, _, hi) in enumerate(zip(bs, be, hi_arr)):
            if hi == 0:
                continue
            if (ends_iv[:hi] > b_s).any():
                mask[bin_idx[k]] = True
    return mask


def _parse_threshold_mode(s: str) -> tuple[str, float]:
    s = (s or 'sparsity_match:1').strip()
    if s == 'sparsity_match':
        return 'sparsity_match', 1.0
    if s.startswith('sparsity_match:'):
        return 'sparsity_match', float(s.split(':', 1)[1])
    if s.startswith('quantile:'):
        return 'quantile', float(s.split(':', 1)[1])
    if s.startswith('fixed:'):
        return 'fixed', float(s.split(':', 1)[1])
    raise SystemExit(
        f'Unknown source_threshold_mode {s!r}; '
        f'expected sparsity_match[:K] | quantile:Q | fixed:T'
    )


def _sample_imp_values(W: np.ndarray, H: np.ndarray, damp: np.ndarray,
                       rescale: float, n_sample: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Sample n_sample imputed values uniformly from (modeled_row, cell) pairs."""
    R, _ = W.shape
    _, C = H.shape
    n_sample = int(min(int(n_sample), int(R) * int(C)))
    if n_sample <= 0:
        return np.zeros(0, dtype=np.float64)
    rows = rng.integers(0, R, size=n_sample, dtype=np.int64)
    cols = rng.integers(0, C, size=n_sample, dtype=np.int64)
    base = np.einsum('ik,ki->i',
                     W[rows].astype(np.float64),
                     H[:, cols].astype(np.float64))
    return base * damp[rows].astype(np.float64) * float(rescale)


def _pick_source_threshold(W: np.ndarray, H: np.ndarray, damp: np.ndarray,
                            rescale: float, mode: str, param: float,
                            target_nnz_modeled: int,
                            sample_size: int, rng: np.random.Generator
                            ) -> tuple[float, str]:
    if mode == 'fixed':
        return float(param), f'fixed:{float(param):.6g}'
    sample = _sample_imp_values(W, H, damp, rescale, sample_size, rng)
    if sample.size == 0:
        return 0.0, 'empty_sample'
    Rm, C = W.shape[0], H.shape[1]
    total_entries = int(Rm) * int(C)
    if mode == 'sparsity_match':
        K = float(param)
        target = max(1, int(round(target_nnz_modeled * K)))
        target = max(1, min(target, total_entries))
        target_fraction = target / total_entries
        q = max(0.0, min(1.0, 1.0 - target_fraction))
        t = float(np.quantile(sample, q))
        return t, (f'sparsity_match:{K} (target_calls={target}, '
                   f'q={q:.6g}, sample={sample.size})')
    if mode == 'quantile':
        q = float(param)
        t = float(np.quantile(sample, q))
        return t, f'quantile:{q} -> t={t:.4g}'
    raise SystemExit(f'Unknown mode {mode!r}')


def _propagate_to_zero_bins(csr_in: sparse.csr_matrix,
                            source_mask: sparse.csr_matrix,
                            regions: list[str],
                            propagate_cfg: dict
                            ) -> tuple[sparse.csr_matrix, dict]:
    """Add new (target_row, cell) entries to csr_in at currently-zero rows.

    ``source_mask`` is a sparse boolean matrix marking which (modeled_row,
    cell) entries qualify as ``imp-positive`` sources for propagation; we
    pull the corresponding values from csr_in.
    """
    if not propagate_cfg.get('enabled', False):
        return csr_in, {'enabled': False, 'n_propagated_bins': 0,
                        'n_new_entries': 0}

    window_bp     = int(propagate_cfg.get('window_bp', 5000))
    min_neighbors = int(propagate_cfg.get('min_neighbors', 2))
    weight        = float(propagate_cfg.get('weight', 0.5))
    chunk_targets = int(propagate_cfg.get('chunk_targets', 5000))
    target_bed    = propagate_cfg.get('target_bed', None)

    log.info(
        'Neighbour propagation: window_bp=%d, min_neighbors=%d, weight=%.4g, '
        'target_bed=%s, chunk_targets=%d',
        window_bp, min_neighbors, weight, target_bed, chunk_targets,
    )

    chroms, starts, ends = _parse_regions(regions)
    n_total_rows = int(csr_in.shape[0])

    has_signal      = np.diff(csr_in.indptr) > 0
    target_eligible = ~has_signal

    if target_bed:
        bed_mask = _load_target_bed_mask(target_bed, chroms, starts, ends)
        if bed_mask is not None:
            n_pre = int(target_eligible.sum())
            target_eligible &= bed_mask
            log.info(
                'Target candidates restricted by BED: %d zero-imp -> %d eligible',
                n_pre, int(target_eligible.sum()),
            )

    n_target_total = int(target_eligible.sum())
    n_source_total = int(has_signal.sum())
    log.info('Propagation candidates: %d eligible target rows, %d source rows',
             n_target_total, n_source_total)
    if n_target_total == 0 or n_source_total < min_neighbors:
        log.info('Nothing to propagate.')
        return csr_in, {
            'enabled': True, 'n_propagated_bins': 0, 'n_new_entries': 0,
            'n_target_total': n_target_total, 'n_source_total': n_source_total,
            'window_bp': window_bp, 'min_neighbors': min_neighbors,
            'weight': weight,
            'target_bed': str(target_bed) if target_bed else None,
        }

    new_rows: list[np.ndarray] = []
    new_cols: list[np.ndarray] = []
    new_vals: list[np.ndarray] = []
    n_propagated_bins = 0

    for chrom in np.unique(chroms):
        in_chrom      = (chroms == chrom)
        chrom_targets = np.where(in_chrom & target_eligible)[0]
        chrom_sources = np.where(in_chrom & has_signal)[0]
        if chrom_targets.size == 0 or chrom_sources.size < min_neighbors:
            continue

        src_starts        = starts[chrom_sources]
        src_order         = np.argsort(src_starts, kind='stable')
        src_starts_sorted = src_starts[src_order]
        sources_sorted    = chrom_sources[src_order]

        # Source values restricted to imp-positive entries: vals * mask.
        # Per (target, cell): count = sum_k_in_window mask[k, c]
        #                     sum   = sum_k_in_window  vals[k, c] * mask[k, c]
        # mean = sum / count when count >= min_neighbors.
        src_vals_dense   = csr_in[sources_sorted].toarray().astype(np.float32, copy=False)
        src_mask_dense   = source_mask[sources_sorted].toarray().astype(np.float32, copy=False)
        src_vals_masked  = src_vals_dense * src_mask_dense

        for ts in range(0, chrom_targets.size, chunk_targets):
            te = min(ts + chunk_targets, chrom_targets.size)
            chunk_target_rows   = chrom_targets[ts:te]
            chunk_target_starts = starts[chunk_target_rows]
            n_chunk = te - ts

            edge_t: list[int] = []
            edge_s: list[int] = []
            for i, t_start in enumerate(chunk_target_starts):
                lo = int(np.searchsorted(src_starts_sorted,
                                          int(t_start) - window_bp, side='left'))
                hi = int(np.searchsorted(src_starts_sorted,
                                          int(t_start) + window_bp, side='right'))
                n_n = hi - lo
                if n_n < min_neighbors:
                    continue
                edge_t.extend([i] * n_n)
                edge_s.extend(range(lo, hi))

            if not edge_t:
                continue

            edge_t_arr = np.asarray(edge_t, dtype=np.int64)
            edge_s_arr = np.asarray(edge_s, dtype=np.int64)
            ones_arr   = np.ones(len(edge_t), dtype=np.float32)

            # Indicator matrix: P_i[i, k] = 1 if source k is in target i's window.
            P_i = sparse.coo_matrix(
                (ones_arr, (edge_t_arr, edge_s_arr)),
                shape=(n_chunk, sources_sorted.size), dtype=np.float32,
            ).tocsr()

            # Per (target, cell) sum of imp-positive source values, and count
            # of imp-positive sources actually in window.
            prop_sum   = (P_i @ src_vals_masked).astype(np.float32, copy=False)
            prop_count = (P_i @ src_mask_dense).astype(np.float32, copy=False)
            keep = prop_count >= float(min_neighbors)

            with np.errstate(divide='ignore', invalid='ignore'):
                prop_mean = np.where(
                    keep,
                    prop_sum / np.maximum(prop_count, 1.0),
                    0.0,
                )
            propagated = (prop_mean.astype(np.float32, copy=False)
                          * np.float32(weight))

            for ti in range(n_chunk):
                vals = propagated[ti]
                nz = np.where(vals > 0.0)[0]
                if nz.size == 0:
                    continue
                target_row_full = int(chunk_target_rows[ti])
                new_rows.append(np.full(nz.size, target_row_full, dtype=np.int64))
                new_cols.append(nz.astype(np.int64))
                new_vals.append(vals[nz].astype(np.float32, copy=False))
                n_propagated_bins += 1

        del src_vals_dense, src_mask_dense, src_vals_masked

    if not new_rows:
        log.info('Propagation produced 0 new entries.')
        return csr_in, {
            'enabled': True, 'n_propagated_bins': 0, 'n_new_entries': 0,
            'n_target_total': n_target_total, 'n_source_total': n_source_total,
            'window_bp': window_bp, 'min_neighbors': min_neighbors,
            'weight': weight,
            'target_bed': str(target_bed) if target_bed else None,
        }

    new_rows_arr = np.concatenate(new_rows)
    new_cols_arr = np.concatenate(new_cols)
    new_vals_arr = np.concatenate(new_vals)
    del new_rows, new_cols, new_vals
    n_new_entries = int(new_vals_arr.size)
    mean_new_per_bin = float(n_new_entries / max(n_propagated_bins, 1))
    log.info('Propagation: added %d new (region, cell) entries across %d new bins',
             n_new_entries, n_propagated_bins)

    # The new entries live in rows that were EMPTY in csr_in (target_eligible =
    # ~has_signal), so they are row-disjoint from csr_in. Build them as a small
    # separate CSR and add, instead of round-tripping the (potentially huge)
    # csr_in through COO -- csr_in.tocoo() alone allocates an int64 row array of
    # size csr_in.nnz (tens of GB for a dense modeled subspace) and the
    # concatenate+rebuild makes 3-4 full copies, which is what OOMs.
    csr_new = sparse.csr_matrix(
        (new_vals_arr, (new_rows_arr, new_cols_arr)),
        shape=csr_in.shape, dtype=np.float32,
    )
    del new_rows_arr, new_cols_arr, new_vals_arr
    csr_out = (csr_in + csr_new).tocsr()
    csr_out.sum_duplicates()

    return csr_out, {
        'enabled': True,
        'n_propagated_bins': n_propagated_bins,
        'n_new_entries': n_new_entries,
        'n_target_total': n_target_total,
        'n_source_total': n_source_total,
        'mean_new_entries_per_propagated_bin': mean_new_per_bin,
        'window_bp': window_bp, 'min_neighbors': min_neighbors,
        'weight': weight,
        'target_bed': str(target_bed) if target_bed else None,
    }


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

        csr_imp_only = sparse.csr_matrix(
            (data, indices, indptr), shape=(N, C), dtype=np.float32,
        )
        csr_imp_only.check_format(full_check=False)

        # ----- Neighbour propagation -----
        propagate_cfg = imp_cfg.get('propagate', {}) or {}
        propagation_stats: dict = {'enabled': False,
                                   'n_propagated_bins': 0,
                                   'n_new_entries': 0}
        source_threshold_value = float('nan')
        source_threshold_origin = 'disabled'
        nnz_source_mask = 0
        csr = csr_imp_only

        if propagate_cfg.get('enabled', False):
            # Build the imp-positive source mask via a sample-based threshold.
            sample_size = int(propagate_cfg.get(
                'source_threshold_sample_size',
                imp_cfg.get('source_threshold_sample_size', 4_000_000),
            ))
            mode_str = str(propagate_cfg.get(
                'source_threshold_mode',
                imp_cfg.get('source_threshold_mode', 'sparsity_match:1'),
            ))
            mode, mode_param = _parse_threshold_mode(mode_str)

            # raw_nnz on the modeled subspace (rows in mm corresponding to W rows)
            target_nnz_modeled = 0
            if mode == 'sparsity_match':
                mm_mtx = mm_dir / 'matrix.mtx.gz'
                if not mm_mtx.exists():
                    log.warning(
                        'Cannot find %s for sparsity_match source threshold; '
                        'falling back to quantile:0.95.', mm_mtx,
                    )
                    mode, mode_param = 'quantile', 0.95
                else:
                    import scipy.io as sio  # local import to avoid cost when unused
                    raw_cnt = sio.mmread(str(mm_mtx)).tocsr()
                    raw_bin = (raw_cnt > 0).astype(np.int8)
                    sub = raw_bin[full_rows]   # only modeled rows
                    target_nnz_modeled = int(sub.nnz)
                    log.info('Raw nonzeros on modeled subspace: %d',
                             target_nnz_modeled)
                    del raw_bin, sub, raw_cnt

            rng = np.random.default_rng(42)
            source_threshold_value, source_threshold_origin = _pick_source_threshold(
                W, H, damp, rescale_factor if rescaled_flag else 1.0,
                mode, mode_param, target_nnz_modeled,
                sample_size, rng,
            )
            log.info('Source threshold (imp-positive cutoff) = %.6g (%s)',
                     source_threshold_value, source_threshold_origin)

            # Build source mask: indicator over the modeled-only CSR of values
            # above the source threshold. Sparse to save memory.
            mask_data = (csr_imp_only.data > source_threshold_value).astype(np.float32)
            source_mask = sparse.csr_matrix(
                (mask_data, csr_imp_only.indices.copy(), csr_imp_only.indptr.copy()),
                shape=csr_imp_only.shape, dtype=np.float32,
            )
            source_mask.eliminate_zeros()
            nnz_source_mask = int(source_mask.nnz)
            log.info('Imp-positive source entries: %d (%.3g%% of modeled cells)',
                     nnz_source_mask,
                     100.0 * nnz_source_mask / max(int(Rm) * int(C), 1))

            # Default the propagation target BED. Priority order:
            #   1. propagate.target_bed = "off" (string) -> no restriction.
            #      This is the bias-free / data-only setting; matches the
            #      framing used by scBasset's Track A.
            #   2. propagate.target_bed = null -> auto-detect:
            #        a) <repo-root>/downstream/cache/CTCF_motif_hg38.bed
            #           (motif-aware -- bulk prior; opt-in only)
            #        b) <work>/mask/mask.bed (geometric mask from step 02)
            # Auto-detect ("null") used to be the default. It introduces a
            # bin-universe leak relative to evaluations whose negatives are
            # also defined by CTCF cCRE membership: PUscOpen's prediction
            # space and the negative bins become near-disjoint by construction
            # and the imputer scores artifactually high. "off" is now the
            # safe default in configs/default.yaml.
            raw_target_bed = propagate_cfg.get('target_bed', None)
            if isinstance(raw_target_bed, str) and raw_target_bed.strip().lower() == 'off':
                propagate_cfg = dict(propagate_cfg)
                propagate_cfg['target_bed'] = None
                log.info('propagate.target_bed = "off": no BED restriction applied (data-only).')
            elif raw_target_bed is None:
                propagate_cfg = dict(propagate_cfg)
                wd_path = Path(cfg['paths']['work_dir']).resolve()
                motif_bed: Path | None = None
                for anc in [wd_path, *wd_path.parents]:
                    cand = anc / 'downstream' / 'cache' / 'CTCF_motif_hg38.bed'
                    if cand.is_file():
                        motif_bed = cand
                        break
                if motif_bed is not None:
                    propagate_cfg['target_bed'] = str(motif_bed)
                    log.info('propagate.target_bed auto-detected (motif) -> %s',
                             motif_bed)
                else:
                    fallback_bed = Path(cfg['paths']['mask']) / 'mask.bed'
                    if fallback_bed.exists():
                        propagate_cfg['target_bed'] = str(fallback_bed)
                        log.info(
                            'propagate.target_bed defaulted (geometric mask) -> %s',
                            fallback_bed,
                        )

            csr, propagation_stats = _propagate_to_zero_bins(
                csr_imp_only, source_mask, full_regions, propagate_cfg,
            )
            propagation_stats.update({
                'source_threshold_value':  float(source_threshold_value),
                'source_threshold_origin': source_threshold_origin,
                'nnz_source_mask':         int(nnz_source_mask),
            })

        # ----- Persist -----
        csr_path          = imp_dir / 'matrix_csr.npz'
        csr_imp_only_path = imp_dir / 'matrix_csr_imputed_only.npz'

        log.info('Wrote %s (shape=%s, nnz=%d) -- imp only, pre-propagation',
                 csr_imp_only_path, csr_imp_only.shape, csr_imp_only.nnz)
        sparse.save_npz(csr_imp_only_path, csr_imp_only)
        # Free the (potentially tens-of-GB) pre-propagation matrix before
        # writing the propagated one, so both are not resident simultaneously.
        if csr is not csr_imp_only:
            del csr_imp_only

        log.info('Wrote %s (shape=%s, nnz=%d) -- after propagation',
                 csr_path, csr.shape, csr.nnz)
        sparse.save_npz(csr_path, csr)

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
            'pipeline':                 'PUscOpen',
            'format':                   'csr_full_mm',
            'matrix_path':              str(csr_path),
            'matrix_imputed_only_path': str(csr_imp_only_path),
            'regions_path':             str(regions_out),
            'barcodes_path':            str(barcodes_out),
            'shape':                    [int(N), int(C)],
            'nnz':                      int(csr.nnz),
            'nnz_imputed_only':         int(csr_imp_only.nnz),
            'n_modeled_rows':           int(Rm),
            'n_cells':                  int(C),
            'rank_k':                   int(W.shape[1]),
            'scale_factor':             desired_scale,
            'rescale_factor':           float(rescale_factor),
            'rescaled':                 rescaled_flag,
            'align_to_mm':              True,
            'source_w':                 str(w_path),
            'source_h':                 str(h_path),
            'damping':                  str(damp_path),
            'propagation':              propagation_stats,
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
