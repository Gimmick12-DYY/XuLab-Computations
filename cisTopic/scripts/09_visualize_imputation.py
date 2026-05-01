#!/usr/bin/env python
from __future__ import annotations

import gzip
import json
import logging
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.cluster.hierarchy import leaves_list, linkage

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger('09_viz')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def read_lines_gz(path: Path) -> list[str]:
    with gzip.open(path, 'rt') as fh:
        return [x.rstrip('\n') for x in fh]


def qstats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64).ravel()
    return {
        'mean': float(x.mean()), 'median': float(np.median(x)), 'std': float(x.std()),
        'min': float(x.min()), 'p10': float(np.percentile(x, 10)), 'p90': float(np.percentile(x, 90)),
        'p99': float(np.percentile(x, 99)), 'max': float(x.max())
    }


def cluster_order(mat: np.ndarray, axis: int) -> np.ndarray:
    data = mat if axis == 0 else mat.T
    if data.shape[0] <= 1:
        return np.arange(data.shape[0])
    z = linkage(data, method='average', metric='correlation', optimal_ordering=False)
    return leaves_list(z)


def main() -> int:
    ap = base_parser(__doc__ or '')
    ap.add_argument('--imputed-h5', type=str, default=None)
    ap.add_argument('--mm-dir', type=str, default=None)
    ap.add_argument('--n-regions', type=int, default=1000)
    ap.add_argument('--n-cells', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-name', type=str, default='visualize_imputation')
    ap.add_argument('--space', type=str, default='full', choices=['full', 'filtered'],
                    help='full: compare in original 3,031,053-row universe (missing imputed rows=0).')
    ap.add_argument('--region-mode', type=str, default='original_top_cells',
                    choices=['original_top_cells', 'original_signal', 'imputed_variable', 'random'])
    ap.add_argument('--no-cluster', action='store_true')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = resolve_paths(load_config(args.config), args.work_dir)

    mm_dir = Path(args.mm_dir) if args.mm_dir else Path(cfg['paths']['mm'])
    mtx_path = mm_dir / 'matrix.mtx.gz'
    reg_path = mm_dir / 'regions.tsv.gz'
    bar_path = mm_dir / 'barcodes.tsv.gz'

    orig = sio.mmread(mtx_path).tocsr()  # regions x cells
    orig_regs = read_lines_gz(reg_path)
    orig_bars = read_lines_gz(bar_path)
    log.info('Original: %d x %d, nnz=%d density=%.3g', *orig.shape, orig.nnz, orig.nnz / (orig.shape[0] * orig.shape[1]))

    h5_path = Path(args.imputed_h5) if args.imputed_h5 else Path(cfg['paths']['impute']) / 'imputed_Prc_hdf5_all.h5'
    with h5py.File(h5_path, 'r') as h5:
        imp_regs = [x.decode() if isinstance(x, bytes) else str(x) for x in h5['regions'][:]]
        imp_bars = [x.decode() if isinstance(x, bytes) else str(x) for x in h5['barcodes'][:]]
        Prc_raw = h5['Prc'][...].astype(np.float32, copy=False)
        sf_attr = float(h5.attrs.get('scale_factor', 0.0)) or 0.0

    # Normalize to probability so sum_r P(r|c)=1
    mean_sum = float(Prc_raw.sum(axis=0).mean())
    scale_in_h5 = mean_sum if mean_sum > 0 else 1.0
    Prc = Prc_raw / scale_in_h5

    reg_to_orig = {r: i for i, r in enumerate(orig_regs)}
    bar_to_orig = {b: i for i, b in enumerate(orig_bars)}
    row_idx = np.array([reg_to_orig[r] for r in imp_regs], dtype=np.int64)
    col_idx = np.array([bar_to_orig[b] for b in imp_bars], dtype=np.int64)

    R0, C0 = orig.shape
    Rf, Cf = Prc.shape
    log.info('Imputed: %d x %d; covers %.2f%% regions, %.2f%% cells', Rf, Cf, 100*Rf/R0, 100*Cf/C0)

    # Fair-space density calculations
    orig_bin = (orig > 0).astype(np.int8)
    orig_density_full = float(orig.nnz) / float(R0 * C0)
    imp_density_full_gt0 = float(Rf * Cf) / float(R0 * C0)  # missing rows treated as 0
    modeled_mask = np.zeros(R0, dtype=bool)
    modeled_mask[row_idx] = True

    # Choose cells
    cell_pick_imp = rng.choice(Cf, size=min(args.n_cells, Cf), replace=False)
    orig_cols = col_idx[cell_pick_imp]

    # Choose region universe + selection score
    if args.space == 'full':
        # full original space; missing imputed rows => 0
        if args.region_mode == 'original_top_cells':
            row_score = np.diff(orig.indptr)  # nonzero cells per region in original
            region_pick_orig = np.argsort(row_score)[::-1][:min(args.n_regions, R0)]
        elif args.region_mode == 'original_signal':
            score = np.asarray(orig.sum(axis=1)).ravel()  # counts per region
            region_pick_orig = np.argsort(score)[::-1][:min(args.n_regions, R0)]
        elif args.region_mode == 'imputed_variable':
            v = Prc.var(axis=1)
            top_imp = np.argsort(v)[::-1][:min(args.n_regions, Rf)]
            region_pick_orig = row_idx[top_imp]
        else:
            region_pick_orig = rng.choice(R0, size=min(args.n_regions, R0), replace=False)

        # Build imputed submatrix in FULL space by lookup; rows missing in imputed get 0.
        lookup = -np.ones(R0, dtype=np.int64)
        lookup[row_idx] = np.arange(Rf)
        imp_rows = lookup[region_pick_orig]

        Prc_sub = np.zeros((region_pick_orig.size, cell_pick_imp.size), dtype=np.float32)
        ok = imp_rows >= 0
        if ok.any():
            Prc_sub[ok] = Prc[imp_rows[ok]][:, cell_pick_imp]

        orig_count_sub = orig[region_pick_orig][:, orig_cols].toarray().astype(np.float32)
        orig_bin_sub = (orig_count_sub > 0).astype(np.int8)

    else:
        # filtered space only
        if args.region_mode == 'original_top_cells':
            region_pick_imp = np.argsort(np.diff(orig[row_idx].indptr))[::-1][:min(args.n_regions, Prc_prob.shape[0])]
        elif args.region_mode == 'imputed_variable':
            score = Prc.var(axis=1)
            region_pick_imp = np.argsort(score)[::-1][:min(args.n_regions, Rf)]
        elif args.region_mode == 'original_signal':
            # project original signal to filtered rows
            score = np.asarray(orig[row_idx].sum(axis=1)).ravel()
            region_pick_imp = np.argsort(score)[::-1][:min(args.n_regions, Rf)]
        else:
            region_pick_imp = rng.choice(Rf, size=min(args.n_regions, Rf), replace=False)

        Prc_sub = Prc[region_pick_imp][:, cell_pick_imp]
        orig_count_sub = orig[row_idx[region_pick_imp]][:, orig_cols].toarray().astype(np.float32)
        orig_bin_sub = (orig_count_sub > 0).astype(np.int8)

    # Cluster on imputed and apply same ordering
    if not args.no_cluster:
        ro = cluster_order(Prc_sub, axis=0)
        co = cluster_order(Prc_sub, axis=1)
    else:
        ro = np.arange(Prc_sub.shape[0]); co = np.arange(Prc_sub.shape[1])

    Prc_sub = Prc_sub[ro][:, co]
    orig_count_sub = orig_count_sub[ro][:, co]
    orig_bin_sub = orig_bin_sub[ro][:, co]

    d = float(orig_bin_sub.mean())
    vmax_imp = float(np.quantile(Prc_sub, 0.99))
    if vmax_imp <= 0:
        vmax_imp = float(Prc_sub.max() or 1.0)

    # Count-space expected counts (fair units)
    fragments = np.asarray(orig.sum(axis=0)).ravel()[orig_cols][co]
    imp_expected_sub = Prc_sub * fragments[None, :]

    # Plots
    eval_dir = Path(cfg['paths']['work_dir']) / 'eval'
    eval_dir.mkdir(parents=True, exist_ok=True)
    heat_png = eval_dir / f'{args.out_name}_heatmap.png'
    cnt_png = eval_dir / f'{args.out_name}_counts.png'
    cell_png = eval_dir / f'{args.out_name}_per_cell.png'

    fig, ax = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    ax[0].imshow(orig_bin_sub, aspect='auto', cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
    ax[0].set_title(f'Original binary\n{orig_bin_sub.shape[0]}x{orig_bin_sub.shape[1]} density={d:.4g}')
    ax[1].imshow(Prc_sub, aspect='auto', cmap='Reds', vmin=0, vmax=vmax_imp, interpolation='nearest')
    ax[1].set_title(f'Imputed P(r|c) (no threshold)\nvmax=q99={vmax_imp:.3g}')
    for a in ax:
        a.set_xlabel('cells')
    ax[0].set_ylabel('regions')
    fig.tight_layout(); fig.savefig(heat_png, dpi=160); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    vmax = max(1.0, float(np.quantile(np.concatenate([orig_count_sub.ravel(), imp_expected_sub.ravel()]), 0.999)))
    ax[0].imshow(orig_count_sub, aspect='auto', cmap='Reds', vmin=0, vmax=vmax, interpolation='nearest')
    ax[0].set_title(f'Original counts (shared vmax={vmax:.3g})')
    ax[1].imshow(imp_expected_sub, aspect='auto', cmap='Reds', vmin=0, vmax=vmax, interpolation='nearest')
    ax[1].set_title('Imputed expected counts')
    for a in ax:
        a.set_xlabel('cells')
    ax[0].set_ylabel('regions')
    fig.tight_layout(); fig.savefig(cnt_png, dpi=160); plt.close(fig)

    counts_per_cell = np.asarray(orig.sum(axis=0)).ravel()
    bin_per_cell = np.diff(orig.tocsc().indptr)
    imp_prob_sum = Prc.sum(axis=0)
    imp_expected_sum = imp_prob_sum * counts_per_cell[col_idx]

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    ax[0,0].hist(counts_per_cell, bins=80); ax[0,0].set_title('Original fragments/cell')
    ax[0,1].hist(bin_per_cell, bins=80); ax[0,1].set_title('Original accessible regions/cell')
    ax[1,0].hist(imp_prob_sum, bins=80); ax[1,0].set_title('Imputed sum P(r|c) per cell')
    ax[1,1].scatter(counts_per_cell[col_idx], imp_expected_sum, s=4, alpha=0.5)
    lim=max(counts_per_cell[col_idx].max(), imp_expected_sum.max())*1.05
    ax[1,1].plot([0,lim],[0,lim],'k--',lw=0.7); ax[1,1].set_title('Original vs imputed expected/cell')
    fig.tight_layout(); fig.savefig(cell_png, dpi=150); plt.close(fig)

    # Extra fairness figure A: strict modeled-space apples-to-apples.
    modeled_png = eval_dir / f'{args.out_name}_modeled_fair.png'
    n_mod = min(args.n_regions, Rf)
    modeled_score = np.diff(orig[row_idx].tocsr().indptr)
    modeled_pick_imp = np.argsort(modeled_score)[::-1][:n_mod]
    Prc_modeled = Prc[modeled_pick_imp][:, cell_pick_imp]
    orig_modeled = orig[row_idx[modeled_pick_imp]][:, orig_cols].toarray().astype(np.float32)
    orig_modeled_bin = (orig_modeled > 0).astype(np.int8)
    if not args.no_cluster:
        ro_m = cluster_order(Prc_modeled, axis=0)
        co_m = cluster_order(Prc_modeled, axis=1)
    else:
        ro_m = np.arange(Prc_modeled.shape[0]); co_m = np.arange(Prc_modeled.shape[1])
    Prc_modeled = Prc_modeled[ro_m][:, co_m]
    orig_modeled_bin = orig_modeled_bin[ro_m][:, co_m]
    d_m = float(orig_modeled_bin.mean())
    vmax_m = float(np.quantile(Prc_modeled, 0.99))
    if vmax_m <= 0:
        vmax_m = float(Prc_modeled.max() or 1.0)
    fig, ax = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    ax[0].imshow(orig_modeled_bin, aspect='auto', cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
    ax[0].set_title(f'Modeled-space original\\ndensity={d_m:.4g}')
    ax[1].imshow(Prc_modeled, aspect='auto', cmap='Reds', vmin=0, vmax=vmax_m, interpolation='nearest')
    ax[1].set_title(f'Modeled-space imputed (no threshold)\\nvmax=q99={vmax_m:.3g}')
    for a in ax: a.set_xlabel('cells')
    ax[0].set_ylabel('modeled regions')
    fig.tight_layout(); fig.savefig(modeled_png, dpi=160); plt.close(fig)

    # Extra fairness figure B: full-space stratified (modeled + unmodeled rows).
    # This makes the "rows not modeled by LDA => imputed 0" effect explicit.
    strat_png = eval_dir / f'{args.out_name}_full_stratified.png'
    n_each = max(1, min(args.n_regions // 2, modeled_mask.sum(), (~modeled_mask).sum()))
    row_signal = np.asarray(orig.sum(axis=1)).ravel()
    modeled_rows = np.where(modeled_mask)[0]
    unmodeled_rows = np.where(~modeled_mask)[0]
    mod_pick = modeled_rows[np.argsort(row_signal[modeled_rows])[::-1][:n_each]]
    unmod_pick = unmodeled_rows[np.argsort(row_signal[unmodeled_rows])[::-1][:n_each]]
    strat_rows = np.concatenate([mod_pick, unmod_pick])
    row_lookup = -np.ones(R0, dtype=np.int64); row_lookup[row_idx] = np.arange(Rf)
    ir = row_lookup[strat_rows]
    Prc_strat = np.zeros((strat_rows.size, cell_pick_imp.size), dtype=np.float32)
    ok = ir >= 0
    if ok.any():
        Prc_strat[ok] = Prc[ir[ok]][:, cell_pick_imp]
    orig_strat = orig[strat_rows][:, orig_cols].toarray().astype(np.float32)
    orig_strat_bin = (orig_strat > 0).astype(np.int8)
    # Keep row blocks explicit; cluster columns only.
    if not args.no_cluster:
        co_s = cluster_order(Prc_strat[:n_each], axis=1)  # cluster by modeled block only
    else:
        co_s = np.arange(Prc_strat.shape[1])
    Prc_strat = Prc_strat[:, co_s]
    orig_strat_bin = orig_strat_bin[:, co_s]
    d_s = float(orig_strat_bin.mean())
    vmax_s = float(np.quantile(Prc_strat, 0.99))
    if vmax_s <= 0:
        vmax_s = float(Prc_strat.max() or 1.0)
    fig, ax = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    ax[0].imshow(orig_strat_bin, aspect='auto', cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
    ax[0].axhline(n_each, color='black', lw=1)
    ax[0].set_title('Full-space original (top: modeled, bottom: unmodeled)')
    ax[1].imshow(Prc_strat, aspect='auto', cmap='Reds', vmin=0, vmax=vmax_s, interpolation='nearest')
    ax[1].axhline(n_each, color='black', lw=1)
    ax[1].set_title(f'Full-space imputed P(r|c) (no threshold)\\nvmax=q99={vmax_s:.3g}')
    for a in ax: a.set_xlabel('cells')
    ax[0].set_ylabel('regions (stratified)')
    fig.tight_layout(); fig.savefig(strat_png, dpi=160); plt.close(fig)

    # per-cell TSV
    tsv = eval_dir / f'{args.out_name}_per_cell.tsv'
    with open(tsv, 'w') as fh:
        fh.write('cell\torig_fragments\torig_accessible_regions\timp_sum_prob\timp_expected_sum\n')
        for j, b in enumerate(imp_bars):
            jo = bar_to_orig[b]
            fh.write(f'{b}\t{int(counts_per_cell[jo])}\t{int(bin_per_cell[jo])}\t{float(imp_prob_sum[j]):.6f}\t{float(imp_expected_sum[j]):.6f}\n')

    out_json = eval_dir / f'{args.out_name}.json'
    summary = {
        'space': args.space,
        'region_mode': args.region_mode,
        'original_shape': [int(R0), int(C0)],
        'imputed_shape_filtered': [int(Rf), int(Cf)],
        'coverage_regions_fraction': float(Rf / R0),
        'original_nnz': int(orig.nnz),
        'original_density_full': orig_density_full,
        'imputed_density_full_gt0': imp_density_full_gt0,
        'scale_factor_attr': sf_attr,
        'scale_in_h5_used': scale_in_h5,
        'imputed_prob_sum_per_cell': qstats(imp_prob_sum),
        'original_fragments_per_cell': qstats(counts_per_cell),
        'imputed_expected_sum_per_cell': qstats(imp_expected_sum),
        'subsample': {
            'shape': [int(orig_bin_sub.shape[0]), int(orig_bin_sub.shape[1])],
            'orig_density': float(orig_bin_sub.mean()),
            'imputed_vmax_q99': float(vmax_imp),
        },
        'modeled_space_subsample': {
            'n_rows': int(n_mod),
            'density_original': float(d_m),
            'imputed_vmax_q99': float(vmax_m),
        },
        'full_space_stratified_subsample': {
            'n_modeled_rows': int(n_each),
            'n_unmodeled_rows': int(n_each),
            'density_original': float(d_s),
            'imputed_vmax_q99': float(vmax_s),
        },
    }
    out_json.write_text(json.dumps(summary, indent=2) + '\n')

    log.info('Wrote %s', heat_png)
    log.info('Wrote %s', cnt_png)
    log.info('Wrote %s', cell_png)
    log.info('Wrote %s', modeled_png)
    log.info('Wrote %s', strat_png)
    log.info('Wrote %s', tsv)
    log.info('Wrote %s', out_json)
    log.info('FULL-SPACE fairness: original density=%.3g, imputed>0 density in full space=%.3g',
             orig_density_full, imp_density_full_gt0)
    return 0


if __name__ == '__main__':
    sys.exit(main())
