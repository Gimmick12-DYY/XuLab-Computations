#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 07_eval_heldout.py
#
# Evaluate FITS-imputed P(r|c) (read from imputed_Prc_hdf5_all.h5) against
# baselines on a balanced positive/negative pair set sampled from the original
# observed matrix. Two modes:
#
#   * default ("reconstruction"):
#         Sample positives from observed nonzero entries in the (binarised)
#         filtered matrix and an equal number of negatives from zero entries.
#         Score baselines + FITS on those pairs and report AUROC / AUPRC. This
#         is a goodness-of-fit diagnostic on the training data — not a strict
#         generalization test, since FITS was built on these entries.
#
#   * strict held-out (two steps):
#         1) Run with --prepare-holdout. Samples a balanced pos/neg split,
#            zeros out the positives in the filtered training matrix, and writes
#            mm_holdout/ + eval/holdout_split.npz.
#         2) Re-run steps 02-05 against a config whose paths.mm points at that
#            mm_holdout/ folder (separate work_dir to avoid clobbering). Then:
#               python 07_eval_heldout.py --config <cfg> \
#                 --holdout-split <orig_work>/eval/holdout_split.npz \
#                 --imputed-h5    <holdout_work>/impute/imputed_Prc_hdf5_all.h5
#
# Restriction: heldout sampling is by default restricted to FITS-modeled rows
# (i.e. positives that FITS could in principle predict). Pass --sample-space
# full to ignore that restriction and force unmodeled rows to predicted-zero.
# -----------------------------------------------------------------------------
from __future__ import annotations

import gzip
import json
import logging
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from _cfg import base_parser, load_config, resolve_paths  # noqa: E402

log = logging.getLogger('07_eval')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def _read_lines_gz(path: Path) -> list[str]:
    with gzip.open(path, 'rt') as fh:
        return [ln.rstrip('\n') for ln in fh]


def _read_lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def _load_filtered_matrix(mm_dir: Path) -> tuple[sp.csr_matrix, list[str], list[str]]:
    mtx_path = mm_dir / 'matrix.mtx.gz'
    reg_path = mm_dir / 'regions.tsv.gz'
    bar_path = mm_dir / 'barcodes.tsv.gz'
    for p in (mtx_path, reg_path, bar_path):
        if not p.exists():
            raise FileNotFoundError(f'Missing {p}')
    mat = sio.mmread(mtx_path).tocsr()
    regions = _read_lines_gz(reg_path)
    barcodes = _read_lines_gz(bar_path)
    if mat.shape != (len(regions), len(barcodes)):
        raise ValueError(f'Shape mismatch: {mat.shape} vs ({len(regions)}, {len(barcodes)})')
    return mat, regions, barcodes


def _sample_pairs(
    mat: sp.csr_matrix,
    n_samples: int,
    rng: np.random.Generator,
    row_pool: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample n_samples positives (observed nonzeros) and n_samples negatives
    (sampled zero entries). row_pool, if given, restricts sampling to rows in
    the pool — used to keep heldout pairs inside the FITS-modeled subset."""
    R, C = mat.shape
    if row_pool is not None:
        if row_pool.size == 0:
            raise ValueError('Empty row pool')
        sub = mat[row_pool]
        coo = sub.tocoo()
        if coo.nnz < n_samples:
            raise ValueError(f'Only {coo.nnz} nonzero entries in row pool; '
                             f'asked for {n_samples} positives.')
        pos_local = rng.choice(coo.nnz, size=n_samples, replace=False)
        pos_rows = row_pool[coo.row[pos_local]].astype(np.int64)
        pos_cols = coo.col[pos_local].astype(np.int64)
        # negatives: sample from row_pool only
        nz_lin = (row_pool[coo.row].astype(np.int64) * C + coo.col.astype(np.int64))
        nz_set = set(nz_lin.tolist())
        seen: set[int] = set()
        neg_rows = np.empty(n_samples, dtype=np.int64)
        neg_cols = np.empty(n_samples, dtype=np.int64)
        filled = 0
        while filled < n_samples:
            need = n_samples - filled
            batch = int(need * 1.2) + 16
            r_idx = rng.integers(0, row_pool.size, size=batch, dtype=np.int64)
            r = row_pool[r_idx]
            c = rng.integers(0, C, size=batch, dtype=np.int64)
            lin = r.astype(np.int64) * C + c
            for ri, ci, li in zip(r, c, lin):
                if filled >= n_samples:
                    break
                li_i = int(li)
                if li_i in nz_set or li_i in seen:
                    continue
                seen.add(li_i)
                neg_rows[filled] = ri
                neg_cols[filled] = ci
                filled += 1
        return pos_rows, pos_cols, neg_rows, neg_cols

    coo = mat.tocoo()
    if coo.nnz < n_samples:
        raise ValueError(f'Only {coo.nnz} nonzero entries; asked for {n_samples} positives.')
    pos_ix = rng.choice(coo.nnz, size=n_samples, replace=False)
    pos_rows = coo.row[pos_ix].astype(np.int64)
    pos_cols = coo.col[pos_ix].astype(np.int64)

    nz_lin = coo.row.astype(np.int64) * C + coo.col.astype(np.int64)
    nz_set = set(nz_lin.tolist())
    seen: set[int] = set()
    neg_rows = np.empty(n_samples, dtype=np.int64)
    neg_cols = np.empty(n_samples, dtype=np.int64)
    filled = 0
    while filled < n_samples:
        need = n_samples - filled
        batch = int(need * 1.2) + 16
        r = rng.integers(0, R, size=batch, dtype=np.int64)
        c = rng.integers(0, C, size=batch, dtype=np.int64)
        lin = r * C + c
        for ri, ci, li in zip(r, c, lin):
            if filled >= n_samples:
                break
            li_i = int(li)
            if li_i in nz_set or li_i in seen:
                continue
            seen.add(li_i)
            neg_rows[filled] = ri
            neg_cols[filled] = ci
            filled += 1
    return pos_rows, pos_cols, neg_rows, neg_cols


def _fits_scores(
    h5_path: Path,
    rows: np.ndarray,
    cols: np.ndarray,
    full_to_h5_row: np.ndarray,
    h5_col_idx: np.ndarray,
    chunk: int = 200_000,
) -> np.ndarray:
    """Look up imputed values at the requested (region_in_orig, cell_in_orig)
    pairs, mapping through the imputed HDF5's row/column index. Pairs that fall
    outside the FITS-modeled set get score 0.0."""
    out = np.zeros(len(rows), dtype=np.float64)
    h5_rows = full_to_h5_row[rows]
    h5_cols = h5_col_idx[cols]
    valid = (h5_rows >= 0) & (h5_cols >= 0)
    if not valid.any():
        return out
    # h5py fancy indexing requires sorted, unique indices for efficiency. Read
    # in chunks of pairs so memory stays bounded.
    with h5py.File(h5_path, 'r') as h5:
        ds = h5['Prc']
        idxs = np.where(valid)[0]
        for s in range(0, len(idxs), chunk):
            sl = idxs[s:s + chunk]
            r_idx = h5_rows[sl]
            c_idx = h5_cols[sl]
            # h5py supports list-of-tuples coordinate selection but it's slow;
            # group by row.
            order = np.argsort(r_idx, kind='stable')
            r_sorted = r_idx[order]
            c_sorted = c_idx[order]
            uniq_rows, starts = np.unique(r_sorted, return_index=True)
            ends = np.r_[starts[1:], len(r_sorted)]
            scores_sorted = np.empty(len(sl), dtype=np.float64)
            for ur, st, en in zip(uniq_rows, starts, ends):
                row_vals = ds[ur, :]  # full row
                scores_sorted[st:en] = row_vals[c_sorted[st:en]]
            scores = np.empty_like(scores_sorted)
            scores[order] = scores_sorted
            out[sl] = scores
    return out


def _baseline_scores(
    mat: sp.csr_matrix,
    rows: np.ndarray,
    cols: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    R, C = mat.shape
    region_density = np.asarray(mat.sum(axis=1)).ravel() / float(C)
    cell_density   = np.asarray(mat.sum(axis=0)).ravel() / float(R)
    return {
        'random':       rng.random(len(rows)),
        'region_freq':  region_density[rows],
        'cell_freq':    cell_density[cols],
        'independence': region_density[rows] * cell_density[cols],
    }


def _metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        'auroc': float(roc_auc_score(y_true, y_score)),
        'auprc': float(average_precision_score(y_true, y_score)),
    }


def _plot(results: dict[str, dict[str, float]], out_png: Path, title: str) -> None:
    names = list(results.keys())
    auroc = [results[n]['auroc'] for n in names]
    auprc = [results[n]['auprc'] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, vals, ylab in ((axes[0], auroc, 'AUROC'), (axes[1], auprc, 'AUPRC')):
        xs = np.arange(len(names))
        bars = ax.bar(xs, vals, color=['#999'] * (len(names) - 1) + ['#c33'])
        ax.set_xticks(xs); ax.set_xticklabels(names, rotation=30, ha='right')
        ax.set_ylabel(ylab); ax.set_ylim(0.0, 1.0)
        ax.axhline(0.5, color='k', lw=0.5, ls=':')
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=8)
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(out_png, dpi=150); plt.close(fig)


def _write_mm_gz(mat: sp.csr_matrix, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('')
    sio.mmwrite(str(tmp), mat.tocoo(), field='integer', symmetry='general')
    with open(str(tmp) + '.mtx', 'rb') as fin, gzip.open(str(path), 'wb') as fout:
        fout.writelines(fin)
    Path(str(tmp) + '.mtx').unlink(missing_ok=True)


def _write_lines_gz(lines: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(path), 'wt') as fh:
        fh.write('\n'.join(lines) + '\n')


def main() -> int:
    ap = base_parser(__doc__ or '')
    ap.add_argument('--n-samples', type=int, default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--imputed-h5', type=str, default=None)
    ap.add_argument('--out-name', type=str, default='heldout_eval')
    ap.add_argument('--sample-space', type=str, default='modeled',
                    choices=['modeled', 'full'],
                    help='modeled: restrict heldout sampling to FITS-modeled rows '
                         '(default). full: sample anywhere; unmodeled rows score 0.')
    ap.add_argument('--prepare-holdout', action='store_true')
    ap.add_argument('--holdout-split', type=str, default=None)
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    eval_cfg = cfg.setdefault('eval', {})
    n_samples = int(args.n_samples or eval_cfg.get('n_samples', 100_000))
    seed = int(args.seed if args.seed is not None else eval_cfg.get('seed', 42))
    rng = np.random.default_rng(seed)

    mm_dir = Path(cfg['paths']['mm'])
    mat, regions, barcodes = _load_filtered_matrix(mm_dir)
    R, C = mat.shape
    log.info('Original matrix: %d regions x %d cells, nnz=%d (density=%.3g)',
             R, C, mat.nnz, mat.nnz / float(R * C))
    mat_bin = (mat > 0).astype(np.int8).tocsr()

    eval_dir = Path(cfg['paths']['eval'])
    eval_dir.mkdir(parents=True, exist_ok=True)

    h5_path = (Path(args.imputed_h5) if args.imputed_h5
               else Path(cfg['paths']['impute']) / 'imputed_Prc_hdf5_all.h5')

    # -- decide pair set ------------------------------------------------------
    if args.holdout_split:
        split = np.load(args.holdout_split)
        pos_rows, pos_cols = split['pos_rows'], split['pos_cols']
        neg_rows, neg_cols = split['neg_rows'], split['neg_cols']
        log.info('Loaded holdout split %s (n_pos=%d)', args.holdout_split, len(pos_rows))
    else:
        row_pool = None
        if args.sample_space == 'modeled':
            if not h5_path.exists():
                log.error('Need imputed HDF5 (%s) to determine modeled rows. '
                          'Run 05_impute.py first or pass --sample-space full.', h5_path)
                return 1
            with h5py.File(h5_path, 'r') as h5:
                imp_regs = [s.decode() if isinstance(s, bytes) else str(s)
                            for s in h5['regions'][:]]
                ds = h5['Prc']
                Rf = ds.shape[0]
                # Modeled rows = any nonzero predicted value in the row.
                modeled_h5 = np.zeros(Rf, dtype=bool)
                chunk = 50_000
                for i0 in range(0, Rf, chunk):
                    i1 = min(Rf, i0 + chunk)
                    modeled_h5[i0:i1] = ds[i0:i1, :].any(axis=1)
            reg_to_orig = {r: i for i, r in enumerate(regions)}
            modeled_orig_idx = []
            for h5_i, name in enumerate(imp_regs):
                if not modeled_h5[h5_i]:
                    continue
                oi = reg_to_orig.get(name, -1)
                if oi >= 0:
                    modeled_orig_idx.append(oi)
            row_pool = np.array(sorted(set(modeled_orig_idx)), dtype=np.int64)
            log.info('Sample space restricted to %d FITS-modeled rows '
                     '(of %d HDF5 rows; %d in mm matrix).',
                     row_pool.size, int(modeled_h5.sum()), len(regions))
            if row_pool.size == 0:
                log.error('No FITS-modeled rows are present in the mm matrix; '
                          'check that 02 and 05 used the same filter universe.')
                return 1

        log.info('Sampling %d positives + %d negatives (seed=%d)',
                 n_samples, n_samples, seed)
        pos_rows, pos_cols, neg_rows, neg_cols = _sample_pairs(
            mat_bin, n_samples, rng, row_pool=row_pool,
        )

    # -- prepare-holdout short-circuit ---------------------------------------
    if args.prepare_holdout:
        split_path = eval_dir / 'holdout_split.npz'
        np.savez_compressed(
            split_path,
            pos_rows=pos_rows.astype(np.int64),
            pos_cols=pos_cols.astype(np.int64),
            neg_rows=neg_rows.astype(np.int64),
            neg_cols=neg_cols.astype(np.int64),
            n_regions=np.int64(R), n_cells=np.int64(C),
            seed=np.int64(seed),
        )
        masked = mat_bin.tolil()
        masked[pos_rows, pos_cols] = 0
        masked = masked.tocsr(); masked.eliminate_zeros()
        mh = Path(cfg['paths']['work_dir']) / 'mm_holdout'
        _write_mm_gz(masked, mh / 'matrix.mtx.gz')
        _write_lines_gz(regions,  mh / 'regions.tsv.gz')
        _write_lines_gz(barcodes, mh / 'barcodes.tsv.gz')
        log.info('Wrote masked matrix -> %s (nnz %d -> %d)',
                 mh / 'matrix.mtx.gz', mat_bin.nnz, masked.nnz)
        log.info('Wrote holdout split -> %s', split_path)
        log.info('Next: rerun 02-05 with paths.mm pointing at %s, then '
                 '07_eval_heldout.py --holdout-split %s --imputed-h5 ...',
                 mh, split_path)
        return 0

    if not h5_path.exists():
        log.error('Imputed HDF5 not found: %s. Run 05_impute.py first.', h5_path)
        return 1

    # -- align HDF5 indices to original row/cell space -----------------------
    with h5py.File(h5_path, 'r') as h5:
        h5_regs = [s.decode() if isinstance(s, bytes) else str(s)
                   for s in h5['regions'][:]]
        h5_bars = [s.decode() if isinstance(s, bytes) else str(s)
                   for s in h5['barcodes'][:]]

    reg_to_orig = {r: i for i, r in enumerate(regions)}
    bar_to_orig = {b: i for i, b in enumerate(barcodes)}

    full_to_h5_row = -np.ones(R, dtype=np.int64)
    for h5_i, name in enumerate(h5_regs):
        oi = reg_to_orig.get(name, -1)
        if oi >= 0:
            full_to_h5_row[oi] = h5_i
    h5_col_idx = -np.ones(C, dtype=np.int64)
    for h5_j, name in enumerate(h5_bars):
        oj = bar_to_orig.get(name, -1)
        if oj >= 0:
            h5_col_idx[oj] = h5_j

    # -- score ----------------------------------------------------------------
    y_true = np.concatenate([np.ones(len(pos_rows), dtype=np.int8),
                              np.zeros(len(neg_rows), dtype=np.int8)])
    rows = np.concatenate([pos_rows, neg_rows])
    cols = np.concatenate([pos_cols, neg_cols])

    log.info('Scoring %d pairs against FITS HDF5 ...', len(rows))
    fits_scores = _fits_scores(h5_path, rows, cols, full_to_h5_row, h5_col_idx)

    scorers = {'FITS (HDF5)': fits_scores, **_baseline_scores(mat_bin, rows, cols, rng)}
    ordered = ['random', 'region_freq', 'cell_freq', 'independence', 'FITS (HDF5)']
    results = {name: _metrics(y_true, scorers[name]) for name in ordered}
    for name, m in results.items():
        log.info('  %-22s  AUROC=%.4f  AUPRC=%.4f', name, m['auroc'], m['auprc'])

    title = (f'FITS held-out reconstruction  (n_pos={len(pos_rows)}, '
             f'sample_space={args.sample_space}, seed={seed})')
    json_path = eval_dir / f'{args.out_name}.json'
    tsv_path  = eval_dir / f'{args.out_name}.tsv'
    png_path  = eval_dir / f'{args.out_name}.png'

    summary = {
        'n_pos':           int(len(pos_rows)),
        'n_neg':           int(len(neg_rows)),
        'n_regions':       int(R),
        'n_cells':         int(C),
        'matrix_density':  float(mat_bin.nnz / (R * C)),
        'sample_space':    args.sample_space,
        'seed':            seed,
        'imputed_h5':      str(h5_path),
        'holdout_split':   args.holdout_split,
        'results':         results,
    }
    json_path.write_text(json.dumps(summary, indent=2) + '\n')
    with open(tsv_path, 'w') as fh:
        fh.write('scorer\tauroc\tauprc\n')
        for name, m in results.items():
            fh.write(f'{name}\t{m["auroc"]:.6f}\t{m["auprc"]:.6f}\n')
    _plot(results, png_path, title)
    log.info('Wrote %s / %s / %s', json_path.name, tsv_path.name, png_path.name)
    return 0


if __name__ == '__main__':
    sys.exit(main())
