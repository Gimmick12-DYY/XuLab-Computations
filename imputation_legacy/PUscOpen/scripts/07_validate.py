#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 07_validate.py
#
# Held-out validation: did the PU + scOpen + post-filter stack recover the
# 20 % of seed positives we *deliberately withheld* during step 03 training?
#
# Sets of bins we use (all in original mm row index):
#   POS_HOLDOUT     held-out seed positives (from <pu>/holdout_positives.tsv)
#   POS_TRAIN       seed positives used for PU training (in pu_scores.tsv)
#   NEG_CONTROL     random sample of size |POS_HOLDOUT| from bins where
#                   pu_scores.tsv has refined_mask_keep == 0 AND mask_keep == 0
#                   (i.e. bins the pipeline never modeled)
#
# For each candidate set we look up per-bin signal in the final imputed
# matrix and report:
#   * recall@thresholds: fraction of POS_HOLDOUT bins with signal > t
#   * specificity@thresholds: fraction of NEG_CONTROL bins with signal <= t
#   * AUROC (POS_HOLDOUT vs NEG_CONTROL)
#   * precision@k (fraction of top-k signal bins that are in POS_HOLDOUT)
#
# Outputs (under <work>/validate/):
#   validate.tsv     per-threshold recall / specificity table
#   validate.json    summary: AUROC, max-F1, sparsity-match, etc.
#   validate_roc.png ROC curve for the held-out positives
#   validate_recall_at_k.png  recall@k plot
# -----------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve  # noqa: E402

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger('07_validate')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def _read_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text().splitlines() if ln]


def main() -> int:
    ap = base_parser(__doc__ or '')
    ap.add_argument('--n-neg-control', type=int, default=None,
                    help='Number of random bins to sample as the control '
                         'negative set (default: |POS_HOLDOUT|).')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-thresholds', type=int, default=48)
    ap.add_argument('--out-name', type=str, default='validate')
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    pu_dir   = Path(cfg['paths']['pu'])
    imp_dir  = Path(cfg['paths']['impute'])
    val_dir  = Path(cfg['paths']['validate'])
    val_dir.mkdir(parents=True, exist_ok=True)

    holdout_path  = pu_dir / 'holdout_positives.tsv'
    pu_scores_path = pu_dir / 'pu_scores.tsv'
    imp_csr_path = imp_dir / 'matrix_csr.npz'
    imp_dense_path = imp_dir / 'matrix.npy'
    imp_regions_path = imp_dir / 'regions.tsv'
    use_csr = imp_csr_path.is_file()
    matrix_path = imp_csr_path if use_csr else imp_dense_path
    for p in (holdout_path, pu_scores_path, matrix_path, imp_regions_path):
        if not p.exists():
            log.error('Missing %s; run earlier steps first.', p)
            return 1

    rng = np.random.default_rng(int(args.seed))

    log.info('Loading held-out positives %s', holdout_path)
    holdout_df = pd.read_csv(holdout_path, sep='\t')
    if 'bin_name' not in holdout_df.columns:
        log.error('holdout_positives.tsv missing bin_name column.')
        return 1
    holdout_names = set(holdout_df['bin_name'].astype(str))
    log.info('  %d holdout positives', len(holdout_names))

    log.info('Loading PU scores %s', pu_scores_path)
    pu_df = pd.read_csv(pu_scores_path, sep='\t')
    in_mask = pu_df['mask_keep'].astype(bool).to_numpy()
    in_refined = pu_df['refined_mask_keep'].astype(bool).to_numpy()
    names_all = pu_df['bin_name'].astype(str).to_numpy()

    # Control negatives: bins not in the geometric mask (the pipeline never
    # touched them, so the model can't trivially recover them by training).
    neg_pool_idx = np.where(~in_mask)[0]
    n_ctrl_default = len(holdout_names)
    n_ctrl = int(args.n_neg_control) if args.n_neg_control else n_ctrl_default
    n_ctrl = min(n_ctrl, len(neg_pool_idx))
    if n_ctrl <= 0:
        log.error('No bins outside the geometric mask to use as controls.')
        return 1
    ctrl_idx = np.sort(rng.choice(neg_pool_idx, size=n_ctrl, replace=False))
    ctrl_names = set(names_all[ctrl_idx].tolist())
    log.info('  %d control bins (outside geometric mask)', n_ctrl)

    log.info('Loading final imputed matrix %s', matrix_path)
    if use_csr:
        matrix = sparse.load_npz(imp_csr_path)
    else:
        matrix = np.load(imp_dense_path, mmap_mode='r')
    imp_regions = _read_lines(imp_regions_path)
    name_to_row = {nm: i for i, nm in enumerate(imp_regions)}
    log.info('  imputed matrix shape=%s; %d region IDs',
             matrix.shape, len(imp_regions))

    def _signal(names: set[str]) -> tuple[np.ndarray, int]:
        rows = []
        absent = 0
        for nm in names:
            r = name_to_row.get(nm)
            if r is None:
                absent += 1
            else:
                rows.append(r)
        if not rows:
            return np.zeros(0, dtype=np.float64), absent
        rows_arr = np.asarray(rows, dtype=np.int64)
        if use_csr:
            sub = matrix[rows_arr]
            sig = np.asarray(sub.sum(axis=1)).ravel().astype(np.float64)
        else:
            sub = np.asarray(matrix[rows_arr, :], dtype=np.float64)
            sig = sub.sum(axis=1)
        return sig, absent

    pos_sig, pos_absent = _signal(holdout_names)
    ctrl_sig, ctrl_absent = _signal(ctrl_names)

    n_pos_modeled = pos_sig.size
    n_pos_unmodeled = pos_absent
    log.info('Holdout positives in modeled regions: %d / %d (%.1f%%)',
             n_pos_modeled, n_pos_modeled + n_pos_unmodeled,
             100.0 * n_pos_modeled / max(n_pos_modeled + n_pos_unmodeled, 1))
    log.info('Control bins in modeled regions: %d / %d',
             ctrl_sig.size, ctrl_sig.size + ctrl_absent)

    # Treat absent (unmodeled) bins as signal=0; they cannot exceed any t > 0
    # so they always count as predicted-negative.
    pos_full = np.concatenate([pos_sig,  np.zeros(pos_absent, dtype=np.float64)])
    ctrl_full = np.concatenate([ctrl_sig, np.zeros(ctrl_absent, dtype=np.float64)])
    y_true = np.concatenate([np.ones(pos_full.size,  dtype=np.int8),
                              np.zeros(ctrl_full.size, dtype=np.int8)])
    y_score = np.concatenate([pos_full, ctrl_full])

    # AUROC + AUPR
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auroc = float('nan')
    try:
        aupr = float(average_precision_score(y_true, y_score))
    except ValueError:
        aupr = float('nan')

    # Threshold sweep over the union of positive + control signals.
    s_sorted = np.unique(y_score)
    if s_sorted.size > args.n_thresholds:
        idx = np.linspace(0, s_sorted.size - 1, args.n_thresholds).astype(int)
        thresholds = s_sorted[idx]
    else:
        thresholds = s_sorted
    thresholds = np.concatenate([[0.0], thresholds])
    thresholds = np.unique(thresholds)

    rows = []
    for t in thresholds:
        tp = int((pos_full > t).sum())
        fn = int(pos_full.size - tp)
        fp = int((ctrl_full > t).sum())
        tn = int(ctrl_full.size - fp)
        recall = tp / max(pos_full.size, 1)
        spec   = tn / max(ctrl_full.size, 1)
        prec   = tp / max(tp + fp, 1)
        f1     = 2 * prec * recall / max(prec + recall, 1e-30)
        rows.append({
            'threshold':   float(t),
            'TP':          tp, 'FN': fn, 'FP': fp, 'TN': tn,
            'recall':      recall,
            'specificity': spec,
            'precision':   prec,
            'F1':          f1,
        })

    df_out = pd.DataFrame(rows)
    tsv_path = val_dir / f'{args.out_name}.tsv'
    df_out.to_csv(tsv_path, sep='\t', index=False, float_format='%.6g')
    log.info('Wrote %s', tsv_path)

    # Recall @ k (rank bins by signal across positive + control union; how
    # many of the top-k are holdout positives?)
    order = np.argsort(-y_score)
    recall_at_k = np.cumsum(y_true[order] == 1) / max(int(y_true.sum()), 1)
    ks = np.linspace(1, y_score.size, min(60, y_score.size)).astype(int)
    recall_at_k_table = [{'k': int(k), 'recall': float(recall_at_k[k - 1])} for k in ks]

    summary = {
        'n_holdout_positives':       int(pos_full.size),
        'n_holdout_in_modeled':      int(n_pos_modeled),
        'n_control':                 int(ctrl_full.size),
        'n_control_in_modeled':      int(ctrl_sig.size),
        'auroc':                     auroc,
        'aupr':                      aupr,
        'best_f1':                   max((r['F1'] for r in rows), default=float('nan')),
        'best_f1_threshold':         float(max(rows, key=lambda r: r['F1'])['threshold']) if rows else float('nan'),
        'recall_at_k':               recall_at_k_table,
        'imputed_matrix':            str(matrix_path),
        'holdout_positives':         str(holdout_path),
    }
    json_path = val_dir / f'{args.out_name}.json'
    json_path.write_text(json.dumps(summary, indent=2) + '\n')
    log.info('Wrote %s', json_path)

    # ROC plot
    fig, ax = plt.subplots(figsize=(6.5, 6))
    if y_true.min() != y_true.max():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, lw=1.4, color='#C44E52',
                label=f'PUscOpen  AUROC={auroc:.3f}')
    ax.plot([0, 1], [0, 1], color='k', lw=0.5, ls=':')
    ax.set_xlabel('False positive rate (1 − specificity)')
    ax.set_ylabel('True positive rate (recall)')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_title('Held-out CTCF positives recovery')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(val_dir / f'{args.out_name}_roc.png', dpi=150)
    plt.close(fig)

    # Recall @ k
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([d['k'] for d in recall_at_k_table],
            [d['recall'] for d in recall_at_k_table],
            color='#3B7EA1', lw=1.5)
    ax.set_xlabel('k (top-k bins by imputed signal)')
    ax.set_ylabel('Recall of held-out positives')
    ax.set_ylim(0, 1.02)
    ax.set_title(f'Recall@k vs imputed-signal rank '
                 f'(n_holdout={pos_full.size}, n_ctrl={ctrl_full.size})')
    fig.tight_layout()
    fig.savefig(val_dir / f'{args.out_name}_recall_at_k.png', dpi=150)
    plt.close(fig)

    log.info('AUROC=%.4f  AUPR=%.4f  best_F1=%.3f@t=%.3g',
             auroc, aupr, summary['best_f1'], summary['best_f1_threshold'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
