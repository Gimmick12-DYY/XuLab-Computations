#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 03_pu_classifier.py
#
# Positive-Unlabeled classifier for refining the geometric mask. Uses the
# Elkan-Noto (2008) re-weighting trick:
#
#   1. Train a probabilistic classifier h(x) = P(s=1 | x) where s is the
#      *labeled-as-positive* indicator (s=1 for seed positives,
#      s=0 for everything else -- treated as unlabeled, NOT as negatives).
#   2. Estimate the class-prior constant c = P(s=1 | y=1) as the mean of
#      h(x) over a held-out positive validation subset.
#   3. The calibrated PU score is g(x) = h(x) / c, clipped to [0, 1].
#
# Seed positives come from `bin_features.tsv` (column in_top_N_CTCF == 1),
# which step 02 derives from the top-N CTCF cCREs by HEK293 z-score subject
# to HEK293T-CA accessibility -- the same recipe Bin_posVSneg.ipynb uses.
# A held-out fraction (default 20 %) is set aside for step 07 validation and
# is *not* used during training.
#
# Outputs (under <work>/pu/):
#   pu_scores.tsv          bin_idx_orig, bin_name, in_train_pos, in_holdout_pos,
#                          h_score, pu_score, refined_mask_keep, mask_keep
#   refined_mask.tsv       chr:start-end strings that survive both the
#                          geometric mask and the PU threshold
#   refined_mask.bed       same in BED3 form
#   holdout_positives.tsv  bin_idx_orig, bin_name of the held-out positives
#                          (used by step 07_validate.py)
#   pu_meta.json           parameters, c estimate, threshold used, etc.
# -----------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger('03_pu')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


_REGION_RE = re.compile(r'^([^:]+):(\d+)-(\d+)$')


def _bed_row(name: str) -> tuple[str, int, int] | None:
    m = _REGION_RE.match(name)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def _coerce_features(df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Pull requested feature columns into a numeric matrix.

    Booleans / ints stay as-is; floats with NaN are filled with 0. Missing
    columns raise.
    """
    cols = []
    for f in feature_names:
        if f not in df.columns:
            raise SystemExit(f'Feature {f!r} not found in bin_features.tsv; '
                             f'available: {list(df.columns)}')
        v = df[f].to_numpy()
        if v.dtype == bool:
            v = v.astype(np.float32)
        else:
            v = np.asarray(v, dtype=np.float32)
            v = np.where(np.isnan(v), 0.0, v)
        cols.append(v)
    return np.stack(cols, axis=1).astype(np.float32)


def _make_classifier(kind: str, random_state: int):
    kind = (kind or 'logistic').lower()
    if kind == 'logistic':
        return LogisticRegression(
            solver='liblinear', class_weight='balanced',
            max_iter=2000, random_state=random_state,
        )
    if kind == 'rf':
        return RandomForestClassifier(
            n_estimators=400, n_jobs=-1, max_depth=None,
            class_weight='balanced_subsample', random_state=random_state,
        )
    raise SystemExit(f'Unknown classifier kind {kind!r}; expected logistic|rf.')


def _cv_oof_scores(X: np.ndarray, s: np.ndarray, kind: str, n_splits: int,
                   random_state: int) -> np.ndarray:
    """Compute out-of-fold P(s=1 | x) so the calibration constant uses
    predictions made by a model that did not see the example."""
    oof = np.zeros(X.shape[0], dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (tr, va) in enumerate(skf.split(X, s), start=1):
        clf = _make_classifier(kind, random_state + fold)
        clf.fit(X[tr], s[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
        log.info('  CV fold %d/%d: pos_train=%d pos_val=%d', fold, n_splits,
                 int(s[tr].sum()), int(s[va].sum()))
    return oof


def _pick_threshold(refined: np.ndarray, seed_in_mask: np.ndarray,
                    mode: str | float) -> tuple[float, str]:
    """Resolve refined_mask_threshold to a numeric cutoff."""
    if isinstance(mode, (int, float)) and not isinstance(mode, bool):
        return float(mode), f'fixed:{float(mode)}'
    s = str(mode).lower() if mode is not None else 'auto'
    if s == 'auto':
        # Keep about as many bins as the union of seed positives + bins whose
        # PU score is >= the median of seed-positive scores.
        if seed_in_mask.any():
            median_pos = float(np.median(refined[seed_in_mask]))
            return median_pos, f'auto:median_seed_pu={median_pos:.4g}'
        return 0.5, 'auto:default0.5_no_seed'
    try:
        return float(s), f'fixed:{float(s)}'
    except ValueError:
        return 0.5, 'auto:fallback0.5'


def main() -> int:
    ap = base_parser(__doc__ or '')
    ap.add_argument('--classifier', type=str, default=None,
                    choices=['logistic', 'rf'])
    ap.add_argument('--hold-out-fraction', type=float, default=None)
    ap.add_argument('--refined-mask-threshold', type=str, default=None)
    ap.add_argument('--seed-positive-column', type=str, default='in_top_N_CTCF')
    ap.add_argument('--mask-keep-column', type=str, default='mask_keep')
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    pu_cfg = cfg.setdefault('pu', {})
    if args.classifier:
        pu_cfg['classifier'] = args.classifier
    if args.hold_out_fraction is not None:
        pu_cfg['hold_out_fraction'] = args.hold_out_fraction
    if args.refined_mask_threshold is not None:
        pu_cfg['refined_mask_threshold'] = args.refined_mask_threshold

    feat_names = list(pu_cfg.get('features', []))
    if not feat_names:
        raise SystemExit('pu.features is empty; nothing to train on.')
    classifier_kind = str(pu_cfg.get('classifier', 'logistic'))
    rs = int(pu_cfg.get('random_state', 42))
    cv = int(pu_cfg.get('cv_folds', 5))
    holdout_frac = float(pu_cfg.get('hold_out_fraction', 0.2))
    refined_thresh_cfg = pu_cfg.get('refined_mask_threshold', 'auto')

    mask_dir = Path(cfg['paths']['mask'])
    pu_dir   = Path(cfg['paths']['pu'])
    bin_feat_path = mask_dir / 'bin_features.tsv'
    if not bin_feat_path.exists():
        raise SystemExit(f'Missing {bin_feat_path}. Run 02_build_mask.R first.')

    log.info('Reading bin features %s', bin_feat_path)
    df = pd.read_csv(bin_feat_path, sep='\t')
    log.info('  %d bins, %d feature columns', len(df), df.shape[1])

    if args.seed_positive_column not in df.columns:
        raise SystemExit(f'Seed column {args.seed_positive_column!r} not in features.')
    if args.mask_keep_column not in df.columns:
        raise SystemExit(f'Mask column {args.mask_keep_column!r} not in features.')

    seed_pos = df[args.seed_positive_column].to_numpy().astype(bool)
    mask_keep = df[args.mask_keep_column].to_numpy().astype(bool)
    n_seed_pos = int(seed_pos.sum())
    if n_seed_pos == 0:
        raise SystemExit('No seed positives; cannot train PU.')
    log.info('Seed positives: %d ; geometric-mask bins: %d',
             n_seed_pos, int(mask_keep.sum()))

    # ------------------------------------------------------------------
    # Holdout split of seed positives (used by step 07 validate.py).
    # ------------------------------------------------------------------
    rng = np.random.default_rng(rs)
    pos_idx_all = np.where(seed_pos)[0]
    n_holdout = max(1, int(round(holdout_frac * n_seed_pos))) if holdout_frac > 0 else 0
    if n_holdout >= n_seed_pos:
        raise SystemExit('hold_out_fraction leaves no training positives.')
    holdout_idx = np.sort(rng.choice(pos_idx_all, size=n_holdout, replace=False))
    holdout_mask = np.zeros(len(df), dtype=bool)
    holdout_mask[holdout_idx] = True
    train_pos = seed_pos & ~holdout_mask
    log.info('Holdout: %d / %d seed positives (%.0f%%) reserved for step 07',
             n_holdout, n_seed_pos, 100.0 * n_holdout / n_seed_pos)

    # ------------------------------------------------------------------
    # PU label: s=1 for *training* positives only; 0 for all others.
    # Hold-out positives are treated as unlabeled during training -- the
    # whole point is to check whether the trained model can recover them.
    # ------------------------------------------------------------------
    s = train_pos.astype(np.int8)
    X = _coerce_features(df, feat_names)
    log.info('Feature matrix: %s, positive class fraction (train) = %.4g',
             X.shape, float(s.mean()))

    # ------------------------------------------------------------------
    # Stage 1: out-of-fold P(s=1 | x) -- used for the calibration constant
    # and to give every bin a score that was not produced by a model that
    # saw it during training. We later refit a final model on ALL train data
    # to score the holdout + unlabeled bins.
    # ------------------------------------------------------------------
    log.info('Computing %d-fold OOF P(s=1 | x) with classifier=%s ...',
             cv, classifier_kind)
    h_oof = _cv_oof_scores(X, s, classifier_kind, cv, rs)

    log.info('Refitting on ALL training data for non-OOF scores ...')
    clf_final = _make_classifier(classifier_kind, rs)
    clf_final.fit(X, s)
    h_full = clf_final.predict_proba(X)[:, 1].astype(np.float64)
    # For training-positive rows, prefer OOF (avoid trivial 1.0); for others
    # use the full-data refit. This gives calibrated scores for unlabeled and
    # holdout bins while not over-rewarding the training positives.
    h_score = np.where(train_pos, h_oof, h_full)

    # ------------------------------------------------------------------
    # Elkan-Noto calibration: c = E_{x : y=1}[ h(x) ] estimated on validation
    # positives. We approximate with the OOF score on training positives,
    # which is the standard plug-in.
    # ------------------------------------------------------------------
    c_hat = float(h_oof[train_pos].mean()) if train_pos.any() else 1.0
    c_hat = max(c_hat, 1e-6)
    pu_score = np.clip(h_score / c_hat, 0.0, 1.0)
    log.info('Elkan-Noto c_hat = %.4g  =>  pu_score = h(x) / c_hat (clipped to [0, 1])',
             c_hat)

    # ------------------------------------------------------------------
    # Threshold -> refined mask. Refined mask must also satisfy the
    # geometric mask_keep flag from step 02.
    # ------------------------------------------------------------------
    thresh_value, thresh_origin = _pick_threshold(pu_score, train_pos, refined_thresh_cfg)
    refined = (pu_score >= thresh_value) & mask_keep
    log.info('Refined-mask threshold=%.4g (%s); refined bins=%d / %d (%.2f%% of mask_keep)',
             thresh_value, thresh_origin,
             int(refined.sum()), int(mask_keep.sum()),
             100.0 * refined.sum() / max(int(mask_keep.sum()), 1))

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    pu_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        'bin_idx_orig':       df['bin_idx_orig'].to_numpy().astype(np.int64),
        'bin_name':           df['bin_name'].astype(str).to_numpy(),
        'in_train_pos':       train_pos.astype(np.int8),
        'in_holdout_pos':     holdout_mask.astype(np.int8),
        'mask_keep':          mask_keep.astype(np.int8),
        'h_score':            h_score,
        'pu_score':           pu_score,
        'refined_mask_keep':  refined.astype(np.int8),
    })
    pu_scores_path = pu_dir / 'pu_scores.tsv'
    out_df.to_csv(pu_scores_path, sep='\t', index=False, float_format='%.6g')
    log.info('Wrote %s', pu_scores_path)

    refined_tsv = pu_dir / 'refined_mask.tsv'
    refined_bed = pu_dir / 'refined_mask.bed'
    refined_names = out_df.loc[refined, 'bin_name'].tolist()
    refined_tsv.write_text('\n'.join(refined_names) + '\n')

    with open(refined_bed, 'w') as fh:
        for nm in refined_names:
            parts = _bed_row(nm)
            if parts is None:
                continue
            chrom, start, end = parts
            fh.write(f'{chrom}\t{start}\t{end}\n')
    log.info('Wrote %s (%d intervals)', refined_bed, len(refined_names))

    holdout_path = pu_dir / 'holdout_positives.tsv'
    out_df[holdout_mask][['bin_idx_orig', 'bin_name']].to_csv(
        holdout_path, sep='\t', index=False)
    log.info('Wrote %s (%d holdout positives)', holdout_path, int(holdout_mask.sum()))

    meta = {
        'classifier':           classifier_kind,
        'features':             feat_names,
        'cv_folds':             cv,
        'random_state':         rs,
        'hold_out_fraction':    holdout_frac,
        'n_bins':               int(len(df)),
        'n_geometric_mask':     int(mask_keep.sum()),
        'n_seed_positive':      int(n_seed_pos),
        'n_train_positive':     int(train_pos.sum()),
        'n_holdout_positive':   int(holdout_mask.sum()),
        'c_hat':                float(c_hat),
        'refined_mask_threshold': {
            'value':  float(thresh_value),
            'origin': thresh_origin,
        },
        'n_refined_mask':       int(refined.sum()),
        'pu_score_distribution': {
            'mean':   float(pu_score.mean()),
            'median': float(np.median(pu_score)),
            'p90':    float(np.quantile(pu_score, 0.9)),
            'p99':    float(np.quantile(pu_score, 0.99)),
        },
        'outputs': {
            'pu_scores':         str(pu_scores_path),
            'refined_mask_tsv':  str(refined_tsv),
            'refined_mask_bed':  str(refined_bed),
            'holdout_positives': str(holdout_path),
        },
    }
    (pu_dir / 'pu_meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    log.info('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
