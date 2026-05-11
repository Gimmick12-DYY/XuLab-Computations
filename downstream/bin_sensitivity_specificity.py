#!/usr/bin/env python
# -----------------------------------------------------------------------------
# bin_sensitivity_specificity.py
#
# Bin-level coverage report. For each pipeline and each bin set (positive /
# negative bins from prepare_pos_neg_bins.R), report the fraction of bins
# whose per-bin total imputed signal exceeds the chosen threshold.
#
#   pos_coverage = #(positive bins with signal > t) / n_pos
#   neg_coverage = #(negative bins with signal > t) / n_neg
#
# Threshold modes (top-K wins when set; otherwise sparsity matching; otherwise
# fixed threshold):
#   --top-k N             : every pipeline calls its top-N bins (apples-to-apples).
#   --match-to LABEL      : SPARSITY-MATCHED. K = number of bins called by LABEL
#                           (default 'raw') at --threshold. K can be scaled by
#                           --match-multiplier. Every pipeline (including LABEL)
#                           then calls its top-K bins.
#   --threshold T         : fallback when neither is set; covered = signal > T.
#
# No ROC / AUPR / F1 -- just coverage at a comparable signal level.
#
# Each --input PATH is one of:
#   * directory with matrix.mtx.gz                (raw counts; e.g. <work>/mm)
#   * directory with factors.npz + regions.tsv    (cisTopic / scOpen)
#   * directory with matrix.npy  + regions.tsv    (FITS / MAGIC)
#   * legacy HDF5 file with /Prc                  (older runs)
#
# Usage:
#   python bin_sensitivity_specificity.py \
#     --bins-tsv /work/.../downstream/bins/pos_neg_bins.tsv \
#     --out-dir  /work/.../downstream/bin_sensitivity_specificity \
#     --input raw=/work/.../mm \
#     --input cisTopic=/work/.../cistopic_ctcf/impute \
#     --input scOpen=/work/.../scOpen/work/ctcf/impute \
#     --input MAGIC=/work/.../MAGIC/work/ctcf/impute
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
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

log = logging.getLogger('bin_coverage')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
def parse_input_arg(s: str) -> tuple[str, Path]:
    if '=' not in s:
        raise argparse.ArgumentTypeError(f"--input expects LABEL=PATH, got {s!r}")
    label, path = s.split('=', 1)
    label = label.strip()
    path = Path(path.strip())
    if not label:
        raise argparse.ArgumentTypeError(f"--input has empty LABEL: {s!r}")
    if not path.exists():
        raise argparse.ArgumentTypeError(f"--input path does not exist: {path}")
    return label, path


# -----------------------------------------------------------------------------
# Bin-index loader (matches compare_pos_neg.py's pos_neg_bins.tsv schema)
# -----------------------------------------------------------------------------
def load_bins_tsv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    pos_idx: list[int] = []
    neg_idx: list[int] = []
    pos_nm:  list[str] = []
    neg_nm:  list[str] = []
    with open(path) as fh:
        header = fh.readline().rstrip('\n').split('\t')
        try:
            i_orig = header.index('bin_idx_orig')
            i_lab  = header.index('label')
            i_nm   = header.index('bin_name')
        except ValueError as e:
            raise SystemExit(f'Bad header in {path}: {e}')
        for ln in fh:
            row = ln.rstrip('\n').split('\t')
            if not row[0]:
                continue
            if row[i_lab] == 'pos':
                pos_idx.append(int(row[i_orig])); pos_nm.append(row[i_nm])
            else:
                neg_idx.append(int(row[i_orig])); neg_nm.append(row[i_nm])
    return (np.asarray(pos_idx, dtype=np.int64),
            np.asarray(neg_idx, dtype=np.int64),
            pos_nm, neg_nm)


# -----------------------------------------------------------------------------
# Per-bin total-signal loaders
#
# Each returns (signal, modeled_mask):
#   signal[k]   = sum of imputed (or raw) counts across cells at bin k
#   modeled[k]  = True if bin k is present in the imputer's modeled universe
# -----------------------------------------------------------------------------
def per_bin_total_from_mm(mm_dir: Path, bin_idx: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray]:
    mtx_path = mm_dir / 'matrix.mtx.gz'
    if not mtx_path.exists():
        raise SystemExit(f'Missing {mtx_path}')
    log.info('  loading mm matrix %s', mtx_path)
    mat = sio.mmread(str(mtx_path)).tocsr()
    n_rows = mat.shape[0]
    if bin_idx.size and bin_idx.max() >= n_rows:
        raise SystemExit(
            f'bin_idx max ({int(bin_idx.max())}) out of range for mm with '
            f'{n_rows} rows. Mismatch between bins TSV and mm directory.')
    signal = np.asarray(mat[bin_idx].sum(axis=1)).ravel().astype(np.float64)
    return signal, np.ones(bin_idx.size, dtype=bool)


def _build_name_to_row(regions_path: Path) -> dict[str, int]:
    if not regions_path.exists():
        raise SystemExit(f'Missing {regions_path}')
    return {ln: i for i, ln in enumerate(regions_path.read_text().splitlines()) if ln}


def per_bin_total_from_factored(impute_dir: Path, bin_names: list[str]
                                 ) -> tuple[np.ndarray, np.ndarray]:
    factors_path = impute_dir / 'factors.npz'
    regions_path = impute_dir / 'regions.tsv'
    log.info('  loading factored output: %s + %s', factors_path, regions_path)
    name_to_row = _build_name_to_row(regions_path)
    with np.load(factors_path) as z:
        W = np.asarray(z['W'], dtype=np.float64)
        H = np.asarray(z['H'], dtype=np.float64)
        per_cell_factor = float(z['per_cell_factor']) if 'per_cell_factor' in z.files else 1.0
    if W.shape[1] != H.shape[0]:
        raise SystemExit(f'Rank mismatch in {factors_path}: W{W.shape} vs H{H.shape}')
    H_sum = H.sum(axis=1)
    bin_totals_modeled = (W @ H_sum) * per_cell_factor

    signal = np.zeros(len(bin_names), dtype=np.float64)
    modeled = np.zeros(len(bin_names), dtype=bool)
    for k, name in enumerate(bin_names):
        row = name_to_row.get(name)
        if row is not None:
            signal[k] = bin_totals_modeled[row]
            modeled[k] = True
    return signal, modeled


def per_bin_total_from_dense(impute_dir: Path, bin_names: list[str]
                              ) -> tuple[np.ndarray, np.ndarray]:
    matrix_path = impute_dir / 'matrix.npy'
    regions_path = impute_dir / 'regions.tsv'
    log.info('  loading dense output: %s + %s', matrix_path, regions_path)
    name_to_row = _build_name_to_row(regions_path)
    arr = np.load(matrix_path, mmap_mode='r')

    wanted_rows: list[int] = []
    name_to_query: list[int] = []
    for k, name in enumerate(bin_names):
        row = name_to_row.get(name)
        if row is not None:
            wanted_rows.append(row); name_to_query.append(k)
    if not wanted_rows:
        return (np.zeros(len(bin_names), dtype=np.float64),
                np.zeros(len(bin_names), dtype=bool))
    wanted = np.asarray(wanted_rows, dtype=np.int64)
    sub = np.asarray(arr[wanted, :], dtype=np.float64)
    bin_totals = sub.sum(axis=1)

    signal = np.zeros(len(bin_names), dtype=np.float64)
    modeled = np.zeros(len(bin_names), dtype=bool)
    for k, q in enumerate(name_to_query):
        signal[q] = bin_totals[k]
        modeled[q] = True
    return signal, modeled


def per_bin_total_from_hdf5(h5_path: Path, bin_idx: np.ndarray,
                             chunk: int = 50_000
                             ) -> tuple[np.ndarray, np.ndarray]:
    log.info('  loading legacy Prc from %s', h5_path)
    sort_order = np.argsort(bin_idx, kind='stable')
    sorted_idx = bin_idx[sort_order]
    uniq, inv = np.unique(sorted_idx, return_inverse=True)
    out_uniq = np.zeros(uniq.size, dtype=np.float64)
    with h5py.File(h5_path, 'r') as h5:
        ds = h5['Prc']
        if uniq.size and uniq.max() >= ds.shape[0]:
            raise SystemExit(
                f'Bin index {int(uniq.max())} out of range for HDF5 with '
                f'{ds.shape[0]} rows.')
        for s in range(0, uniq.size, chunk):
            sl = uniq[s:s + chunk]
            block = ds[list(sl), :]
            out_uniq[s:s + chunk] = block.sum(axis=1, dtype=np.float64)
    out_sorted = out_uniq[inv]
    out = np.empty_like(out_sorted)
    out[sort_order] = out_sorted
    return out, (out > 0)


def detect_input_kind(path: Path) -> str:
    if path.is_file():
        if path.suffix.lower() in ('.h5', '.hdf5'):
            return 'hdf5'
        raise SystemExit(f'Unsupported input file: {path}')
    if path.is_dir():
        if (path / 'factors.npz').exists() and (path / 'regions.tsv').exists():
            return 'factored'
        if (path / 'matrix.npy').exists() and (path / 'regions.tsv').exists():
            return 'dense'
        if (path / 'matrix.mtx.gz').exists():
            return 'mm'
        cand = list(path.glob('*.h5')) + list(path.glob('*.hdf5'))
        if cand:
            return 'hdf5_in_dir'
        try:
            names = sorted(p.name for p in path.iterdir() if p.is_file())
        except OSError as exc:
            names = [f'<oserror: {exc}>']
        head = names[:40]
        more = f' (+{len(names) - 40} more)' if len(names) > 40 else ''
        raise SystemExit(
            f'Cannot determine input kind from {path}. Expected one of: '
            'factors.npz + regions.tsv; matrix.npy + regions.tsv; matrix.mtx.gz; '
            f'or a *.h5 / *.hdf5 file (FITS: run scripts/05_impute.py after Phase 2). '
            f'Files present: {head!r}{more}'
        )
    if not path.exists():
        raise SystemExit(f'Input path does not exist: {path}')
    raise SystemExit(f'Unsupported input path (not a file or directory): {path}')


def per_bin_total(label: str, path: Path, bin_idx: np.ndarray,
                   bin_names: list[str]) -> tuple[np.ndarray, np.ndarray, str]:
    kind = detect_input_kind(path)
    if kind == 'mm':
        sig, mod = per_bin_total_from_mm(path, bin_idx)
    elif kind == 'factored':
        sig, mod = per_bin_total_from_factored(path, bin_names)
    elif kind == 'dense':
        sig, mod = per_bin_total_from_dense(path, bin_names)
    elif kind == 'hdf5':
        sig, mod = per_bin_total_from_hdf5(path, bin_idx)
    elif kind == 'hdf5_in_dir':
        cand = sorted(list(path.glob('*.h5')) + list(path.glob('*.hdf5')))
        sig, mod = per_bin_total_from_hdf5(cand[0], bin_idx)
        kind = 'hdf5'
    else:
        raise SystemExit(f'Unhandled input kind {kind} for {label}={path}')
    return sig, mod, kind


# -----------------------------------------------------------------------------
# Coverage core
# -----------------------------------------------------------------------------
def coverage_for_group(signal: np.ndarray, modeled: np.ndarray,
                        covered: np.ndarray) -> dict[str, float | int]:
    """Coverage stats for a single bin group (pos or neg) given a pre-computed
    boolean ``covered`` mask aligned with ``signal``."""
    n = int(signal.size)
    n_modeled = int(modeled.sum())
    n_covered = int(covered.sum())
    n_covered_modeled = int((covered & modeled).sum())
    return {
        'n':                       n,
        'n_modeled':               n_modeled,
        'n_covered':               n_covered,
        'n_covered_modeled':       n_covered_modeled,
        'coverage':                (n_covered / n) if n else float('nan'),
        'coverage_pct':            (100.0 * n_covered / n) if n else float('nan'),
        'coverage_modeled':        (n_covered_modeled / n_modeled) if n_modeled else float('nan'),
        'coverage_modeled_pct':    (100.0 * n_covered_modeled / n_modeled) if n_modeled else float('nan'),
    }


def called_mask_threshold(signal: np.ndarray, threshold: float) -> np.ndarray:
    """Mask of bins with signal > threshold."""
    return signal > threshold


def called_mask_top_k(signal: np.ndarray, k: int,
                       rng: np.random.Generator | None = None
                       ) -> tuple[np.ndarray, float, dict[str, int]]:
    """Pick the top-``k`` bins by signal with **random tie-breaking** at the
    cutoff value.

    Without random tie-breaking, ``argpartition`` resolves ties by array
    position. Because our bin array is ``concatenate([pos_bins, neg_bins])``,
    that gives a spurious advantage to pos bins whenever an imputer has many
    tied zeros (e.g. cisTopic, which models a small subset and is zero
    everywhere else). With random tie-breaking, ties at the cutoff are
    resolved by uniform sampling so neither group is favored.

    Returns ``(mask, cutoff_value, info)`` where ``info`` reports how many bins
    were strictly above the cutoff vs. tied at it (and how many of the tied
    ones were sampled into the mask). ``mask.sum() == k`` (clipped to [0, n]).
    """
    n = signal.size
    info: dict[str, int] = {'n_above_cutoff': 0, 'n_tied_at_cutoff': 0,
                             'n_tied_picked': 0}
    if k <= 0:
        return np.zeros(n, dtype=bool), float('inf'), info
    if k >= n:
        return np.ones(n, dtype=bool), float('-inf'), info
    cutoff = float(np.partition(signal, n - k)[n - k])
    above = signal > cutoff
    n_above = int(above.sum())
    tied_idx = np.where(signal == cutoff)[0]
    info['n_above_cutoff']   = n_above
    info['n_tied_at_cutoff'] = int(tied_idx.size)
    need = k - n_above
    info['n_tied_picked']    = int(need)
    mask = above.copy()
    if need > 0 and tied_idx.size:
        if rng is None:
            picked = tied_idx[:need]  # deterministic fallback
        else:
            picked = rng.choice(tied_idx, size=min(need, tied_idx.size),
                                 replace=False)
        mask[picked] = True
    return mask, cutoff, info


def _safe_log2_ratio(num: float, den: float, eps: float = 1e-30) -> float:
    return float(np.log2(max(num, eps) / max(den, eps)))


def intensity_stats(signal: np.ndarray, pos_mask: np.ndarray,
                    neg_mask: np.ndarray) -> dict[str, float | int]:
    """Intensity separation stats on the *called* bins."""
    pos_vals = signal[pos_mask]
    neg_vals = signal[neg_mask]
    pos_n = int(pos_vals.size)
    neg_n = int(neg_vals.size)
    pos_mean = float(pos_vals.mean()) if pos_n else float('nan')
    neg_mean = float(neg_vals.mean()) if neg_n else float('nan')
    pos_median = float(np.median(pos_vals)) if pos_n else float('nan')
    neg_median = float(np.median(neg_vals)) if neg_n else float('nan')
    return {
        'n_pos_called': pos_n,
        'n_neg_called': neg_n,
        'pos_mean_called': pos_mean,
        'neg_mean_called': neg_mean,
        'pos_median_called': pos_median,
        'neg_median_called': neg_median,
        'log2_mean_ratio_called': _safe_log2_ratio(pos_mean, neg_mean),
        'log2_median_ratio_called': _safe_log2_ratio(pos_median, neg_median),
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_coverage_bars(per_input: list[dict], out_path: Path,
                        title: str, y_label: str) -> None:
    labels = [i['label'] for i in per_input]
    pos_pct = [i['pos']['coverage_pct'] for i in per_input]
    neg_pct = [i['neg']['coverage_pct'] for i in per_input]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(labels) + 2), 4.2))
    b1 = ax.bar(x - width / 2, pos_pct, width, label='Positive bins',
                 color='#C44E52')
    b2 = ax.bar(x + width / 2, neg_pct, width, label='Negative bins',
                 color='#4C72B0')
    for grp, vals in ((b1, pos_pct), (b2, neg_pct)):
        for b, v in zip(grp, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.5,
                        f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
    sublabels = [
        f't={i["threshold"]:.3g}\n(K={i["pos"]["n_covered"] + i["neg"]["n_covered"]})'
        for i in per_input
    ]
    ax2 = ax.secondary_xaxis('bottom')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sublabels, fontsize=7)
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params(axis='x', which='both', length=0, pad=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='center')
    y_top = max(100, (max(pos_pct + neg_pct + [0]) + 3))
    ax.set_ylim(0, y_top)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.axhline(100, color='k', lw=0.4, ls=':')
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_intensity_lift_bars(per_input: list[dict], out_path: Path) -> None:
    labels = [i['label'] for i in per_input]
    med_lift = [i['intensity_called']['log2_median_ratio_called'] for i in per_input]
    mean_lift = [i['intensity_called']['log2_mean_ratio_called'] for i in per_input]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(labels) + 2), 4.2))
    b1 = ax.bar(x - width / 2, med_lift, width, label='log2 median(pos/neg)',
                 color='#55A868')
    b2 = ax.bar(x + width / 2, mean_lift, width, label='log2 mean(pos/neg)',
                 color='#8172B2')
    for grp, vals in ((b1, med_lift), (b2, mean_lift)):
        for b, v in zip(grp, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + (0.05 if v >= 0 else -0.1),
                        f'{v:.2f}', ha='center',
                        va='bottom' if v >= 0 else 'top', fontsize=8)
    ax.axhline(0, color='k', lw=0.6, ls=':')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.set_ylabel('Positive-vs-negative intensity lift (called bins)')
    ax.set_title('Imputation complexity gain on called bins')
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------
def write_tsv(per_input: list[dict], out_path: Path) -> None:
    cols = ['input', 'kind', 'group', 'mode', 'threshold',
            'n', 'n_modeled', 'n_covered', 'n_covered_modeled',
            'coverage_pct', 'coverage_modeled_pct']
    with open(out_path, 'w') as fh:
        fh.write('\t'.join(cols) + '\n')
        for info in per_input:
            for group in ('pos', 'neg'):
                s = info[group]
                fh.write('\t'.join([
                    info['label'], info['kind'], group,
                    info['mode'], f"{info['threshold']:.6g}",
                    str(s['n']), str(s['n_modeled']),
                    str(s['n_covered']), str(s['n_covered_modeled']),
                    f"{s['coverage_pct']:.4f}",
                    f"{s['coverage_modeled_pct']:.4f}",
                ]) + '\n')


def build_summary(per_input: list[dict], bins_tsv: str,
                   default_threshold: float, match_to: str | None,
                   match_multiplier: float, top_k: int | None,
                   target_calls: int | None) -> dict:
    out_inputs: list[dict] = []
    for info in per_input:
        out_inputs.append({
            'label':     info['label'],
            'kind':      info['kind'],
            'path':      info['path'],
            'mode':      info['mode'],
            'threshold': info['threshold'],
            'n_called':  info['n_called'],
            'intensity_called': info['intensity_called'],
            'pos':       info['pos'],
            'neg':       info['neg'],
        })
    return {
        'bins_tsv':          bins_tsv,
        'default_threshold': default_threshold,
        'match_to':          match_to,
        'match_multiplier':  match_multiplier,
        'top_k':             top_k,
        'target_calls':      target_calls,
        'inputs':            out_inputs,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__ or '',
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bins-tsv', type=Path, required=True,
                    help='Output of prepare_pos_neg_bins.R')
    ap.add_argument('--out-dir',  type=Path, required=True)
    ap.add_argument('--input',    type=parse_input_arg, action='append',
                    required=True, dest='inputs',
                    metavar='LABEL=PATH',
                    help='LABEL=PATH; PATH = mm/ directory, factored/dense '
                         'impute/ directory, or legacy HDF5 file. May be '
                         'passed multiple times.')
    ap.add_argument('--out-name', type=str, default='bin_coverage')
    ap.add_argument('--threshold', type=float, default=0.0,
                    help='Per-bin signal threshold for the reference input '
                         '(or, with --no-match, every input). Default 0.0.')
    ap.add_argument('--match-to', type=str, default='raw',
                    help='Reference input label for sparsity-matched coverage. '
                         'Default "raw"; pass "" or use --no-match to disable.')
    ap.add_argument('--no-match', dest='match_to', action='store_const', const='',
                    help='Disable sparsity matching; use --threshold everywhere.')
    ap.add_argument('--match-multiplier', type=float, default=1.0,
                    help='Scale the reference K by this factor when --match-to '
                         'is active (default 1.0).')
    ap.add_argument('--top-k', type=int, default=None,
                    help='Explicit number of top bins to call for every '
                         'pipeline. Overrides --match-to / --match-multiplier.')
    ap.add_argument('--seed', type=int, default=2026,
                    help='RNG seed for top-K tie-breaking (default 2026).')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info('Loading bin indices from %s', args.bins_tsv)
    pos_idx, neg_idx, pos_nm, neg_nm = load_bins_tsv(args.bins_tsv)
    log.info('Loaded %d positives, %d negatives', pos_idx.size, neg_idx.size)

    bin_idx = np.concatenate([pos_idx, neg_idx])
    bin_names = pos_nm + neg_nm
    n_pos = int(pos_idx.size)

    # First pass: load per-bin signal for every input.
    raw_inputs: list[dict] = []
    input_labels = [lbl for lbl, _ in args.inputs]
    for label, path in args.inputs:
        log.info('Input %r -> %s', label, path)
        signal, modeled, kind = per_bin_total(label, path, bin_idx, bin_names)
        raw_inputs.append({
            'label': label, 'path': str(path), 'kind': kind,
            'signal': signal, 'modeled': modeled,
        })

    # Decide target_calls (K). Top-K (if set) wins; else sparsity match against
    # the reference at --threshold, scaled by --match-multiplier; else fixed
    # threshold for everyone.
    target_calls: int | None = None
    ref_label = args.match_to or ''
    match_source = ''
    if args.top_k is not None and args.top_k > 0:
        target_calls = int(args.top_k)
        match_source = f'--top-k {target_calls}'
        ref_label = ''  # uniform top-K; no asymmetric reference
    elif ref_label:
        ref = next((r for r in raw_inputs if r['label'] == ref_label), None)
        if ref is None:
            log.warning('--match-to %r not in inputs %s; falling back to fixed threshold',
                        ref_label, input_labels)
            ref_label = ''
        else:
            base_k = int(called_mask_threshold(ref['signal'], args.threshold).sum())
            target_calls = int(round(base_k * args.match_multiplier))
            target_calls = min(max(target_calls, 0), int(ref['signal'].size))
            match_source = (
                f'reference={ref_label!r} at t={args.threshold:g} '
                f'(base_K={base_k}) × {args.match_multiplier:g}'
            )
    if target_calls is not None:
        log.info('Top-K = %d (%s)', target_calls, match_source)
    else:
        log.info('No top-K; using fixed threshold %.6g for every input',
                 args.threshold)

    # Second pass: per-input call masks. When target_calls is set, every input
    # (including any reference) gets the same top-K rule for honest comparison.
    per_input: list[dict] = []
    for rec in raw_inputs:
        label = rec['label']
        signal = rec['signal']; modeled = rec['modeled']; kind = rec['kind']

        if target_calls is not None:
            mask, cutoff, tie_info = called_mask_top_k(signal, target_calls, rng)
            thr = float(cutoff)
            mode = 'top_k'
            if tie_info['n_tied_at_cutoff'] > tie_info['n_tied_picked']:
                log.info('  %s: cutoff %.6g had %d ties; %d above, %d sampled '
                         'from ties (random tie-break; seed %d)',
                         label, cutoff,
                         tie_info['n_tied_at_cutoff'],
                         tie_info['n_above_cutoff'],
                         tie_info['n_tied_picked'],
                         args.seed)
        else:
            mask = called_mask_threshold(signal, args.threshold)
            thr = float(args.threshold)
            mode = 'fixed'

        pos_mask = mask[:n_pos];   neg_mask = mask[n_pos:]
        pos_stats = coverage_for_group(signal[:n_pos], modeled[:n_pos], pos_mask)
        neg_stats = coverage_for_group(signal[n_pos:], modeled[n_pos:], neg_mask)
        # intensity_stats expects full-length boolean masks aligned to full signal
        pos_mask_full = np.zeros(signal.size, dtype=bool)
        neg_mask_full = np.zeros(signal.size, dtype=bool)
        pos_mask_full[:n_pos] = pos_mask
        neg_mask_full[n_pos:] = neg_mask
        lift_stats = intensity_stats(signal, pos_mask_full, neg_mask_full)

        log.info(
            '  %s [%s, t=%.6g]  pos: %d/%d (%.2f%%; modeled %.2f%%)  '
            'neg: %d/%d (%.2f%%; modeled %.2f%%)  total_calls=%d  '
            'log2(med pos/neg)=%.3f',
            label, mode, thr,
            pos_stats['n_covered'], pos_stats['n'], pos_stats['coverage_pct'],
            pos_stats['coverage_modeled_pct'],
            neg_stats['n_covered'], neg_stats['n'], neg_stats['coverage_pct'],
            neg_stats['coverage_modeled_pct'],
            int(mask.sum()),
            lift_stats['log2_median_ratio_called'],
        )

        per_input.append({
            'label': label, 'path': rec['path'], 'kind': kind,
            'mode': mode, 'threshold': thr,
            'n_called': int(mask.sum()),
            'intensity_called': lift_stats,
            'pos': pos_stats, 'neg': neg_stats,
        })

    tsv_path = args.out_dir / f'{args.out_name}.tsv'
    write_tsv(per_input, tsv_path)
    log.info('Wrote %s', tsv_path)

    summary = build_summary(per_input, str(args.bins_tsv),
                             args.threshold, ref_label or None,
                             args.match_multiplier, args.top_k,
                             target_calls)
    json_path = args.out_dir / f'{args.out_name}.json'
    json_path.write_text(json.dumps(summary, indent=2, default=float) + '\n')
    log.info('Wrote %s', json_path)

    png_path = args.out_dir / f'{args.out_name}.png'
    if target_calls is not None:
        if args.top_k is not None and args.top_k > 0:
            title = f'Bin coverage at top-K (K={target_calls}; --top-k)'
        else:
            mult_suffix = (f' × {args.match_multiplier:g}'
                           if args.match_multiplier != 1.0 else '')
            title = (f'Bin coverage at top-K (K={target_calls}; matched to '
                     f'{ref_label!r} at t={args.threshold:g}{mult_suffix})')
        y_lbl = '% of bins among each pipeline\u2019s top-K'
    else:
        title = f'Bin coverage (signal > {args.threshold:g})'
        y_lbl = f'% of bins with signal > {args.threshold:g}'
    plot_coverage_bars(per_input, png_path, title=title, y_label=y_lbl)
    log.info('Wrote %s', png_path)
    lift_png_path = args.out_dir / f'{args.out_name}_intensity_lift.png'
    plot_intensity_lift_bars(per_input, lift_png_path)
    log.info('Wrote %s', lift_png_path)

    return 0


if __name__ == '__main__':
    sys.exit(main())
