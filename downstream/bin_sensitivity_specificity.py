#!/usr/bin/env python
# -----------------------------------------------------------------------------
# bin_sensitivity_specificity.py
#
# **Bin-level** sensitivity / specificity using the cCRE-derived ground truth
# from prepare_pos_neg_bins.R (the same positive / negative bin set that
# `Bin_posVSneg.ipynb` produces). For each imputation pipeline:
#
#   1. Compute per-bin signal at the positive bins and the negative bins
#      (sum across cells, exactly like compare_pos_neg.py).
#   2. Treat each bin as one classification example:
#         pos bin with signal > t   -> TP
#         pos bin with signal <= t  -> FN
#         neg bin with signal > t   -> FP
#         neg bin with signal <= t  -> TN
#      so   sensitivity = TP / n_pos,    specificity = TN / n_neg.
#   3. Sweep thresholds and report sens / spec / precision / F1 / MCC, plus
#      AUROC / AUPR and two natural operating points:
#         * max-F1 threshold
#         * Youden J = max(sens + spec - 1)
#
# This is the bin-level analogue of sensitivity_specificity.py (which uses raw
# observed counts as ground truth across all entries). The two are complementary:
#   * `sensitivity_specificity.py` -- "does the imputer recover what raw saw?"
#   * `bin_sensitivity_specificity.py` -- "does the imputer separate true CTCF
#     regulatory bins from background?"
#
# Each --input PATH is one of:
#   * directory with matrix.mtx.gz       (raw counts; e.g. <work>/mm)
#   * directory with factors.npz + regions.tsv   (cisTopic / scOpen)
#   * directory with matrix.npy  + regions.tsv   (FITS / MAGIC)
#   * legacy HDF5 file with /Prc                 (older runs)
#
# Usage:
#   python bin_sensitivity_specificity.py \
#     --bins-tsv /work/.../downstream/bins/pos_neg_bins.tsv \
#     --out-dir  /work/.../downstream/bin_sensitivity_specificity \
#     --input raw=/work/.../mm \
#     --input cisTopic=/work/.../cistopic_ctcf/impute \
#     --input FITS=/work/.../FITS/work/ctcf/impute \
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
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve  # noqa: E402

log = logging.getLogger('bin_sens_spec')
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
# Per-bin total-signal loaders (mirror compare_pos_neg.py)
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
    raise SystemExit(f'Cannot determine input kind from {path}')


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
# Threshold sweep + metrics
# -----------------------------------------------------------------------------
def auto_thresholds(signal: np.ndarray, n_thresholds: int) -> np.ndarray:
    """Per-pipeline quantile-based threshold grid over the union of pos+neg signals.

    Includes 0.0 as a top-right ROC anchor, plus the inf anchor in the
    threshold sweep code itself.
    """
    s = signal[~np.isnan(signal)]
    if s.size == 0:
        return np.array([0.0])
    qs = np.concatenate([
        np.linspace(0.001, 0.5,  n_thresholds // 4, endpoint=False),
        np.linspace(0.5,   0.9,  n_thresholds // 4, endpoint=False),
        np.linspace(0.9,   0.99, n_thresholds // 4, endpoint=False),
        np.linspace(0.99,  0.999, n_thresholds - 3 * (n_thresholds // 4), endpoint=True),
    ])
    thr = np.quantile(s, np.clip(qs, 0.0, 1.0))
    thr = np.concatenate([[0.0], thr])
    return np.unique(thr)


def sweep_metrics(y_true: np.ndarray, signal: np.ndarray,
                   thresholds: np.ndarray) -> dict[str, np.ndarray]:
    """For each threshold t, compute TP/FP/FN/TN treating signal > t as pred-positive."""
    pos_mask = y_true.astype(bool)
    pos_total = int(pos_mask.sum())
    neg_total = int((~pos_mask).sum())

    T = len(thresholds)
    TP = np.empty(T, dtype=np.int64)
    FP = np.empty(T, dtype=np.int64)
    for i, t in enumerate(thresholds):
        pred = signal > t
        tp = int(np.sum(pred & pos_mask))
        fp = int(np.sum(pred & ~pos_mask))
        TP[i] = tp; FP[i] = fp
    FN = pos_total - TP
    TN = neg_total - FP

    eps = 1e-30
    sens = TP / max(pos_total, 1)
    spec = TN / max(neg_total, 1)
    prec = np.where(TP + FP > 0, TP / np.maximum(TP + FP, eps), 0.0)
    f1   = np.where(prec + sens > 0, 2 * prec * sens / np.maximum(prec + sens, eps), 0.0)
    acc  = (TP + TN) / max(pos_total + neg_total, 1)
    fpr  = 1.0 - spec
    num  = TP.astype(np.float64) * TN - FP.astype(np.float64) * FN
    denom = np.sqrt((TP + FP).astype(np.float64) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = np.where(denom > 0, num / np.maximum(denom, eps), 0.0)
    youden = sens + spec - 1.0

    return {
        'thresholds': np.asarray(thresholds, dtype=np.float64),
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'sens': sens, 'spec': spec, 'prec': prec,
        'f1': f1, 'mcc': mcc, 'acc': acc, 'fpr': fpr, 'tpr': sens,
        'youden': youden,
        'pos_total': pos_total, 'neg_total': neg_total,
    }


def operating_points(metrics: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Pick max-F1 and max-Youden thresholds; return their (t, sens, spec, F1, MCC)."""
    out: dict[str, dict[str, float]] = {}
    thr = metrics['thresholds']
    if not thr.size:
        return out
    bf = int(np.argmax(metrics['f1']))
    by = int(np.argmax(metrics['youden']))
    for name, idx in (('max_f1', bf), ('youden_j', by)):
        out[name] = {
            'threshold':   float(thr[idx]),
            'sensitivity': float(metrics['sens'][idx]),
            'specificity': float(metrics['spec'][idx]),
            'precision':   float(metrics['prec'][idx]),
            'f1':          float(metrics['f1'][idx]),
            'mcc':         float(metrics['mcc'][idx]),
            'youden_j':    float(metrics['youden'][idx]),
            'tp':          int(metrics['TP'][idx]),
            'fp':          int(metrics['FP'][idx]),
            'fn':          int(metrics['FN'][idx]),
            'tn':          int(metrics['TN'][idx]),
        }
    return out


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def _color_cycle(n: int) -> list[str]:
    cmap = plt.get_cmap('tab10' if n <= 10 else 'tab20')
    return [cmap(i % cmap.N) for i in range(n)]


def plot_roc(per_input: list[dict], view: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], color='k', lw=0.5, ls=':')
    colors = _color_cycle(len(per_input))
    for info, c in zip(per_input, colors):
        v = info['views'].get(view)
        if not v or v['y_true'].size == 0:
            continue
        fpr, tpr, _ = roc_curve(v['y_true'], v['signal'])
        ax.plot(fpr, tpr, lw=1.4, color=c,
                label=f'{info["label"]}  AUROC={v["auroc"]:.3f}')
    ax.set_xlabel('False positive rate (1 − specificity)')
    ax.set_ylabel('True positive rate (sensitivity)')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_title(f'Bin-level ROC vs cCRE ground truth  [{view}]')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_pr(per_input: list[dict], view: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    colors = _color_cycle(len(per_input))
    for info, c in zip(per_input, colors):
        v = info['views'].get(view)
        if not v or v['y_true'].size == 0:
            continue
        prec, rec, _ = precision_recall_curve(v['y_true'], v['signal'])
        ax.plot(rec, prec, lw=1.4, color=c,
                label=f'{info["label"]}  AUPR={v["aupr"]:.3f}')
    ax.set_xlabel('Recall (sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_title(f'Bin-level precision-recall vs cCRE ground truth  [{view}]')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_sens_spec_vs_threshold(per_input: list[dict], view: str,
                                  out_path: Path) -> None:
    n = len(per_input)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows),
                              squeeze=False)
    flat = axes.flatten()
    for i, info in enumerate(per_input):
        ax = flat[i]
        v = info['views'].get(view)
        if not v or v['y_true'].size == 0:
            ax.set_title(f'{info["label"]}  (empty subset)')
            ax.axis('off'); continue
        m = v['metrics']
        thr = m['thresholds']
        ax.plot(thr, m['sens'], color='#c33', lw=1.4, label='sensitivity')
        ax.plot(thr, m['spec'], color='#33c', lw=1.4, label='specificity')
        ax.plot(thr, m['f1'],   color='#888', lw=1.0, ls='--', label='F1')
        try:
            min_pos = thr[thr > 0].min() if (thr > 0).any() else 1e-6
            ax.set_xscale('symlog', linthresh=max(min_pos, 1e-6))
        except Exception:
            pass
        ops = v['op_points']
        if 'max_f1' in ops:
            ax.axvline(ops['max_f1']['threshold'], color='k', lw=0.7, ls=':',
                       label=f'max-F1 t={ops["max_f1"]["threshold"]:.3g}')
        if 'youden_j' in ops:
            ax.axvline(ops['youden_j']['threshold'], color='gray', lw=0.7, ls='-.',
                       label=f'Youden J t={ops["youden_j"]["threshold"]:.3g}')
        ax.set_xlabel('per-bin signal threshold')
        ax.set_ylim(0, 1.02)
        ax.set_title(f'{info["label"]}  AUROC={v["auroc"]:.3f}  AUPR={v["aupr"]:.3f}',
                     fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    for j in range(len(per_input), len(flat)):
        flat[j].axis('off')
    fig.suptitle(f'Bin-level sens / spec / F1 vs threshold  [{view}]')
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_metric_bars(per_input: list[dict], view: str, out_path: Path) -> None:
    labels = [i['label'] for i in per_input]
    def _g(info: dict, key: str, fallback=float('nan')) -> float:
        v = info['views'].get(view)
        if not v: return fallback
        if key in v: return float(v[key])
        op = v['op_points'].get('max_f1')
        return float(op[key]) if (op and key in op) else fallback
    auroc = [_g(i, 'auroc') for i in per_input]
    aupr  = [_g(i, 'aupr')  for i in per_input]
    f1max = [_g(i, 'f1')    for i in per_input]
    sens  = [_g(i, 'sensitivity') for i in per_input]
    spec  = [_g(i, 'specificity') for i in per_input]
    x = np.arange(len(labels))
    width = 0.16
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels) + 3), 4))
    bars = []
    bars.append(ax.bar(x - 2*width, auroc, width, label='AUROC',           color='#3B7EA1'))
    bars.append(ax.bar(x -   width, aupr,  width, label='AUPR',            color='#888'))
    bars.append(ax.bar(x,           f1max, width, label='max F1',          color='#C44E52'))
    bars.append(ax.bar(x +   width, sens,  width, label='sens @ max-F1',   color='#55A868'))
    bars.append(ax.bar(x + 2*width, spec,  width, label='spec @ max-F1',   color='#CCB974'))
    for grp, vals in zip(bars, (auroc, aupr, f1max, sens, spec)):
        for b, val in zip(grp, vals):
            if not np.isnan(val):
                ax.text(b.get_x() + b.get_width() / 2, val + 0.01, f'{val:.3f}',
                        ha='center', va='bottom', fontsize=7)
    ax.axhline(0.5, color='k', lw=0.4, ls=':')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylim(0, 1.05); ax.set_ylabel('metric')
    ax.set_title(f'Bin-level sens / spec summary  [{view}]')
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------
def write_tsv(per_input: list[dict], out_path: Path) -> None:
    cols = ['input', 'view', 'kind', 'threshold',
            'TP', 'FP', 'FN', 'TN',
            'sensitivity', 'specificity', 'precision',
            'F1', 'MCC', 'accuracy', 'youden_J']
    with open(out_path, 'w') as fh:
        fh.write('\t'.join(cols) + '\n')
        for info in per_input:
            for view, v in info['views'].items():
                if not v or v['y_true'].size == 0:
                    continue
                m = v['metrics']
                thr = m['thresholds']
                for k in range(thr.size):
                    fh.write('\t'.join([
                        info['label'], view, info['kind'],
                        f'{thr[k]:.6g}',
                        str(int(m['TP'][k])),
                        str(int(m['FP'][k])),
                        str(int(m['FN'][k])),
                        str(int(m['TN'][k])),
                        f"{m['sens'][k]:.6g}",
                        f"{m['spec'][k]:.6g}",
                        f"{m['prec'][k]:.6g}",
                        f"{m['f1'][k]:.6g}",
                        f"{m['mcc'][k]:.6g}",
                        f"{m['acc'][k]:.6g}",
                        f"{m['youden'][k]:.6g}",
                    ]) + '\n')


def build_summary(per_input: list[dict], bins_tsv: str,
                   n_pos: int, n_neg: int) -> dict:
    out_inputs: list[dict] = []
    for info in per_input:
        for view, v in info['views'].items():
            if not v: continue
            out_inputs.append({
                'label':       info['label'],
                'view':        view,
                'kind':        info['kind'],
                'n_pos':       int(v['y_true'].sum()) if v['y_true'].size else 0,
                'n_neg':       int((v['y_true'] == 0).sum()) if v['y_true'].size else 0,
                'auroc':       float(v.get('auroc', float('nan'))),
                'aupr':        float(v.get('aupr',  float('nan'))),
                'op_points':   v['op_points'],
            })
    return {
        'bins_tsv': bins_tsv,
        'n_pos_total': int(n_pos),
        'n_neg_total': int(n_neg),
        'inputs': out_inputs,
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
    ap.add_argument('--out-name', type=str, default='bin_sensitivity_specificity')
    ap.add_argument('--n-thresholds', type=int, default=64,
                    help='Number of auto thresholds per pipeline (default 64).')
    ap.add_argument('--thresholds', type=str, default=None,
                    help='Comma-separated absolute thresholds. Default: auto '
                         '(per-pipeline quantile-based over the union of pos+neg signals).')
    ap.add_argument('--no-modeled-view', dest='modeled_view', action='store_false',
                    default=True,
                    help='Skip the "modeled-only" view.')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info('Loading bin indices from %s', args.bins_tsv)
    pos_idx, neg_idx, pos_nm, neg_nm = load_bins_tsv(args.bins_tsv)
    log.info('Loaded %d positives, %d negatives', pos_idx.size, neg_idx.size)

    bin_idx = np.concatenate([pos_idx, neg_idx])
    bin_names = pos_nm + neg_nm
    y_true = np.concatenate([np.ones(pos_idx.size,  dtype=np.int8),
                             np.zeros(neg_idx.size, dtype=np.int8)])

    user_thr: np.ndarray | None = None
    if args.thresholds:
        ts = sorted(float(t.strip()) for t in args.thresholds.split(',') if t.strip())
        if ts:
            user_thr = np.asarray(ts, dtype=np.float64)
            log.info('Using %d user-supplied thresholds across all pipelines', user_thr.size)

    per_input: list[dict] = []
    for label, path in args.inputs:
        log.info('Input %r -> %s', label, path)
        signal, modeled, kind = per_bin_total(label, path, bin_idx, bin_names)
        log.info('  kind=%s  modeled=%d/%d (%.1f%%)  signal[mean=%.4g, max=%.4g]',
                 kind, int(modeled.sum()), modeled.size,
                 100.0 * modeled.mean(),
                 float(signal.mean()), float(signal.max()))

        views: dict[str, dict] = {}

        # FULL view: all positive + negative bins; unmodeled => signal already 0.
        full_thr = user_thr if user_thr is not None else auto_thresholds(
            signal, args.n_thresholds
        )
        if y_true.size and signal.size:
            try:
                auroc = float(roc_auc_score(y_true, signal))
            except Exception:
                auroc = float('nan')
            try:
                aupr = float(average_precision_score(y_true, signal))
            except Exception:
                aupr = float('nan')
        else:
            auroc = aupr = float('nan')
        full_metrics = sweep_metrics(y_true, signal, full_thr)
        views['full'] = {
            'y_true':    y_true,
            'signal':    signal,
            'metrics':   full_metrics,
            'auroc':     auroc,
            'aupr':      aupr,
            'op_points': operating_points(full_metrics),
        }

        # MODELED view: bins whose name is present in the imputer's modeled
        # universe (signal lookup hit the imputed regions.tsv). Only meaningful
        # for non-mm inputs; for mm we treat all bins as modeled.
        if args.modeled_view and kind != 'mm':
            mod_mask = modeled.astype(bool)
            y_mod = y_true[mod_mask]
            sig_mod = signal[mod_mask]
            if y_mod.size and y_mod.min() != y_mod.max():
                m_thr = user_thr if user_thr is not None else auto_thresholds(
                    sig_mod, args.n_thresholds
                )
                try:
                    auroc_m = float(roc_auc_score(y_mod, sig_mod))
                except Exception:
                    auroc_m = float('nan')
                try:
                    aupr_m = float(average_precision_score(y_mod, sig_mod))
                except Exception:
                    aupr_m = float('nan')
                m_metrics = sweep_metrics(y_mod, sig_mod, m_thr)
                views['modeled'] = {
                    'y_true':    y_mod,
                    'signal':    sig_mod,
                    'metrics':   m_metrics,
                    'auroc':     auroc_m,
                    'aupr':      aupr_m,
                    'op_points': operating_points(m_metrics),
                    'n_in_view': int(mod_mask.sum()),
                }

        per_input.append({
            'label': label, 'path': str(path), 'kind': kind,
            'views': views,
            'signal_modeled_mask': modeled,
        })

        for view, v in views.items():
            ops = v['op_points'].get('max_f1', {})
            log.info(
                '  [%-7s] AUROC=%.4f AUPR=%.4f  max-F1=%.3f@t=%.3g  '
                'sens=%.3f spec=%.3f',
                view, v['auroc'], v['aupr'],
                ops.get('f1', float('nan')),
                ops.get('threshold', float('nan')),
                ops.get('sensitivity', float('nan')),
                ops.get('specificity', float('nan')),
            )

    tsv_path = args.out_dir / f'{args.out_name}.tsv'
    write_tsv(per_input, tsv_path)
    log.info('Wrote %s', tsv_path)

    summary = build_summary(per_input, str(args.bins_tsv), pos_idx.size, neg_idx.size)
    json_path = args.out_dir / f'{args.out_name}.json'
    json_path.write_text(json.dumps(summary, indent=2, default=float) + '\n')
    log.info('Wrote %s', json_path)

    views_to_plot = ['full']
    if args.modeled_view:
        views_to_plot.append('modeled')
    for view in views_to_plot:
        plot_roc(per_input, view, args.out_dir / f'{args.out_name}_roc_{view}.png')
        plot_pr( per_input, view, args.out_dir / f'{args.out_name}_pr_{view}.png')
        plot_sens_spec_vs_threshold(
            per_input, view,
            args.out_dir / f'{args.out_name}_sens_spec_{view}.png',
        )
        plot_metric_bars(
            per_input, view,
            args.out_dir / f'{args.out_name}_bars_{view}.png',
        )
        log.info('Wrote ROC / PR / sens-spec / bars figures for view=%s', view)

    return 0


if __name__ == '__main__':
    sys.exit(main())
