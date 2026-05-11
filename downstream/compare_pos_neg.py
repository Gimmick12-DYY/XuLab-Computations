#!/usr/bin/env python
# -----------------------------------------------------------------------------
# compare_pos_neg.py
#
# Companion to prepare_pos_neg_bins.R. For each input matrix (raw mm, factored
# .npz, dense .npy, or legacy HDF5), compute per-bin total signal at the
# positive and negative bins and report:
#   * AUROC of "is positive bin" classification using per-bin total signal
#   * Kolmogorov-Smirnov statistic + p-value (pos vs neg distribution)
#   * Median signal in pos vs neg + log2(median_pos / median_neg)
# Writes a multi-panel ECDF figure (one panel per input), a long-form TSV of
# per-bin signals, and a JSON summary. By default, also emits a "modeled-only"
# view restricted to bins that lie inside the modeled-region universe of any
# imputed input (since the full-universe view is dominated by zero-padding
# outside the modeled regions).
#
# Input formats (auto-detected from the path):
#   * directory with matrix.mtx.gz       -> raw observed counts (mm)
#   * directory with factors.npz         -> factored output (cisTopic / scOpen)
#                                           => signal = (W[i] @ H.sum(axis=1))
#                                                       * per_cell_factor
#   * directory with matrix.npy          -> dense imputed matrix (FITS / MAGIC)
#   * directory with matrix_csr.npz    -> CSR aligned to full mm rows (PUscOpen)
#   * file ending in .h5 / .hdf5         -> legacy padded HDF5 with /Prc
#
# All non-mm formats are matched by region NAME (chr:start-end) via regions.tsv;
# bins whose names are missing from the modeled regions get signal=0.
# CSR full-mm inputs list every mm bin; ``modeled`` is True only for rows with
# stored nonzeros (actually imputed), not for all-zero padding rows.
#
# Usage:
#   python compare_pos_neg.py \
#     --bins-tsv  /work/.../downstream/bins/pos_neg_bins.tsv \
#     --out-dir   /work/.../downstream/pos_neg \
#     --input raw=/work/.../mm \
#     --input cisTopic=/work/.../cistopic_ctcf/impute \
#     --input FITS=/work/.../FITS/work/ctcf/impute \
#     --input scOpen=/work/.../scOpen/work/ctcf/impute
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
from scipy import sparse  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

log = logging.getLogger('compare_pos_neg')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


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


def load_bins_tsv(path: Path) -> tuple[np.ndarray, np.ndarray,
                                        list[str], list[str]]:
    """Return (pos_idx_orig, neg_idx_orig, pos_names, neg_names)."""
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
# Per-input loaders. Each returns (signal, modeled_mask) where:
#   - signal is shape (n_query,) float64 of per-bin total over cells
#   - modeled_mask is shape (n_query,) bool: True if the bin was in this
#     input's modeled region universe.
# -----------------------------------------------------------------------------


def per_bin_total_from_mm(mm_dir: Path, bin_idx: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray]:
    mtx_path = mm_dir / 'matrix.mtx.gz'
    if not mtx_path.exists():
        raise SystemExit(f'Missing {mtx_path}')
    log.info('  loading mm matrix %s', mtx_path)
    mat = sio.mmread(str(mtx_path)).tocsr()
    n_rows = mat.shape[0]
    if bin_idx.max() >= n_rows:
        raise SystemExit(
            f'bin_idx max ({int(bin_idx.max())}) out of range for mm with '
            f'{n_rows} rows. Did the mm and bins TSV come from the same input?')
    signal = np.asarray(mat[bin_idx].sum(axis=1)).ravel().astype(np.float64)
    return signal, np.ones(bin_idx.size, dtype=bool)


def _build_name_to_row(regions_path: Path) -> dict[str, int]:
    if not regions_path.exists():
        raise SystemExit(f'Missing {regions_path}')
    name_to_row: dict[str, int] = {}
    for i, line in enumerate(regions_path.read_text().splitlines()):
        if line:
            name_to_row[line] = i
    return name_to_row


def per_bin_total_from_factored(impute_dir: Path, bin_names: list[str]
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin total from a factored .npz: signal = (W[i] @ H.sum(axis=1)) * f."""
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
    log.info('  W=%s  H=%s  per_cell_factor=%.4g', W.shape, H.shape, per_cell_factor)

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
    """Per-bin total from a dense matrix.npy + regions.tsv."""
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


def per_bin_total_from_csr(impute_dir: Path, bin_names: list[str]
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin total from scipy CSR (full mm row alignment, nnz on modeled rows)."""
    csr_path = impute_dir / 'matrix_csr.npz'
    regions_path = impute_dir / 'regions.tsv'
    log.info('  loading CSR output: %s + %s', csr_path, regions_path)
    sm = sparse.load_npz(csr_path)
    name_to_row = _build_name_to_row(regions_path)

    wanted_rows: list[int] = []
    name_to_query: list[int] = []
    for k, name in enumerate(bin_names):
        row = name_to_row.get(name)
        if row is not None:
            wanted_rows.append(row)
            name_to_query.append(k)
    if not wanted_rows:
        return (np.zeros(len(bin_names), dtype=np.float64),
                np.zeros(len(bin_names), dtype=bool))
    wanted = np.asarray(wanted_rows, dtype=np.int64)
    sub = sm[wanted]
    bin_totals = np.asarray(sub.sum(axis=1)).ravel().astype(np.float64)
    # Rows exist in the full-mm matrix for every benchmark bin, but most are
    # explicit all-zero rows (no nnz). Count as "modeled" only if scOpen wrote
    # mass into that row — otherwise the union mask would cover the whole
    # panel and modeled-only AUROC would duplicate the full-universe view.
    row_nnz = np.asarray(sub.getnnz(axis=1)).ravel().astype(np.int32)

    signal = np.zeros(len(bin_names), dtype=np.float64)
    modeled = np.zeros(len(bin_names), dtype=bool)
    for ti, q in enumerate(name_to_query):
        signal[q] = bin_totals[ti]
        modeled[q] = bool(row_nnz[ti] > 0)
    return signal, modeled


def per_bin_total_from_hdf5(h5_path: Path, bin_idx: np.ndarray,
                             chunk: int = 50_000
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Legacy fallback: padded HDF5 with /Prc indexed by full mm row."""
    log.info('  loading legacy Prc from %s', h5_path)
    sort_order = np.argsort(bin_idx, kind='stable')
    sorted_idx = bin_idx[sort_order]
    uniq, inv = np.unique(sorted_idx, return_inverse=True)
    out_uniq = np.zeros(uniq.size, dtype=np.float64)
    with h5py.File(h5_path, 'r') as h5:
        ds = h5['Prc']
        if uniq.max() >= ds.shape[0]:
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
    """Return one of: 'mm', 'factored', 'dense', 'hdf5', 'hdf5_in_dir'."""
    if path.is_file():
        if path.suffix.lower() in ('.h5', '.hdf5'):
            return 'hdf5'
        raise SystemExit(f'Unsupported input file: {path}')
    if path.is_dir():
        if (path / 'matrix_csr.npz').exists() and (path / 'regions.tsv').exists():
            return 'csr_full'
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
    elif kind == 'csr_full':
        sig, mod = per_bin_total_from_csr(path, bin_names)
    elif kind == 'factored':
        sig, mod = per_bin_total_from_factored(path, bin_names)
    elif kind == 'dense':
        sig, mod = per_bin_total_from_dense(path, bin_names)
    elif kind == 'hdf5':
        sig, mod = per_bin_total_from_hdf5(path, bin_idx)
    elif kind == 'hdf5_in_dir':
        cand = sorted(list(path.glob('*.h5')) + list(path.glob('*.hdf5')))
        log.info('  treating %s as legacy HDF5 dir; using %s', path, cand[0])
        sig, mod = per_bin_total_from_hdf5(cand[0], bin_idx)
        kind = 'hdf5'
    else:  # pragma: no cover
        raise SystemExit(f'Unhandled input kind {kind} for {label}={path}')
    return sig, mod, kind


def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    if y.size == 0 or y.min() == y.max():
        return float('nan')
    return float(roc_auc_score(y, s))


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
    ap.add_argument('--out-name', type=str, default='compare_pos_neg')
    ap.add_argument('--ecdf-log', action='store_true', default=True,
                    help='Plot ECDF on log1p x-axis (default).')
    ap.add_argument('--no-ecdf-log', dest='ecdf_log', action='store_false')
    ap.add_argument('--no-modeled-view', dest='modeled_view', action='store_false',
                    default=True,
                    help='Disable the additional "modeled-only" report.')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info('Loading bin indices from %s', args.bins_tsv)
    pos_idx, neg_idx, pos_nm, neg_nm = load_bins_tsv(args.bins_tsv)
    log.info('Loaded %d positives, %d negatives', pos_idx.size, neg_idx.size)

    bin_idx = np.concatenate([pos_idx, neg_idx])
    bin_names = pos_nm + neg_nm
    y_true = np.concatenate([np.ones(pos_idx.size,  dtype=np.int8),
                             np.zeros(neg_idx.size, dtype=np.int8)])

    per_input: dict[str, np.ndarray] = {}
    per_input_modeled: dict[str, np.ndarray] = {}
    per_input_kind: dict[str, str] = {}
    for label, path in args.inputs:
        log.info('Input %r -> %s', label, path)
        signal, modeled, kind = per_bin_total(label, path, bin_idx, bin_names)
        per_input[label] = signal
        per_input_modeled[label] = modeled
        per_input_kind[label] = kind
        log.info('  kind=%s  modeled=%d/%d (%.1f%%)  signal[mean=%.4g, max=%.4g]',
                 kind, int(modeled.sum()), modeled.size,
                 100.0 * modeled.mean(),
                 float(signal.mean()), float(signal.max()))

    imp_labels = [lbl for lbl, k in per_input_kind.items() if k != 'mm']
    if imp_labels:
        modeled_mask = np.zeros(bin_idx.size, dtype=bool)
        for lbl in imp_labels:
            modeled_mask |= per_input_modeled[lbl]
    else:
        modeled_mask = np.ones(bin_idx.size, dtype=bool)
    log.info('Modeled mask (union over %s): %d / %d (%.1f%%) bins',
             imp_labels, int(modeled_mask.sum()), modeled_mask.size,
             100.0 * modeled_mask.mean())

    tsv_path = args.out_dir / f'{args.out_name}.tsv'
    log.info('Writing %s', tsv_path)
    with open(tsv_path, 'w') as fh:
        cols = ['input', 'group', 'bin_idx_orig', 'bin_name',
                'modeled_in_input', 'in_modeled_union', 'signal']
        fh.write('\t'.join(cols) + '\n')
        for label, sig in per_input.items():
            mod_in = per_input_modeled[label]
            for k in range(bin_idx.size):
                grp = 'pos' if k < pos_idx.size else 'neg'
                fh.write(
                    f'{label}\t{grp}\t{int(bin_idx[k])}\t{bin_names[k]}\t'
                    f'{int(mod_in[k])}\t{int(modeled_mask[k])}\t{sig[k]:.6g}\n'
                )

    def _summarize_view(view_label: str, mask: np.ndarray) -> list[dict]:
        out: list[dict] = []
        for label, sig in per_input.items():
            sig_v = sig[mask]
            y_v = y_true[mask]
            pos_sig = sig_v[y_v == 1]
            neg_sig = sig_v[y_v == 0]
            ks = ks_2samp(pos_sig, neg_sig) if (pos_sig.size and neg_sig.size) else None
            out.append({
                'label':           label,
                'view':            view_label,
                'kind':            per_input_kind[label],
                'n_pos':           int(pos_sig.size),
                'n_neg':           int(neg_sig.size),
                'auroc':           safe_auroc(y_v, sig_v),
                'ks_statistic':    float(ks.statistic) if ks is not None else float('nan'),
                'ks_pvalue':       float(ks.pvalue)    if ks is not None else float('nan'),
                'pos_median':      float(np.median(pos_sig)) if pos_sig.size else float('nan'),
                'neg_median':      float(np.median(neg_sig)) if neg_sig.size else float('nan'),
                'pos_mean':        float(pos_sig.mean())     if pos_sig.size else float('nan'),
                'neg_mean':        float(neg_sig.mean())     if neg_sig.size else float('nan'),
                'pos_zero_frac':   float((pos_sig == 0).mean()) if pos_sig.size else float('nan'),
                'neg_zero_frac':   float((neg_sig == 0).mean()) if neg_sig.size else float('nan'),
                'log2_median_ratio': float(
                    np.log2(max(np.median(pos_sig) if pos_sig.size else 0.0, 1e-30)
                            / max(np.median(neg_sig) if neg_sig.size else 0.0, 1e-30))
                ) if (pos_sig.size and neg_sig.size) else float('nan'),
            })
        return out

    def _emit_figs_and_json(view_label: str, mask: np.ndarray, suffix: str,
                             rows: list[dict]) -> None:
        json_path = args.out_dir / f'{args.out_name}{suffix}.json'
        summary = {
            'bins_tsv': str(args.bins_tsv),
            'view':     view_label,
            'n_pos':    int(mask[:pos_idx.size].sum()),
            'n_neg':    int(mask[pos_idx.size:].sum()),
            'inputs':   rows,
        }
        json_path.write_text(json.dumps(summary, indent=2) + '\n')
        log.info('Wrote %s', json_path)

        n = len(per_input)
        cols_n = min(n, 3)
        rows_n = (n + cols_n - 1) // cols_n
        fig, axes = plt.subplots(rows_n, cols_n,
                                  figsize=(5 * cols_n, 4 * rows_n),
                                  squeeze=False)
        flat = axes.flatten()
        POS_COLOR = '#E41A1C'
        NEG_COLOR = '#377EB8'
        for ax_idx, (label, sig) in enumerate(per_input.items()):
            ax = flat[ax_idx]
            sig_v = sig[mask]
            y_v = y_true[mask]
            pos_sig = sig_v[y_v == 1]
            neg_sig = sig_v[y_v == 0]
            if pos_sig.size == 0 or neg_sig.size == 0:
                ax.set_title(f'{label}\n(empty subset)')
                ax.axis('off')
                continue
            if args.ecdf_log:
                x_pos = np.log1p(np.sort(pos_sig))
                x_neg = np.log1p(np.sort(neg_sig))
                xlab  = 'log1p(per-bin signal)'
            else:
                x_pos = np.sort(pos_sig); x_neg = np.sort(neg_sig)
                xlab  = 'per-bin signal'
            y_pos = np.linspace(0, 1, len(x_pos), endpoint=False)
            y_neg = np.linspace(0, 1, len(x_neg), endpoint=False)
            ax.plot(x_pos, y_pos, color=POS_COLOR, lw=1.4, label='Positive (CTCF cCRE)')
            ax.plot(x_neg, y_neg, color=NEG_COLOR, lw=1.4, label='Negative (random)')
            info = next(d for d in rows if d['label'] == label)
            ax.set_title(
                f'{label}\nAUROC={info["auroc"]:.3f}  '
                f'KS={info["ks_statistic"]:.3f} '
                f'(log2 med ratio={info["log2_median_ratio"]:+.2f})',
                fontsize=10,
            )
            ax.set_xlabel(xlab); ax.set_ylabel('CDF')
            ax.set_ylim(0, 1.0)
            ax.legend(frameon=False, fontsize=8)
        for k in range(len(per_input), len(flat)):
            flat[k].axis('off')
        n_pos_v = int(mask[:pos_idx.size].sum())
        n_neg_v = int(mask[pos_idx.size:].sum())
        fig.suptitle(
            f'Per-bin signal at CTCF positive vs negative bins '
            f'[{view_label}]  (n_pos={n_pos_v}, n_neg={n_neg_v})'
        )
        fig.tight_layout()
        png_path = args.out_dir / f'{args.out_name}{suffix}.png'
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        log.info('Wrote %s', png_path)

        fig, ax = plt.subplots(figsize=(max(4, 0.7 * len(per_input) + 2), 4))
        labels = [r['label'] for r in rows]
        aurocs = [r['auroc']  for r in rows]
        bars = ax.bar(np.arange(len(labels)), aurocs,
                      color=['#777'] * len(labels))
        for b, v in zip(bars, aurocs):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=9)
        ax.axhline(0.5, color='k', lw=0.5, ls=':')
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylim(0, 1.0); ax.set_ylabel('AUROC (positive vs negative bins)')
        ax.set_title(f'Bin-level CTCF separability across pipelines [{view_label}]')
        fig.tight_layout()
        bar_path = args.out_dir / f'{args.out_name}{suffix}_auroc.png'
        fig.savefig(bar_path, dpi=150)
        plt.close(fig)
        log.info('Wrote %s', bar_path)

    full_mask = np.ones(bin_idx.size, dtype=bool)
    full_rows = _summarize_view('full', full_mask)
    _emit_figs_and_json('full', full_mask, '', full_rows)

    for r in full_rows:
        log.info('  [full]      %-12s  AUROC=%.4f  KS=%.4f (p=%.2e)  median_pos=%.4g  median_neg=%.4g  log2_ratio=%+.3f',
                 r['label'], r['auroc'], r['ks_statistic'], r['ks_pvalue'],
                 r['pos_median'], r['neg_median'], r['log2_median_ratio'])

    if args.modeled_view and imp_labels:
        modeled_rows = _summarize_view('modeled', modeled_mask)
        _emit_figs_and_json('modeled', modeled_mask, '_modeled', modeled_rows)
        for r in modeled_rows:
            log.info('  [modeled]   %-12s  AUROC=%.4f  KS=%.4f (p=%.2e)  median_pos=%.4g  median_neg=%.4g  log2_ratio=%+.3f',
                     r['label'], r['auroc'], r['ks_statistic'], r['ks_pvalue'],
                     r['pos_median'], r['neg_median'], r['log2_median_ratio'])

        labels = [r['label'] for r in full_rows]
        full_a    = [r['auroc'] for r in full_rows]
        modeled_a = [next(m['auroc'] for m in modeled_rows if m['label'] == lbl)
                      for lbl in labels]
        x = np.arange(len(labels))
        width = 0.4
        fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(labels) + 2), 4))
        b1 = ax.bar(x - width/2, full_a,    width, label='full universe',     color='#999')
        b2 = ax.bar(x + width/2, modeled_a, width, label='modeled-only',      color='#3B7EA1')
        for bars_grp, vals in ((b1, full_a), (b2, modeled_a)):
            for b, v in zip(bars_grp, vals):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.3f}',
                            ha='center', va='bottom', fontsize=8)
        ax.axhline(0.5, color='k', lw=0.5, ls=':')
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylim(0, 1.0); ax.set_ylabel('AUROC')
        ax.set_title('Bin-level CTCF separability: full universe vs modeled-only')
        ax.legend(frameon=False)
        fig.tight_layout()
        cmp_path = args.out_dir / f'{args.out_name}_auroc_full_vs_modeled.png'
        fig.savefig(cmp_path, dpi=150)
        plt.close(fig)
        log.info('Wrote %s', cmp_path)

    return 0


if __name__ == '__main__':
    sys.exit(main())
