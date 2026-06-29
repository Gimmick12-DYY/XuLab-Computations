#!/usr/bin/env python
# -----------------------------------------------------------------------------
# recompute_metrics.py
#
# FAST re-scoring when only the pos/neg BIN DEFINITION changed (not the imputed
# matrices). It REUSES the peaks already called by a previous peak_coverage run
# (the slow MACS3 step) and only recomputes the bin-dependent metrics:
#   - Pos/Neg signal ceiling   (matrix signal at the new bins)
#   - Pos/Neg in-peaks coverage (bedtools intersect: NEW bins vs EXISTING peaks)
#   - AUROC                     (matrix signal at the new bins)
#
# This skips synthetic-BED generation + MACS3 callpeak entirely, so it runs in
# seconds-to-minutes instead of the full peak_coverage time.
#
# Reuses peak_coverage.py's loaders/formatter so definitions match exactly.
#
# Usage:
#   python downstream/recompute_metrics.py \
#     --bins-tsv  /work/.../downstream/bins/pos_neg_bins.tsv \         # the NEW bins
#     --peaks-dir /work/.../downstream/peak_coverage_ctcf \            # prior peak_coverage out-dir
#     --input raw=/work/.../mm \
#     --input unified=/work/.../unified/work/ctcf/impute \
#     --out-dir /work/.../downstream/peak_coverage_ctcf \
#     --out-name peak_coverage_recompute
#
# Requires: bedtools (peak calling is NOT re-run). The <peaks-dir>/<label>/
# <label>_called_peaks*.bed files must already exist from a prior peak_coverage.
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from peak_coverage import (  # noqa: E402
    load_per_bin_signal, load_reference_bins, signal_auroc,
    coverage_via_bedtools, write_ref_bed, format_summary_table,
)


def parse_input(s: str) -> tuple[str, Path]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--input must be label=path; got {s!r}")
    label, p = s.split("=", 1)
    return label, Path(p).expanduser().resolve()


def find_called_peaks(peaks_dir: Path, label: str) -> Path | None:
    """Locate the called-peaks BED a previous peak_coverage wrote for `label`."""
    d = peaks_dir / label
    exact = d / f"{label}_called_peaks.bed"
    if exact.is_file():
        return exact
    # sweep runs tag the file (e.g. _q0.05_bg1_ext200) -- take the first match.
    hits = sorted(d.glob(f"{label}_called_peaks*.bed"))
    return hits[0] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bins-tsv", required=True, type=Path)
    ap.add_argument("--peaks-dir", required=True, type=Path,
                    help="A prior peak_coverage out-dir containing <label>/<label>_called_peaks*.bed")
    ap.add_argument("--input", action="append", required=True, type=parse_input,
                    dest="inputs", metavar="LABEL=PATH")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--out-name", default="peak_coverage_recompute")
    ap.add_argument("--macs-q", type=float, default=0.05, help="for the table header only")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pos_bins, neg_bins = load_reference_bins(args.bins_tsv)
    pos_names = {f"{c}:{s}-{e}" for c, s, e in pos_bins}
    neg_names = {f"{c}:{s}-{e}" for c, s, e in neg_bins}
    ref_pos_bed = args.out_dir / "_ref_pos.bed"
    ref_neg_bed = args.out_dir / "_ref_neg.bed"
    write_ref_bed(pos_bins, ref_pos_bed)
    write_ref_bed(neg_bins, ref_neg_bed)

    rows: list[dict] = []
    for label, path in args.inputs:
        called = find_called_peaks(args.peaks_dir, label)
        if called is None:
            print(f"[recompute] {label}: no called-peaks BED under {args.peaks_dir/label} "
                  f"-- run full peak_coverage once first. Skipping.", file=sys.stderr)
            continue
        signal, regions, kind, n_cells = load_per_bin_signal(path)
        nnz = int(np.count_nonzero(signal))
        total_signal = float(signal.sum())
        mean_umi = total_signal / n_cells if n_cells else float("nan")

        sig_by_name = {nm: v for nm, v in zip(regions, signal) if v > 0}
        pos_scores = np.fromiter((sig_by_name.get(nm, 0.0) for nm in pos_names),
                                 dtype=np.float64, count=len(pos_names))
        neg_scores = np.fromiter((sig_by_name.get(nm, 0.0) for nm in neg_names),
                                 dtype=np.float64, count=len(neg_names))
        n_pos_sig = int(np.count_nonzero(pos_scores))
        n_neg_sig = int(np.count_nonzero(neg_scores))
        auroc = signal_auroc(pos_scores, neg_scores)

        n_pos_cov, n_pos = coverage_via_bedtools(ref_pos_bed, called)
        n_neg_cov, n_neg = coverage_via_bedtools(ref_neg_bed, called)
        n_peaks = sum(1 for ln in called.read_text().splitlines() if ln.strip())

        rows.append({
            "run": label,
            "n_nonzero_bins": nnz,
            "mean_umi_per_cell": round(mean_umi, 6),
            "pos_signal_ceiling_pct": round(100.0 * n_pos_sig / max(len(pos_names), 1), 4),
            "neg_signal_ceiling_pct": round(100.0 * n_neg_sig / max(len(neg_names), 1), 4),
            "positive_peak_coverage_pct": round(100.0 * n_pos_cov / n_pos, 4) if n_pos else float("nan"),
            "negative_peak_coverage_pct": round(100.0 * n_neg_cov / n_neg, 4) if n_neg else float("nan"),
            "n_called_peaks": int(n_peaks),
            "auroc": round(auroc, 6),
            "reused_peaks": str(called),
        })
        print(f"[recompute] {label}: reused {called.name} ({n_peaks} peaks)")

    if not rows:
        print("[recompute] no rows produced (no reusable peaks found).", file=sys.stderr)
        return 1

    tsv = args.out_dir / f"{args.out_name}.tsv"
    cols = list(rows[0].keys())
    with open(tsv, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")
    table = format_summary_table(rows, args.macs_q)
    (args.out_dir / f"{args.out_name}_table.txt").write_text(table + "\n")
    print(f"\n{table}\n\nWrote {tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
