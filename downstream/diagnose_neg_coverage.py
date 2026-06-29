#!/usr/bin/env python
# -----------------------------------------------------------------------------
# diagnose_neg_coverage.py
#
# Explain a 0% (or surprisingly low) NEGATIVE peak coverage WITHOUT running
# MACS. Peak coverage can only be nonzero where the matrix actually has signal,
# so this reports the per-method "signal ceiling": the fraction of pos / neg
# reference bins that carry ANY nonzero signal (plus mean/percentiles). If the
# negative ceiling is ~0, the 0% peak coverage is fully explained -- the chosen
# negatives are signal-dead (e.g. absolute deserts: closed heterochromatin), so
# peak calling never had anything to call. Nothing downstream can recover it.
#
# It reuses peak_coverage.py's loaders, so "signal" is defined identically.
#
# Usage:
#   python downstream/diagnose_neg_coverage.py \
#     --bins-tsv  /work/.../downstream/bins/pos_neg_bins.tsv \
#     --input raw=/work/.../mm \
#     --input unified=/work/.../unified/work/ctcf/impute \
#     [--desert-ranking /work/.../downstream/bins/neg_desert_ranking.tsv]
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from peak_coverage import load_per_bin_signal, load_reference_bins  # noqa: E402


def parse_input(s: str) -> tuple[str, Path]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--input must be label=path; got {s!r}")
    label, p = s.split("=", 1)
    return label, Path(p).expanduser().resolve()


def ceiling_stats(signal_by_name: dict, names: list[str]) -> dict:
    vals = np.fromiter((signal_by_name.get(nm, 0.0) for nm in names),
                       dtype=np.float64, count=len(names))
    nz = vals[vals > 0]
    n = len(names)
    return {
        "n": n,
        "n_signal": int((vals > 0).sum()),
        "ceiling_pct": 100.0 * (vals > 0).sum() / n if n else float("nan"),
        "mean_nonzero": float(nz.mean()) if nz.size else 0.0,
        "p90_nonzero": float(np.quantile(nz, 0.9)) if nz.size else 0.0,
        "max": float(vals.max()) if n else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bins-tsv", required=True, type=Path)
    ap.add_argument("--input", action="append", required=True, type=parse_input,
                    dest="inputs", metavar="LABEL=PATH")
    ap.add_argument("--desert-ranking", type=Path, default=None,
                    help="optional neg_desert_ranking.tsv to show bulk coverage of the negatives")
    args = ap.parse_args()

    pos_bins, neg_bins = load_reference_bins(args.bins_tsv)
    pos_names = [f"{c}:{s}-{e}" for c, s, e in pos_bins]
    neg_names = [f"{c}:{s}-{e}" for c, s, e in neg_bins]
    neg_set = set(neg_names)
    print(f"Reference: {len(pos_names)} pos, {len(neg_names)} neg bins\n")

    hdr = f"{'method':<14}{'pos_ceil%':>10}{'neg_ceil%':>10}{'neg_w/sig':>11}{'neg_mean':>10}{'neg_max':>10}"
    print(hdr); print("-" * len(hdr))
    for label, path in args.inputs:
        if not path.is_dir():
            print(f"{label:<14}  (skip: {path} not a dir)"); continue
        signal, regions, kind, n_cells = load_per_bin_signal(path)
        sig_by_name = {nm: v for nm, v in zip(regions, signal) if v > 0}
        ps = ceiling_stats(sig_by_name, pos_names)
        ns = ceiling_stats(sig_by_name, neg_names)
        print(f"{label:<14}{ps['ceiling_pct']:>10.2f}{ns['ceiling_pct']:>10.2f}"
              f"{ns['n_signal']:>11d}{ns['mean_nonzero']:>10.3g}{ns['max']:>10.3g}")

    if args.desert_ranking and args.desert_ranking.is_file():
        # Show how dead the chosen negatives are in the bulk reference itself.
        import csv
        cov_sel, cov_all = [], []
        with open(args.desert_ranking) as fh:
            r = csv.DictReader(fh, delimiter="\t")
            for row in r:
                try:
                    cov = float(row["bulk_ctcf_coverage"])
                except (KeyError, ValueError):
                    continue
                cov_all.append(cov)
                nm = row.get("bin_name", "")
                if nm in neg_set or str(row.get("selected", "")).upper() in ("TRUE", "1"):
                    cov_sel.append(cov)
        if cov_sel:
            cov_sel = np.array(cov_sel); cov_all = np.array(cov_all)
            print(f"\nBulk CTCF coverage of the {len(cov_sel)} chosen negatives:")
            print(f"  selected: median={np.median(cov_sel):.3g}  mean={cov_sel.mean():.3g}  "
                  f"max={cov_sel.max():.3g}  (==0: {int((cov_sel==0).sum())})")
            print(f"  all candidates: median={np.median(cov_all):.3g}  mean={cov_all.mean():.3g}")

    print("\nReading: if neg_ceil% ~ 0, the negatives carry no matrix signal -> "
          "peak coverage is 0 by construction (signal-dead deserts), not a bug.\n"
          "Fix: use representative negatives (NEGATIVE_MODE=all_candidates, or "
          "N_POS/N_NEG balanced) instead of the extreme desert tail.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
