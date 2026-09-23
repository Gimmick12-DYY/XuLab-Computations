#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# CTCF raw vs imputed coverage at the 38,102 consensus ChIP peaks.
#
# Same idea as the ATAC CPM plot: one number per peak = mean signal over the
# peak interval. Signal is the 1 kb-bin scCUT&Tag matrix, summed across cells
# (downstream/peak_coverage.py load_per_bin_signal).
#
#   raw:     unified/work/ctcf/mm/matrix.mtx.gz
#   imputed: unified/work/ctcf/impute/matrix_csr.npz
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "downstream"))
from peak_coverage import load_per_bin_signal, parse_region_names  # noqa: E402


def load_bed(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt") as fh:
        for ln in fh:
            if not ln.strip() or ln.startswith(("#", "track", "browser")):
                continue
            p = ln.rstrip("\n").split("\t")
            rest = p[3:] if len(p) > 3 else ["."]
            rows.append((p[0], int(p[1]), int(p[2]), rest))
    return rows


def index_bins(regions, raw, imp):
    chroms, starts, ends = parse_region_names(regions)
    by = defaultdict(lambda: {"i": [], "s": [], "e": []})
    for i, c in enumerate(chroms):
        by[c]["i"].append(i)
        by[c]["s"].append(int(starts[i]))
        by[c]["e"].append(int(ends[i]))
    out = {}
    for c, d in by.items():
        order = np.argsort(d["s"], kind="mergesort")
        idx = np.asarray(d["i"], dtype=np.int64)[order]
        out[c] = (
            np.asarray(d["s"], dtype=np.int64)[order],
            np.asarray(d["e"], dtype=np.int64)[order],
            raw[idx],
            imp[idx],
        )
    return out


def score_peak(chrom, s, e, by):
    if chrom not in by:
        return 0, np.nan, np.nan, np.nan, np.nan
    starts, ends, raw, imp = by[chrom]
    lo = int(np.searchsorted(starts, s, side="right")) - 1
    if lo < 0:
        lo = 0
    hi = int(np.searchsorted(starts, e, side="left"))
    raw_w = imp_w = ov_bp = 0.0
    n = 0
    for k in range(lo, hi):
        ov = min(e, int(ends[k])) - max(s, int(starts[k]))
        if ov <= 0:
            continue
        n += 1
        ov_bp += ov
        raw_w += ov * float(raw[k])
        imp_w += ov * float(imp[k])
    if ov_bp <= 0:
        return 0, np.nan, np.nan, np.nan, np.nan
    return n, raw_w, raw_w / ov_bp, imp_w, imp_w / ov_bp


def plot_two(raw_mean, imp_mean, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    def _panel(ax, x, title, color):
        x = x[np.isfinite(x)]
        pos = x[x > 0]
        n_zero = int((x <= 0).sum())
        if len(pos) == 0:
            ax.set_title(title)
            return
        xmin = max(float(np.quantile(pos, 0.001)) * 0.9, 1e-4)
        xmax = float(np.quantile(x, 0.999))
        bins = np.logspace(np.log10(xmin), np.log10(xmax), 45)
        ax.hist(np.clip(pos, xmin, xmax), bins=bins, color=color,
                alpha=0.85, edgecolor="white", linewidth=0.4)
        med = float(np.median(x))
        ax.axvline(med, color="#333", lw=1.2, ls=":",
                   label=f"median={med:.3g}   n_zero={n_zero:,}")
        ax.set_xscale("log")
        ax.set_xlim(xmin, xmax)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title)
        ax.set_ylabel("Number of CTCF peaks")
        ax.legend(frameon=False, fontsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    _panel(axes[0], raw_mean, "Raw scCUT&Tag  (mean UMIs / 1 kb bin)", "#2c4f7c")
    _panel(axes[1], imp_mean, "Imputed  (mean cells / 1 kb bin)", "#c0392b")
    axes[0].set_xlabel("raw coverage over the ChIP peak")
    axes[1].set_xlabel("imputed coverage over the ChIP peak")
    fig.suptitle("CTCF coverage at consensus ChIP peaks  (n=38,102)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", type=Path, default=ROOT / "data" / "CTCF_majority2of3.bed")
    ap.add_argument("--raw", type=Path, default=ROOT / "unified" / "work" / "ctcf" / "mm")
    ap.add_argument("--impute", type=Path, default=ROOT / "unified" / "work" / "ctcf" / "impute")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "cobinding" / "results" / "CTCF_majority2of3" / "raw_imputed_coverage")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] raw", args.raw, flush=True)
    raw, reg_raw, kind_r, ncell_r = load_per_bin_signal(args.raw)
    print("[load] impute", args.impute, flush=True)
    imp, reg_imp, kind_i, ncell_i = load_per_bin_signal(args.impute)
    if len(reg_raw) != len(reg_imp):
        raise SystemExit(f"region count mismatch raw={len(reg_raw)} impute={len(reg_imp)}")
    print(f"[bins] n={len(reg_raw):,}  raw_kind={kind_r} n_cells={ncell_r}  "
          f"impute_kind={kind_i} n_cells={ncell_i}", flush=True)
    print(f"[signal] raw_sum={raw.sum():.4g}  imputed_sum={imp.sum():.4g}", flush=True)

    by = index_bins(reg_raw, raw, imp)
    peaks = load_bed(args.peaks)
    print(f"[peaks] {args.peaks} n={len(peaks)}", flush=True)

    tsv = args.out_dir / "ctcf_consensus_raw_imputed_coverage.tsv"
    n_bins = np.zeros(len(peaks), dtype=np.int32)
    raw_sum = np.full(len(peaks), np.nan)
    raw_mean = np.full(len(peaks), np.nan)
    imp_sum = np.full(len(peaks), np.nan)
    imp_mean = np.full(len(peaks), np.nan)
    with tsv.open("w") as fh:
        fh.write("chrom\tstart\tend\tname\tn_bins\traw_sum\traw_mean\t"
                 "imputed_sum\timputed_mean\n")
        for i, (c, s, e, rest) in enumerate(peaks):
            n, rs, rm, ims, imm = score_peak(c, s, e, by)
            n_bins[i] = n
            raw_sum[i], raw_mean[i] = rs, rm
            imp_sum[i], imp_mean[i] = ims, imm
            name = rest[0] if rest else "."
            fh.write(f"{c}\t{s}\t{e}\t{name}\t{n}\t{rs:.6g}\t{rm:.6g}\t"
                     f"{ims:.6g}\t{imm:.6g}\n")

    def _summ(lab, v):
        x = v[np.isfinite(v)]
        print(f"[{lab}] median={np.median(x):.4g}  mean={np.mean(x):.4g}  "
              f"pct_zero={100 * (x <= 0).mean():.1f}  n={len(x)}", flush=True)

    _summ("raw_mean", raw_mean)
    _summ("imputed_mean", imp_mean)
    print(f"[overlap] median n_bins={np.median(n_bins):.1f}  "
          f"n_no_bin={(n_bins == 0).sum()}", flush=True)

    png = args.out_dir / "ctcf_consensus_raw_imputed_coverage.png"
    plot_two(raw_mean, imp_mean, png)
    (args.out_dir / "README.txt").write_text(
        "Coverage at CTCF majority ≥2-of-3 consensus peaks (n=38,102).\n"
        "One number per peak = overlap-weighted mean of 1 kb-bin signal "
        "(signal = sum across cells).\n\n"
        f"raw:     {args.raw}  ({kind_r}, {ncell_r} cells)\n"
        f"imputed: {args.impute}  ({kind_i}, {ncell_i} cells)\n\n"
        "ctcf_consensus_raw_imputed_coverage.tsv / .png\n"
    )
    print(f"[out] {tsv}")
    print(f"[out] {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
