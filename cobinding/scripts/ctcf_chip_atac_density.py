#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# ctcf_chip_atac_density.py
#
# ATAC accessibility of HEK293 CTCF ChIP-seq peaks (the peaks themselves), used
# as a *soft* filter instead of a binary open-chromatin (ATAC overlap) mask:
# score each ChIP peak by mean OmniATAC fold-change, inspect the distribution,
# and drop the lowest --drop-frac (default 5%) rather than requiring an ATAC
# union-mask overlap.
#
# X: OmniATAC accessibility (fold-change)
# Y: density (KDE of fold-change)
#
# Needs pyBigWig (deepTools env on Longleaf). Re-execs that interpreter if the
# current one cannot import it.
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DT_PY = Path(
    "/nas/longleaf/rhel9/apps/deeptools/3.5.6/miniconda3/envs/deeptools/bin/python"
)


def _ensure_pybigwig():
    try:
        import pyBigWig  # noqa: F401
        return
    except ImportError:
        pass
    if DT_PY.is_file() and Path(sys.executable).resolve() != DT_PY.resolve():
        os.execv(str(DT_PY), [str(DT_PY), *sys.argv])
    raise SystemExit("pyBigWig is required (deepTools python on Longleaf)")


_ensure_pybigwig()

import numpy as np  # noqa: E402
import pyBigWig  # noqa: E402


def load_bed(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt") as fh:
        for ln in fh:
            if not ln.strip() or ln.startswith(("#", "track", "browser")):
                continue
            p = ln.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            rows.append(p)
    return rows


def max_overlap_bp(chip_rows, atac_rows):
    """Max overlapping bases of each ChIP peak vs any ATAC peak.

    BED is 0-based half-open [start, end). Overlap in bp is
        min(end_a, end_b) - max(start_a, start_b)
    and a peak is called overlapping when that value is >= 1 (a single
    shared base is enough; no fraction of the peak is required).
    Adjacent/bookended intervals (overlap 0) are not counted.
    """
    atac = defaultdict(lambda: ([], []))
    for p in atac_rows:
        s, e = int(p[1]), int(p[2])
        if e > s:
            atac[p[0]][0].append(s)
            atac[p[0]][1].append(e)
    atac_np = {c: (np.asarray(ss, np.int64), np.asarray(ee, np.int64))
               for c, (ss, ee) in atac.items()}

    ov = np.zeros(len(chip_rows), dtype=np.int32)
    by = defaultdict(list)
    for i, p in enumerate(chip_rows):
        by[p[0]].append(i)
    for chrom, idxs in by.items():
        if chrom not in atac_np:
            continue
        as_, ae_ = atac_np[chrom]
        order = np.argsort(as_, kind="mergesort")
        as_ = as_[order]
        ae_ = ae_[order]
        max_end = np.maximum.accumulate(ae_)
        for i in idxs:
            s, e = int(chip_rows[i][1]), int(chip_rows[i][2])
            j = int(np.searchsorted(as_, e, side="left"))
            if j == 0 or int(max_end[j - 1]) <= s:
                continue
            o = np.minimum(e, ae_[:j]) - np.maximum(s, as_[:j])
            ov[i] = int(max(0, int(o.max())))
    return ov


def mean_signal(bw, chrom, start, end, chroms):
    if chrom not in chroms:
        return np.nan
    clen = chroms[chrom]
    s = max(0, start)
    e = min(clen, end)
    if e <= s:
        return np.nan
    v = bw.stats(chrom, s, e, type="mean")
    if not v or v[0] is None:
        return 0.0
    return max(float(v[0]), 0.0)


def drop_lowest(fc, frac):
    """Boolean mask: True = keep. Drops the lowest `frac` of peaks (exact count)."""
    n = len(fc)
    n_drop = int(round(frac * n))
    dropped = np.zeros(n, dtype=bool)
    if n_drop <= 0:
        return ~dropped, 0.0, 0
    order = np.argsort(fc, kind="mergesort")
    dropped[order[:n_drop]] = True
    cutoff = float(fc[order[n_drop - 1]])
    return ~dropped, cutoff, n_drop


def kde_xy(values, n=512, xmin=0.0):
    from scipy.stats import gaussian_kde
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    hi = float(v.max())
    xs = np.linspace(xmin, hi * 1.02 if hi > 0 else 1.0, n)
    return xs, gaussian_kde(v)(xs)


def plot_density(fc, open_mask, keep, cutoff, drop_frac, out_png,
                 n_peaks, median_fc, title, open_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, ys = kde_xy(fc)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.fill_between(xs, ys, color="#3b6ea5", alpha=0.22, linewidth=0)
    ax.plot(xs, ys, color="#2c4f7c", lw=2.2, label=f"CTCF ChIP peaks (n={n_peaks:,})")

    dropped = ~keep
    ax.axvspan(0, cutoff, color="#c0392b", alpha=0.12, zorder=0)
    ax.axvline(cutoff, color="#c0392b", ls="--", lw=1.4,
               label=f"drop lowest {100 * drop_frac:.0f}%  (FC ≤ {cutoff:.2f}, n={int(dropped.sum()):,})")
    ax.axvline(median_fc, color="#555555", ls=":", lw=1.1,
               label=f"median FC = {median_fc:.2f}")

    if open_mask is not None and open_mask.any() and (~open_mask).any():
        xs_o, ys_o = kde_xy(fc[open_mask])
        xs_c, ys_c = kde_xy(fc[~open_mask])
        ax.plot(xs_o, ys_o, color="#2a9d8f", lw=1.3, alpha=0.9,
                label=f"≥1 bp {open_label} overlap (n={int(open_mask.sum()):,})")
        ax.plot(xs_c, ys_c, color="#9aa3ad", lw=1.3, alpha=0.9,
                label=f"no {open_label} overlap (n={int((~open_mask).sum()):,})")

    ax.set_xlim(0, float(np.quantile(fc, 0.995)))
    ax.set_xlabel("ATAC-seq accessibility  (fold-change)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8.2, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_union_frac(frac, out_png, n_peaks, title, open_label):
    """X = fraction of each ChIP peak overlapping the ATAC-union mask."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frac = np.asarray(frac, float)
    n_open = int((frac > 0).sum())
    n_zero = int((frac <= 0).sum())
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bins = np.linspace(0.0, 1.0, 41)
    ax.hist(
        [frac[frac <= 0], frac[frac > 0]],
        bins=bins, stacked=True, density=False,
        color=["#9aa3ad", "#2a9d8f"], alpha=0.9,
        edgecolor="white", linewidth=0.3,
        label=[
            f"no {open_label} overlap (n={n_zero:,})",
            f"≥1 bp {open_label} (n={n_open:,})",
        ],
    )
    med_all = float(np.median(frac))
    ax.axvline(med_all, color="#555555", ls=":", lw=1.1,
               label=f"median overlap = {100 * med_all:.0f}%")
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"ATAC-seq accessibility  ({open_label} overlap fraction)")
    ax.set_ylabel("Number of CTCF peaks")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8.2, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_bed(path, rows, mask):
    with path.open("w") as fh:
        for row, keep in zip(rows, mask):
            if keep:
                fh.write("\t".join(row) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--peaks", type=Path,
                   default=ROOT / "data" / "CTCF_HEK293_peaks.bed")
    p.add_argument(
        "--atac-bw", type=Path,
        default=ROOT / "atac_omniatac" / "caper_work" / "atac" /
        "70bfc7e8-3760-496f-bd8e-5a4bc9e02b51" / "call-macs2_signal_track_pooled" /
        "execution" / "rep.pooled.fc.signal.bigwig",
    )
    p.add_argument(
        "--open-bed", type=Path,
        default=ROOT / "data" / "HEK293T_ATAC_union3.bed.gz",
        help="Open-chromatin mask (default: HEK293T ATAC union3).",
    )
    p.add_argument("--open-label", default="ATAC-union",
                   help="Legend/summary name for the open mask.")
    p.add_argument(
        "--x-metric",
        choices=("fc", "union-frac"),
        default="fc",
        help="Horizontal axis: OmniATAC fold-change, or ATAC-union overlap fraction.",
    )
    p.add_argument("--drop-frac", type=float, default=0.05,
                   help="Drop this lowest fraction of ChIP peaks by ATAC FC (default 0.05).")
    p.add_argument("--label", default="CTCF_HEK293",
                   help="Stem for kept/dropped BED filenames.")
    p.add_argument("--title", default="ATAC accessibility of HEK293 CTCF ChIP-seq peaks")
    p.add_argument("--caption", default=(
        "Peaks: ENCFF314ZAL HEK293 CTCF.  "
        "Accessibility: mean pooled HEK293T OmniATAC FC (GSE302716)."
    ))
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "cobinding" / "results" / "CTCF")
    args = p.parse_args()
    if not (0 < args.drop_frac < 1):
        raise SystemExit("--drop-frac must be in (0, 1)")

    peaks = load_bed(args.peaks)
    if not peaks:
        raise SystemExit(f"no peaks in {args.peaks}")
    if not args.atac_bw.is_file():
        raise SystemExit(f"missing ATAC bigWig: {args.atac_bw}")

    bw = pyBigWig.open(str(args.atac_bw))
    chroms = bw.chroms()
    fc = np.empty(len(peaks), dtype=float)
    for i, p_row in enumerate(peaks):
        fc[i] = mean_signal(bw, p_row[0], int(p_row[1]), int(p_row[2]), chroms)
    bw.close()

    atac_rows = load_bed(args.open_bed) if args.open_bed.is_file() else []
    overlap_bp = max_overlap_bp(peaks, atac_rows)
    open_mask = overlap_bp >= 1

    ok = np.isfinite(fc)
    n_nan = int((~ok).sum())
    fc = np.maximum(fc[ok], 0.0)
    peaks_ok = [peaks[i] for i, g in enumerate(ok) if g]
    open_mask = open_mask[ok]
    overlap_bp = overlap_bp[ok]
    n_zero = int((fc <= 0).sum())

    keep, cutoff, n_drop = drop_lowest(fc, args.drop_frac)
    n_keep = int(keep.sum())
    n_drop_also_open = int((~keep & open_mask).sum())
    n_keep_not_open = int((keep & ~open_mask).sum())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    peak_len = np.array(
        [max(1, int(p_row[2]) - int(p_row[1])) for p_row in peaks_ok],
        dtype=float,
    )
    union_frac = np.clip(overlap_bp / peak_len, 0.0, 1.0)
    tsv = args.out_dir / "chipseq_atac_density.tsv"
    with tsv.open("w") as fh:
        fh.write("chrom\tstart\tend\tatac_fc\toverlap_bp\toverlap_frac\toverlaps_open\tkeep\n")
        for p_row, v, ob, uf, o, k in zip(
                peaks_ok, fc, overlap_bp, union_frac, open_mask, keep):
            fh.write(
                f"{p_row[0]}\t{p_row[1]}\t{p_row[2]}\t{v:.6g}\t{int(ob)}\t"
                f"{uf:.6g}\t{int(o)}\t{int(k)}\n"
            )

    pct = int(round(100 * args.drop_frac))
    kept_bed = args.out_dir / f"{args.label}.atac_drop_q{pct:02d}.bed"
    drop_bed = args.out_dir / f"{args.label}.atac_low_q{pct:02d}.bed"
    write_bed(kept_bed, peaks_ok, keep)
    write_bed(drop_bed, peaks_ok, ~keep)

    png = args.out_dir / "chipseq_atac_density.png"
    if args.x_metric == "union-frac":
        plot_union_frac(
            union_frac, png, n_peaks=len(fc),
            title=args.title, open_label=args.open_label,
        )
    else:
        plot_density(fc, open_mask, keep, cutoff, args.drop_frac, png,
                     n_peaks=len(fc),
                     median_fc=float(np.median(fc)),
                     title=args.title,
                     open_label=args.open_label)

    summary = args.out_dir / "chipseq_atac_density_summary.txt"
    qs = np.quantile(fc, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9])
    with summary.open("w") as fh:
        fh.write(f"peaks_file\t{args.peaks}\n")
        fh.write(f"atac_bw\t{args.atac_bw}\n")
        fh.write(f"n_peaks\t{len(peaks)}\n")
        fh.write(f"n_scored\t{len(fc)}\n")
        fh.write(f"n_nan\t{n_nan}\n")
        fh.write(f"n_zero_atac\t{n_zero}\n")
        fh.write(f"open_bed\t{args.open_bed}\n")
        fh.write(f"open_label\t{args.open_label}\n")
        fh.write(f"n_overlap_open\t{int(open_mask.sum())}\n")
        fh.write(f"frac_overlap_open\t{open_mask.mean():.4f}\n")
        fh.write("overlap_rule\t>=1 bp (no minimum fraction)\n")
        ov_pos = overlap_bp[open_mask]
        if len(ov_pos):
            fh.write(f"n_overlap_eq_1bp\t{int((overlap_bp == 1).sum())}\n")
            fh.write(f"n_overlap_lt_10bp\t{int(((overlap_bp >= 1) & (overlap_bp < 10)).sum())}\n")
            fh.write(f"median_overlap_bp\t{float(np.median(ov_pos)):.1f}\n")
            fh.write(f"min_overlap_bp\t{int(ov_pos.min())}\n")
            fh.write(f"max_overlap_bp\t{int(ov_pos.max())}\n")
        fh.write(f"drop_frac\t{args.drop_frac}\n")
        fh.write(f"drop_cutoff_atac_fc\t{cutoff:.6g}\n")
        fh.write(f"n_dropped\t{n_drop}\n")
        fh.write(f"n_kept\t{n_keep}\n")
        fh.write(f"n_dropped_but_open\t{n_drop_also_open}\n")
        fh.write(f"n_kept_not_open\t{n_keep_not_open}\n")
        fh.write(f"median_atac_fc\t{np.median(fc):.4f}\n")
        fh.write(f"mean_atac_fc\t{np.mean(fc):.4f}\n")
        fh.write(f"q05_q10_q25_q50_q75_q90\t"
                 f"{qs[0]:.4f},{qs[1]:.4f},{qs[2]:.4f},{qs[3]:.4f},{qs[4]:.4f},{qs[5]:.4f}\n")
        fh.write(f"median_atac_fc_open\t{np.median(fc[open_mask]):.4f}\n")
        fh.write(f"median_atac_fc_closed\t{np.median(fc[~open_mask]):.4f}\n")
        fh.write(f"x_metric\t{args.x_metric}\n")
        fh.write(f"median_union_frac\t{float(np.median(union_frac)):.4f}\n")
        fh.write(f"median_union_frac_open\t{float(np.median(union_frac[open_mask])):.4f}\n")
        fh.write(f"kept_bed\t{kept_bed}\n")
        fh.write(f"dropped_bed\t{drop_bed}\n")

    print(f"[peaks] {len(peaks):,}  scored {len(fc):,}  nan {n_nan}")
    n_open = int(open_mask.sum())
    ov_pos = overlap_bp[open_mask]
    n_eq1 = int((overlap_bp == 1).sum())
    print(f"[open]  {n_open:,} / {len(fc):,} ({100 * open_mask.mean():.1f}%) "
          f"with >=1 bp {args.open_label} overlap")
    if n_open:
        print(f"[ovlp]  among overlapping: min {int(ov_pos.min())} bp  "
              f"median {float(np.median(ov_pos)):.0f} bp  "
              f"exactly 1 bp: {n_eq1:,}")
    print(f"[drop]  lowest {100 * args.drop_frac:.0f}%  cutoff FC={cutoff:.3f}  "
          f"drop {n_drop:,}  keep {n_keep:,}")
    print(f"[vs open] dropped-but-open {n_drop_also_open:,}  "
          f"kept-but-not-open {n_keep_not_open:,}")
    print(f"[fc]    median {np.median(fc):.3f}  q05 {qs[0]:.3f}")
    print(f"[out]   {png}")
    print(f"[out]   {kept_bed}")
    print(f"[out]   {drop_bed}")
    print(f"[out]   {tsv}")
    print(f"[out]   {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
