#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# ATAC CPM at CTCF consensus ChIP peaks (one number per peak).
#
# For each peak: mean coverage in GSE283384 CPM bigWig, mean coverage in
# GSE152177 CPM bigWig, then average those two. Histogram that vector.
# Lower 5% cutoff = 5th percentile of the same vector.
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DT_PY = Path(
    "/nas/longleaf/rhel9/apps/deeptools/3.5.6/miniconda3/envs/deeptools/bin/python"
)


def _ensure():
    try:
        import pyBigWig  # noqa: F401
        return
    except ImportError:
        pass
    if DT_PY.is_file() and Path(sys.executable).resolve() != DT_PY.resolve():
        os.execv(str(DT_PY), [str(DT_PY), *sys.argv])
    raise SystemExit("pyBigWig required")


_ensure()
import numpy as np  # noqa: E402
import pyBigWig  # noqa: E402


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


def mean_cpm(bw_path: Path, rows):
    bw = pyBigWig.open(str(bw_path))
    chroms = bw.chroms() or {}
    out = np.full(len(rows), np.nan)
    for i, (c, s, e, *_) in enumerate(rows):
        if c not in chroms:
            continue
        clen = chroms[c]
        s2, e2 = max(0, s), min(clen, e)
        if e2 <= s2:
            continue
        v = bw.stats(c, s2, e2, type="mean")
        if v and v[0] is not None:
            out[i] = max(float(v[0]), 0.0)
        else:
            out[i] = 0.0
    bw.close()
    return out


def plot_hist(x, cutoff, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    x = x[np.isfinite(x)]
    # Linear axis, zoomed to the bulk. 0–2 CPM holds ~91% of peaks; the
    # long tail (up to ~13) is omitted so the main mass is readable.
    xmax = 2.0
    n_shown = int((x <= xmax).sum())
    n_hi = int((x > xmax).sum())
    bins = np.linspace(0, xmax, 41)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.hist(x[x <= xmax], bins=bins, color="#2c4f7c", alpha=0.85,
            edgecolor="white", linewidth=0.4)
    ax.axvline(cutoff, color="#c0392b", lw=1.8, ls="--",
               label=f"lower 5%  =  {cutoff:.3f} CPM")
    ax.axvline(np.median(x), color="#7f8c8d", lw=1.2, ls=":",
               label=f"median  =  {np.median(x):.2f} CPM")
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.set_xlabel("ATAC-seq accessibility  (CPM)")
    ax.set_ylabel("Number of CTCF peaks")
    ax.set_title(f"ATAC CPM at CTCF consensus ChIP peaks  (n={len(x):,})")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.text(0.98, 0.72, f"showing {n_shown:,} peaks ≤ {xmax:g} CPM\n"
            f"{n_hi:,} peaks > {xmax:g} not plotted",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color="#555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip-peaks", type=Path,
                    default=ROOT / "data" / "CTCF_majority2of3.bed")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "cobinding" / "results" / "CTCF_majority2of3" / "atac_cpm")
    args = ap.parse_args()
    tracks = [
        ("GSE283384", ROOT / "downstream" / "tracks" / "atac_bulk_sources" / "2_GSE283384_pooled_CPM.bw"),
        ("GSE152177", ROOT / "downstream" / "tracks" / "atac_bulk_sources" / "3_GSE152177_pooled_CPM.bw"),
    ]
    rows = load_bed(args.chip_peaks)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[chip] {args.chip_peaks} n={len(rows)}", flush=True)
    stack = []
    for name, path in tracks:
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        v = mean_cpm(path, rows)
        stack.append(v)
        print(f"[{name}] median={np.nanmedian(v):.3f}", flush=True)
    cpm = np.nanmean(np.vstack(stack), axis=0)
    cutoff = float(np.nanquantile(cpm, 0.05))
    n_drop = int(np.sum(cpm <= cutoff))
    print(f"[cutoff] 5th percentile = {cutoff:.6g} CPM   n_below={n_drop}", flush=True)

    tsv = args.out_dir / "ctcf_consensus_atac_cpm.tsv"
    drop_bed = args.out_dir / "ctcf_consensus_below5pct.bed"
    keep_bed = args.out_dir / "ctcf_consensus_keep95.bed"
    with tsv.open("w") as fh, drop_bed.open("w") as fd, keep_bed.open("w") as fk:
        fh.write("chrom\tstart\tend\tname\tmean_atac_cpm\tbelow_5pct\n")
        for (c, s, e, rest), v in zip(rows, cpm):
            name = rest[0] if rest else "."
            flag = int(v <= cutoff)
            fh.write(f"{c}\t{s}\t{e}\t{name}\t{v:.6g}\t{flag}\n")
            cols = [c, str(s), str(e)] + rest
            line = "\t".join(cols) + "\n"
            (fd if flag else fk).write(line)

    (args.out_dir / "cutoff_lower5pct.txt").write_text(
        f"metric\tmean ATAC CPM over the CTCF peak (GSE283384 + GSE152177)\n"
        f"n_peaks\t{len(rows)}\n"
        f"median_cpm\t{float(np.nanmedian(cpm)):.6g}\n"
        f"lower_5pct_cutoff_cpm\t{cutoff:.6g}\n"
        f"n_below_cutoff\t{n_drop}\n"
        f"n_kept\t{len(rows) - n_drop}\n"
    )
    png = args.out_dir / "ctcf_consensus_atac_cpm.png"
    plot_hist(cpm, cutoff, png)
    print(f"[out] {png}")
    print(f"[out] {drop_bed}")
    print(f"[out] {keep_bed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
