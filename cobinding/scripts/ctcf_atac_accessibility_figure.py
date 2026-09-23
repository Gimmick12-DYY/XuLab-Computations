#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Published-convention ATAC accessibility at CTCF consensus peaks.
#
# Heatmap + metaprofile (deepTools computeMatrix / plotHeatmap):
#   bamCoverage CPM 10 bp → summit ± 2 kb → sort by mean ATAC or k-means.
# ECDF + ChIP-decile boxplots: mean CPM over the original peak (count-based).
# Backgrounds: shuffled 4 kb windows (blacklist-excluded) and ChromHMM-18.
#
# Usage:
#   python cobinding/scripts/ctcf_atac_accessibility_figure.py
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import os
import subprocess
import sys
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DT = Path("/nas/longleaf/rhel9/apps/deeptools/3.5.6/miniconda3/envs/deeptools/bin")
DT_PY = DT / "python"
BEDTOOLS = Path("/nas/longleaf/rhel9/apps/bedtools/2.31.1/bedtools2/bin/bedtools")
MAIN = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
FLANK = 2000
BIN = 10
BROAD = OrderedDict([
    ("Promoter", ["TssA", "TssFlnk", "TssFlnkU", "TssFlnkD", "TssBiv"]),
    ("Enhancer", ["EnhA1", "EnhA2", "EnhG1", "EnhG2", "EnhWk", "EnhBiv"]),
    ("Transcribed", ["Tx", "TxWk"]),
    ("Polycomb", ["ReprPC", "ReprPCWk"]),
    ("Het/repeats", ["Het", "ZNF/Rpts"]),
    ("Quiescent", ["Quies"]),
])
STATE2BROAD = {s: b for b, ss in BROAD.items() for s in ss}
EPI = list(BROAD.keys()) + ["Unassigned"]
BCOL = {"Promoter": "#d93a34", "Enhancer": "#eda320", "Transcribed": "#3f9e4d",
        "Polycomb": "#7b68c4", "Het/repeats": "#3a4750", "Quiescent": "#d9dce1",
        "Unassigned": "#bbbbbb"}


def _ensure():
    try:
        import pyBigWig  # noqa: F401
        return
    except ImportError:
        pass
    if DT_PY.is_file() and Path(sys.executable).resolve() != DT_PY.resolve():
        os.execv(str(DT_PY), [str(DT_PY), *sys.argv])
    raise SystemExit("need deepTools python (pyBigWig)")


_ensure()
import numpy as np  # noqa: E402
import pyBigWig  # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from distal_chromhmm_composition import assign, load_segments  # noqa: E402


def _open(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def load_bed(path: Path):
    rows = []
    with _open(path) as fh:
        for ln in fh:
            if not ln.strip() or ln.startswith(("#", "track", "browser")):
                continue
            p = ln.rstrip("\n").split("\t")
            rows.append((p[0], int(p[1]), int(p[2]), p[3] if len(p) > 3 else "."))
    return rows


def load_sizes(path: Path):
    out = {}
    with open(path) as fh:
        for ln in fh:
            c, n = ln.split()[:2]
            if c in MAIN:
                out[c] = int(n)
    return out


def load_blacklist(path: Path):
    by = {}
    if not path.is_file():
        return by
    for c, s, e, *_ in load_bed(path):
        by.setdefault(c, []).append((s, e))
    for c in by:
        by[c].sort()
    return by


def hits_bl(chrom, s, e, bl):
    ivs = bl.get(chrom)
    if not ivs:
        return False
    for a, b in ivs:
        if a >= e:
            break
        if b > s:
            return True
    return False


def mean_bw(bw, chroms, chrom, s, e):
    if chrom not in chroms:
        return np.nan
    clen = chroms[chrom]
    s2, e2 = max(0, s), min(clen, e)
    if e2 <= s2:
        return np.nan
    v = bw.stats(chrom, s2, e2, type="mean")
    if not v or v[0] is None:
        return 0.0
    return max(float(v[0]), 0.0)


def score_rows(bw_path: Path, rows, mode="peak"):
    bw = pyBigWig.open(str(bw_path))
    chroms = bw.chroms() or {}
    out = np.full(len(rows), np.nan)
    sizes = {c: chroms[c] for c in chroms}
    for i, (c, s, e, *_) in enumerate(rows):
        if mode == "window":
            mid = (s + e) // 2
            s, e = mid - FLANK, mid + FLANK
        if c in sizes:
            e = min(e, sizes[c])
            s = max(0, s)
        out[i] = mean_bw(bw, chroms, c, s, e)
    bw.close()
    return out


def write_bed(path: Path, rows):
    with path.open("w") as fh:
        for i, r in enumerate(rows):
            c, s, e = r[0], r[1], r[2]
            name = r[3] if len(r) > 3 else f"r{i}"
            fh.write(f"{c}\t{s}\t{e}\t{name}\t0\t.\n")


def windows_from_peaks(rows, sizes):
    out = []
    for c, s, e, name in rows:
        mid = (s + e) // 2
        w0, w1 = mid - FLANK, mid + FLANK
        clen = sizes.get(c, 10**12)
        w0, w1 = max(0, w0), min(clen, w1)
        if w1 - w0 < 100:
            continue
        out.append((c, w0, w1, name))
    return out


def load_tss(path: Path):
    by = {}
    with open(path) as fh:
        for ln in fh:
            p = ln.split()
            if len(p) < 4:
                continue
            # HOMER: name chrom start end strand
            c, a, b = p[1], int(p[2]), int(p[3])
            by.setdefault(c, []).append((a + b) // 2)
    for c in by:
        by[c] = np.array(sorted(set(by[c])), dtype=np.int64)
    return by


def tss_dist(chrom, mid, tss):
    arr = tss.get(chrom)
    if arr is None or len(arr) == 0:
        return np.nan
    i = int(np.searchsorted(arr, mid))
    d = np.inf
    if i < len(arr):
        d = min(d, abs(int(arr[i]) - mid))
    if i > 0:
        d = min(d, abs(int(arr[i - 1]) - mid))
    return float(d)


def run(cmd):
    print("[cmd]", " ".join(str(x) for x in cmd[:8]), "...", flush=True)
    subprocess.run(cmd, check=True)


def plot_ecdf(series, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    cols = ["#2c4f7c", "#c0392b", "#7f8c8d", "#2a9d8f", "#8e44ad"]
    for i, (lab, v) in enumerate(series.items()):
        x = np.sort(v[np.isfinite(v)])
        if len(x) < 2:
            continue
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=cols[i % len(cols)], lw=2.0, label=f"{lab} (n={len(x):,})")
    ax.set_xlabel("ATAC-seq  (mean CPM in region)")
    ax.set_ylabel("ECDF")
    xmax = max((np.nanquantile(v[np.isfinite(v)], 0.995) for v in series.values()
                if np.isfinite(v).sum() > 10), default=5)
    ax.set_xlim(0, float(xmax))
    ax.set_ylim(0, 1)
    ax.set_title("ATAC accessibility at CTCF consensus vs background")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_deciles(chip, atac, distal, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _boxes(c, a):
        ok = np.isfinite(c) & np.isfinite(a)
        c, a = c[ok], a[ok]
        qs = np.quantile(c, np.linspace(0, 1, 11))
        qs[0] -= 1e-9
        boxes = []
        for i in range(10):
            m = (c > qs[i]) & (c <= qs[i + 1])
            boxes.append(a[m])
        return boxes

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
    panels = [
        (axes[0], "all CTCF consensus", _boxes(chip, atac)),
        (axes[1], "TSS-distal only  (≥2 kb)", _boxes(chip[distal], atac[distal])),
    ]
    for ax, title, boxes in panels:
        ax.boxplot(boxes, showfliers=False,
                   medianprops={"color": "#c0392b", "lw": 1.6},
                   boxprops={"color": "#2c4f7c"}, whiskerprops={"color": "#2c4f7c"})
        ax.set_xticklabels([f"D{i}" for i in range(1, 11)])
        ax.set_title(title, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("ATAC-seq  (mean CPM)")
    fig.supxlabel("CTCF ChIP CPM decile  (mean of ENCFF139DCW + GSE103651; D1 = lowest)",
                  fontsize=9)
    fig.suptitle("ATAC at CTCF peaks, by ChIP occupancy", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_bars(counts, order, colors, title, ylabel, out_png, genome=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = sum(counts.get(k, 0) for k in order) or 1
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(order))
    frac = [100 * counts.get(k, 0) / n for k in order]
    ax.bar(x, frac, color=[colors.get(k, "#888") for k in order], width=0.7,
           edgecolor="white")
    if genome:
        gtot = sum(genome.get(k, 0) for k in order) or 1
        gfrac = [100 * genome.get(k, 0) / gtot for k in order]
        ax.plot(x, gfrac, "o--", color="#333", ms=5, label="genome bp")
        ax.legend(frameon=False, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", type=Path, default=ROOT / "data" / "CTCF_majority2of3.bed")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "cobinding" / "results" / "CTCF_majority2of3" / "atac_accessibility")
    ap.add_argument("--kmeans", type=int, default=3)
    ap.add_argument("--skip-matrix", action="store_true",
                    help="reuse existing computeMatrix.gz / heatmaps")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    atac = [
        ("GSE283384", ROOT / "downstream/tracks/atac_bulk_sources/2_GSE283384_pooled_CPM.bw"),
        ("GSE152177", ROOT / "downstream/tracks/atac_bulk_sources/3_GSE152177_pooled_CPM.bw"),
    ]
    chip_bws = [
        ("ENCODE", ROOT / "downstream/tracks/ctcf_bulk_sources/1_ENCODE_HEK293_ENCFF139DCW_CPM.bw"),
        ("GSE103651", ROOT / "downstream/tracks/ctcf_bulk_sources/2_GSE103651_pooled_CPM.bw"),
    ]
    sizes = load_sizes(ROOT / "downstream/cache/hg38.chrom.sizes")
    bl = load_blacklist(ROOT / "downstream/cache/hg38-blacklist.v2.bed.gz")
    chrom_sizes_bed = out / "hg38.main.chrom.sizes"
    with chrom_sizes_bed.open("w") as fh:
        for c in MAIN:
            fh.write(f"{c}\t{sizes[c]}\n")

    peaks_raw = load_bed(args.peaks)
    n_in = len(peaks_raw)
    peaks = [r for r in peaks_raw if r[0] in MAIN and not hits_bl(r[0], r[1], r[2], bl)]
    print(f"[peaks] in={n_in}  after chrom+blacklist={len(peaks)}", flush=True)
    wins = windows_from_peaks(peaks, sizes)
    peak_bed = out / "ctcf_consensus.bed"
    win_bed = out / "ctcf_summit_pm2kb.bed"
    write_bed(peak_bed, peaks)
    write_bed(win_bed, wins)

    # shuffled 4 kb windows, same count, same chrom distribution
    rand_bed = out / "random_4kb.bed"
    tmp = subprocess.check_output(
        [str(BEDTOOLS), "shuffle", "-i", str(win_bed), "-g", str(chrom_sizes_bed),
         "-excl", str(ROOT / "downstream/cache/hg38-blacklist.v2.bed.gz"),
         "-chrom", "-seed", "1"], text=True)
    rand_bed.write_text(tmp)
    random_rows = load_bed(rand_bed)
    print(f"[random] {len(random_rows)} 4 kb windows", flush=True)

    # scores
    atac_peak = {}
    atac_win = {}
    atac_rand = {}
    bw_paths = []
    for name, p in atac:
        if not p.is_file():
            raise SystemExit(f"missing {p}")
        bw_paths.append(str(p))
        atac_peak[name] = score_rows(p, peaks, "peak")
        atac_win[name] = score_rows(p, peaks, "window")
        atac_rand[name] = score_rows(p, random_rows, "peak")
        print(f"[{name}] peak median={np.nanmedian(atac_peak[name]):.3f}  "
              f"window median={np.nanmedian(atac_win[name]):.3f}  "
              f"random median={np.nanmedian(atac_rand[name]):.3f}", flush=True)
    mean_peak = np.nanmean(np.vstack(list(atac_peak.values())), axis=0)
    mean_win = np.nanmean(np.vstack(list(atac_win.values())), axis=0)
    mean_rand = np.nanmean(np.vstack(list(atac_rand.values())), axis=0)
    chip_tracks = []
    for name, p in chip_bws:
        if not p.is_file():
            print(f"[warn] missing ChIP track {p}", flush=True)
            continue
        v = score_rows(p, peaks, "peak")
        print(f"[ChIP {name}] peak median={np.nanmedian(v):.3f}", flush=True)
        chip_tracks.append(v)
    chip = np.nanmean(np.vstack(chip_tracks), axis=0) if chip_tracks else np.full(len(peaks), np.nan)

    seg_idx, states = load_segments(ROOT / "data/HEK293T_chromHMM18.bed.gz")
    hmm = []
    for c, s, e, *_ in peaks:
        i = assign(f"{c}:{s}-{e}", seg_idx, len(states))
        hmm.append("Unassigned" if i is None else states[i])
    broad = [STATE2BROAD.get(s, "Unassigned") for s in hmm]
    tss = load_tss(Path("/nas/longleaf/rhel9/apps/homer/5.1/data/genomes/hg38/hg38.tss"))
    dist = np.array([tss_dist(c, (s + e) // 2, tss) for c, s, e, *_ in peaks])
    proximal = dist < 2000

    # genome ChromHMM bp
    g_bp = Counter()
    with gzip.open(ROOT / "data/HEK293T_chromHMM18.bed.gz", "rt") as fh:
        for ln in fh:
            p = ln.split("\t")
            if p[0] not in MAIN:
                continue
            st = p[3].strip()
            g_bp[STATE2BROAD.get(st, "Unassigned")] += int(p[2]) - int(p[1])

    tsv = out / "per_peak.tsv"
    with tsv.open("w") as fh:
        fh.write("chrom\tstart\tend\tname\tatac_gse283384\tatac_gse152177\t"
                 "atac_mean_cpm\tchip_encode_cpm\tchromhmm\tbroad\ttss_dist\tproximal_2kb\n")
        for i, (c, s, e, name) in enumerate(peaks):
            fh.write(
                f"{c}\t{s}\t{e}\t{name}\t{atac_peak['GSE283384'][i]:.6g}\t"
                f"{atac_peak['GSE152177'][i]:.6g}\t{mean_peak[i]:.6g}\t"
                f"{chip[i]:.6g}\t{hmm[i]}\t{broad[i]}\t{dist[i]:.0f}\t"
                f"{int(proximal[i])}\n"
            )

    is_prom = np.array([b == "Promoter" for b in broad])
    is_quies = np.array([b == "Quiescent" for b in broad])
    plot_ecdf(OrderedDict([
        ("CTCF consensus", mean_peak),
        ("random 4 kb (shuffled)", mean_rand),
        ("CTCF in Promoter ChromHMM", mean_peak[is_prom]),
        ("CTCF in Quiescent ChromHMM", mean_peak[is_quies]),
        ("CTCF TSS-distal (≥2 kb)", mean_peak[~proximal]),
    ]), out / "ecdf_atac_cpm.png")
    plot_deciles(chip, mean_peak, ~proximal, out / "chip_decile_atac_boxplot.png")
    plot_bars(Counter(broad), EPI, BCOL,
              "ChromHMM-18 class of CTCF consensus peaks",
              "% of CTCF peaks", out / "chromhmm_composition.png", genome=g_bp)
    ann = Counter({
        "TSS-proximal (<2 kb)": int(proximal.sum()),
        "TSS-distal": int((~proximal).sum()),
    })
    plot_bars(ann, ["TSS-proximal (<2 kb)", "TSS-distal"],
              {"TSS-proximal (<2 kb)": "#d93a34", "TSS-distal": "#3a4750"},
              "Distance to nearest TSS", "% of CTCF peaks",
              out / "tss_annotation.png")

    mtx = out / "computeMatrix.gz"
    if args.skip_matrix and mtx.is_file():
        print(f"[skip] reusing {mtx}", flush=True)
    else:
        run([
            str(DT / "computeMatrix"), "reference-point",
            "--referencePoint", "center",
            "-b", str(FLANK), "-a", str(FLANK), "-bs", str(BIN),
            "-R", str(win_bed),
            "-S", *bw_paths,
            "--samplesLabel", *[n for n, _ in atac],
            "--missingDataAsZero",
            "-p", "8",
            "-o", str(mtx),
        ])
    run([
        str(DT / "plotHeatmap"), "-m", str(mtx),
        "-out", str(out / "heatmap_sortedByMean.png"),
        "--sortUsing", "mean", "--sortRegions", "descend",
        "--colorMap", "Reds",
        "--whatToShow", "plot, heatmap and colorbar",
        "--zMin", "0", "--zMax", "6",
        "--regionsLabel", "CTCF peaks",
        "--yAxisLabel", "CTCF consensus peaks",
        "--xAxisLabel", "distance from CTCF summit (bp)",
        "--plotTitle", "ATAC CPM at CTCF consensus (rows sorted by mean ATAC)",
        "--refPointLabel", "summit",
    ])
    if not (args.skip_matrix and (out / f"heatmap_kmeans{args.kmeans}.png").is_file()):
        run([
            str(DT / "plotHeatmap"), "-m", str(mtx),
            "-out", str(out / f"heatmap_kmeans{args.kmeans}.png"),
            "--kmeans", str(args.kmeans),
            "--colorMap", "Reds",
            "--whatToShow", "plot, heatmap and colorbar",
            "--zMin", "0", "--zMax", "6",
            "--yAxisLabel", "CTCF consensus peaks",
            "--xAxisLabel", "distance from CTCF summit (bp)",
            "--plotTitle", f"ATAC CPM at CTCF consensus (k-means k={args.kmeans})",
            "--refPointLabel", "summit",
            "--outFileSortedRegions", str(out / f"kmeans{args.kmeans}_regions.bed"),
        ])

    summary = out / "summary.tsv"
    with summary.open("w") as fh:
        fh.write("set\tn\tmedian_atac_cpm\tmean_atac_cpm\tpct_le_0.5\tpct_gt_1\n")
        def _row(name, v):
            v = v[np.isfinite(v)]
            fh.write(f"{name}\t{len(v)}\t{np.median(v):.4g}\t{np.mean(v):.4g}\t"
                     f"{100*(v<=0.5).mean():.1f}\t{100*(v>1).mean():.1f}\n")
        _row("CTCF_peak", mean_peak)
        _row("CTCF_summit_pm2kb", mean_win)
        _row("random_4kb", mean_rand)
        _row("CTCF_TSS_proximal", mean_peak[proximal])
        _row("CTCF_TSS_distal", mean_peak[~proximal])
        _row("CTCF_Promoter", mean_peak[is_prom])
        _row("CTCF_Quiescent", mean_peak[is_quies])

    methods = out / "methods.txt"
    methods.write_text(
        f"""ATAC accessibility at CTCF consensus peaks — methods
peaks:                 {args.peaks}  n_in={n_in}  n_after_blacklist={len(peaks)}
region (heatmap):      CTCF summit ± {FLANK} bp (fixed {2*FLANK} bp window)
region (ECDF/boxplots): original peak interval (variable width)
ATAC tracks:           GSE283384 and GSE152177 pooled BAM → bamCoverage CPM, binSize={BIN}
                       --extendReads, MAPQ≥5, no Tn5 ±4/5 shift applied
CTCF occupancy:        mean of ENCFF139DCW + GSE103651 bamCoverage CPM (deciles)
matrix:                computeMatrix reference-point --referencePoint center
                       -b {FLANK} -a {FLANK} -bs {BIN} --missingDataAsZero
                       (closed peaks kept; --skipZeros NOT set)
heatmap sort:          mean ATAC signal, descending
heatmap clustering:    k-means k={args.kmeans} on the same matrix
blacklist:             hg38-blacklist.v2; peaks overlapping it dropped
background:            bedtools shuffle of the ±2 kb windows, -chrom -seed 1,
                       excluding blacklist (n={len(random_rows)} 4 kb regions)
ChromHMM:              HEK293T 18-state, max-bp overlap; genome-bp as dashed line
TSS:                   HOMER hg38.tss midpoint; proximal = < 2 kb
mappability:           not applied
copy-number / WGS:     not applied (HEK293T caveat)
assembly:              GRCh38/hg38, main chroms chr1–22,X,Y
"""
    )
    print(f"[done] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
