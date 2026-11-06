#!/usr/bin/env python3
"""Palette C copy of plot_hub_browser.py (colours only). Genome-browser view of one multi-TF clique hub (Signac CoveragePlot / pyGenomeTracks style).

Tracks, top to bottom, on one shared chromosome axis:
  Hi-C triangle (balanced, log10) -> Peakachu loops (arcs) -> shared clique edges (arcs; width = no. of TFs,
  red = on a loop) -> per-TF read coverage (CPM per bin; top TFs, plus all other hub TFs pooled) -> RefSeq genes.
Hub elements are shaded through every track. A second figure zooms into chosen windows.
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import cooler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
for _p in Path(__file__).resolve().parents:
    if (_p / "palette").is_dir():
        sys.path.insert(0, str(_p / "palette"))
import regulome_palette as rp

P = rp.use()
# keep the browser's original geometry and type sizes (palette C colours, font, line and text style)
plt.rcParams.update({"font.size": 11, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10})
EDGE, LOOPC, SHADE = P.evidence.cicero_only, P.evidence.hic_loop, P.highlight
RAW, OTHER = P.arms.raw, P.neutral.dark


def parse(r):
    c, se = r.split(":")
    s, e = se.split("-")
    return c, int(s), int(e)


def arc(ax, x1, x2, h, color, lw, down=False, alpha=0.85):
    t = np.linspace(0, np.pi, 100)
    xs = (x1 + x2) / 2 - (x2 - x1) / 2 * np.cos(t)
    ax.plot(xs, (-1 if down else 1) * h * np.sin(t), color=color, lw=lw, alpha=alpha, solid_capstyle="round")


def read_window(bed, chrom, lo, hi):
    """5' ends of reads in [lo, hi) plus the TF's total read count (for CPM)."""
    pos, n = [], 0
    with gzip.open(bed, "rt") as fh:
        for ln in fh:
            n += 1
            f = ln.split("\t", 6)
            if f[0] != chrom:
                continue
            p = int(f[1]) if f[5].startswith("+") else int(f[2]) - 1
            if lo <= p < hi:
                pos.append(p)
    return np.array(pos), n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edges", type=Path, required=True, help="shared edges of the hub (region1, region2, n_tfs, on_loop_0.7, tfs)")
    ap.add_argument("--genes", type=Path, required=True)
    ap.add_argument("--mcool", type=Path, required=True)
    ap.add_argument("--loops", type=Path, nargs="+", required=True)
    ap.add_argument("--frag-dir", type=Path, required=True)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--zoom", nargs="*", default=[], help="chrom:start-end windows for the zoom figure")
    ap.add_argument("--title", default="")
    ap.add_argument("--imputed", type=Path, default=None, help="tsv tf/chrom/start/frac_cells (1 kb imputed bins); replaces raw coverage")
    ap.add_argument("--per-track-scale", action="store_true", help="each coverage track gets its own y-limit (99.5th pct in the window)")
    ap.add_argument("--chromhmm", type=Path, default=None, help="18-state ChromHMM bed(.gz); adds a state track and colours hub bands by class")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    E = pd.read_csv(a.edges, sep="\t").rename(columns={"on_loop_0.7": "loop"})
    E["tfl"] = E.tfs.str.split(",")
    elems = sorted({r for c in ("region1", "region2") for r in E[c]}, key=lambda r: parse(r)[1])
    chrom = parse(elems[0])[0]
    s0, e0 = parse(elems[0])[1], parse(elems[-1])[2]
    pad = int(0.05 * (e0 - s0))
    lo, hi = s0 - pad, e0 + pad
    tf_n = pd.Series([t for l in E.tfl for t in l]).value_counts()
    top = list(tf_n.index[: a.top])
    rest = list(tf_n.index[a.top:])

    IMP = None
    if a.imputed:
        IMP = pd.read_csv(a.imputed, sep="\t"); IMP = IMP[IMP.chrom == chrom]
    # coverage: 5' ends per TF in the window (whole zoom windows are inside it)
    cov, tot = {}, {}
    for tf in tf_n.index:
        cov[tf], tot[tf] = read_window(a.frag_dir / f"TF.{tf}.bed.gz", chrom, lo, hi)
        print(tf, len(cov[tf]), tot[tf], flush=True)

    loops = []
    for f in a.loops:
        for ln in open(f):
            p = ln.split()
            if p[0] == chrom:
                m1, m2 = (int(p[1]) + int(p[2])) // 2, (int(p[4]) + int(p[5])) // 2
                if lo <= m1 and m2 <= hi:
                    loops.append((m1, m2))
    G = pd.read_csv(a.genes, sep="\t")

    def track_coverage(ax, x0, x1, binsz, label_side=True):
        rows = [(t, None, None) for t in top]
        if rest:
            rows.append((f"other {len(rest)} TFs", None, None))
        if IMP is not None:   # imputed: fraction of cells per 1 kb bin (others = mean of the rest)
            edges = np.arange((x0 // 1000) * 1000, x1 + 1000, 1000)
            hs = []
            for name, _, _ in rows:
                ts = rest if name.startswith("other") else [name]
                d = IMP[IMP.tf.isin(ts) & (IMP.start >= edges[0]) & (IMP.start < edges[-1])]
                v = d.groupby("start").frac_cells.mean().reindex(edges[:-1], fill_value=0).values
                hs.append(v)
        else:
            edges = np.arange(x0, x1 + binsz, binsz)
            hs = []
            for name, _, _ in rows:
                ts = rest if name.startswith("other") else [name]
                p = np.concatenate([cov[t] for t in ts]); n = sum(tot[t] for t in ts)
                h, _ = np.histogram(p, bins=edges); hs.append(h / n * 1e6)
        # per-track range: the highest bin over a hub element in this window (so hub peaks fill the track);
        # stray single-read spikes elsewhere are clipped. Falls back to the 99.5th percentile.
        ctr = (edges[:-1] + edges[1:]) / 2
        onhub = np.zeros(len(ctr), bool)
        for r in elems:
            _, es, ee = parse(r)
            onhub |= (ctr >= es - 500) & (ctr <= ee + 500)
        lims = [max(h[onhub].max() if onhub.any() else 0, np.percentile(h, 99.5), 1e-9) * 1.05 for h in hs]
        ymax = max(lims)
        return rows, hs, edges, (lims if a.per_track_scale else [ymax] * len(hs))

    el_mid = [(parse(r)[1] + parse(r)[2]) / 2 for r in elems]

    def pack_genes(x0, x1, width_in, fs):
        """Lane per gene so that neither the gene body nor its label overlaps another (pyGenomeTracks-style)."""
        g = G[(G.txEnd > x0) & (G.txStart < x1)].sort_values("txStart")
        bp_per_in = (x1 - x0) / width_in
        lanes, out = [], []
        for r in g.itertuples():
            s, e = max(r.txStart, x0), min(r.txEnd, x1)
            lw = len(r.name) * fs * 0.62 / 72 * bp_per_in
            xm = (s + e) / 2
            L, R = min(s, xm - lw / 2), max(e, xm + lw / 2)
            lane = next((k for k, end in enumerate(lanes) if L > end + 0.004 * (x1 - x0)), len(lanes))
            if lane == len(lanes):
                lanes.append(0)
            lanes[lane] = R
            tss = r.txStart if r.strand == "+" else r.txEnd
            hub = any(abs(tss - m) <= 1500 for m in el_mid)
            out.append((r, lane, xm, hub))
        return out, max(len(lanes), 1)

    def draw_genes(ax, x0, x1, packed, nl, fs=9):
        for r, lane, xm, hub in packed:
            y = -lane * 1.25
            col = P.chromhmm.promoter if hub else P.neutral.dark
            ax.plot([max(r.txStart, x0), min(r.txEnd, x1)], [y, y], color=col, lw=0.9)
            for s, e in zip(map(int, str(r.exonStarts).strip(",").split(",")), map(int, str(r.exonEnds).strip(",").split(","))):
                if e > x0 and s < x1:
                    ax.add_patch(Rectangle((s, y - 0.2), max(e - s, 0.0006 * (x1 - x0)), 0.4, color=col, lw=0))
            tss = r.txStart if r.strand == "+" else r.txEnd
            if x0 <= tss <= x1:
                d = (1 if r.strand == "+" else -1) * 0.006 * (x1 - x0)
                ax.plot([tss, tss], [y, y + 0.42], color=col, lw=0.8)
                ax.annotate("", xy=(tss + d, y + 0.42), xytext=(tss, y + 0.42),
                            arrowprops=dict(arrowstyle="-|>", lw=0.8, color=col, mutation_scale=6))
            ax.text(xm, y - 0.28, r.name, ha="center", va="top", fontsize=fs, style="italic", color=col,
                    fontweight="bold" if hub else "normal")
        ax.set_ylim(-(nl - 1) * 1.25 - 1.0, 0.8)
        ax.axis("off")

    GROUP = {"EnhA1": "enhancer", "EnhA2": "enhancer", "EnhG1": "enhancer", "EnhG2": "enhancer", "EnhWk": "enhancer", "EnhBiv": "enhancer",
             "TssA": "TSS/flank", "TssFlnk": "TSS/flank", "TssFlnkU": "TSS/flank", "TssFlnkD": "TSS/flank", "TssBiv": "TSS/flank",
             "Tx": "transcribed", "TxWk": "transcribed", "ReprPC": "repressed", "ReprPCWk": "repressed", "Het": "repressed", "ZNF/Rpts": "repressed"}
    GCOL = {"enhancer": P.chromhmm.active_enh, "TSS/flank": P.chromhmm.tss_flank, "transcribed": P.chromhmm.transcribed, "repressed": P.chromhmm.repressed, "quiescent": P.chromhmm.quiescent}
    BAND = {"promoter": P.highlight, "enhancer": P.highlight, "other": P.highlight}
    states = []
    if a.chromhmm:
        op = gzip.open if str(a.chromhmm).endswith(".gz") else open
        with op(a.chromhmm, "rt") as fh:
            for ln in fh:
                f = ln.split("\t")
                if f[0] == chrom and int(f[2]) > lo and int(f[1]) < hi:
                    states.append((int(f[1]), int(f[2]), GROUP.get(f[3], "quiescent")))
    tss = [(r.txStart if r.strand == "+" else r.txEnd) for r in G.itertuples()]
    def el_class(r):
        _, s, e = parse(r)
        if any(s - 1000 <= t <= e + 1000 for t in tss): return "promoter"
        m = (s + e) // 2
        g = next((st for a0, a1, st in states if a0 <= m < a1), "quiescent")
        return "enhancer" if g == "enhancer" else "other"
    ecls = {r: el_class(r) for r in elems}
    print("element classes:", pd.Series(ecls).value_counts().to_dict())

    def draw_states(ax, x0, x1):
        for a0, a1, g in states:
            if a1 > x0 and a0 < x1 and g != "quiescent":
                ax.add_patch(Rectangle((max(a0, x0), 0.15), min(a1, x1) - max(a0, x0), 0.7, color=GCOL[g], lw=0))
        ax.set_ylim(0, 1); ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(P.neutral.light); ax.tick_params(axis="x", labelbottom=False, bottom=False)

    def shade(axs, x0, x1):
        for r in elems:
            _, s, e = parse(r)
            if e > x0 and s < x1:
                w = max(e - s, 0.002 * (x1 - x0))
                c = BAND[ecls[r]] if states else SHADE
                for ax in axs:
                    ax.axvspan(s - (w - (e - s)) / 2, s - (w - (e - s)) / 2 + w, color=c, zorder=0, lw=0)

    def label(ax, text):
        ax.text(-0.01, 0.5, text, transform=ax.transAxes, ha="right", va="center", fontsize=10.5)

    # ---------------- full view ----------------
    nrow_cov = len(top) + (1 if rest else 0)
    gpack, gnl = pack_genes(lo, hi, 15 * 0.775, 8)
    NS = 1 if states else 0
    heights = [3.2, 0.9, 1.6] + [0.42] * nrow_cov + [0.38] * NS + [0.3 * gnl + 0.3]
    fig, axs = plt.subplots(len(heights), 1, figsize=(15, sum(heights) + 1.2), sharex=True,
                            gridspec_kw={"height_ratios": heights, "hspace": 0.08})
    # Hi-C triangle
    clr = cooler.Cooler(f"{a.mcool}::/resolutions/10000")
    M = clr.matrix(balance=True).fetch(f"{chrom}:{lo}-{hi}")
    M = np.log10(np.nan_to_num(M) + 1e-5)
    n = M.shape[0]
    b = clr.binsize
    x0b = (lo // b) * b
    X, Y = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
    xs = x0b + (X + Y) / 2 * b
    ys = (Y - X) / 2 * b
    Mu = M.copy()
    Mu[np.tril_indices(n, -1)] = np.nan
    axh = axs[0]
    axh.pcolormesh(xs, ys, np.ma.masked_invalid(Mu.T), cmap=P.seq_cmap(), vmin=np.nanpercentile(M, 50),
                   vmax=np.nanpercentile(M, 99.5), shading="flat", rasterized=True)
    axh.set_ylim(0, (hi - lo) / 2.2)
    axh.set_yticks([])
    for sp in axh.spines.values():
        sp.set_visible(False)
    label(axh, "Hi-C (WT, 10 kb)")
    # loops
    axl = axs[1]
    for m1, m2 in loops:
        arc(axl, m1, m2, (m2 - m1) / (hi - lo), P.neutral.light, 0.8, down=True, alpha=1.0)
    axl.set_ylim(-0.75, 0.02)
    axl.axis("off")
    label(axl, f"loops ≥0.7 (n={len(loops)})")
    # shared edges
    axe = axs[2]
    for r in E.sort_values("n_tfs").itertuples():
        m1, m2 = sorted(((parse(r.region1)[1] + parse(r.region1)[2]) / 2, (parse(r.region2)[1] + parse(r.region2)[2]) / 2))
        arc(axe, m1, m2, (m2 - m1) / (hi - lo), LOOPC if r.loop else EDGE, 0.6 + 0.3 * r.n_tfs)
    axe.set_ylim(0, 0.62)
    axe.axis("off")
    label(axe, "shared clique edges")
    axe.plot([], [], color=EDGE, lw=3, label=f"shared by ≥{E.n_tfs.min()} TFs' cliques")
    axe.plot([], [], color=LOOPC, lw=3, label="… and on a 0.7 loop (±30 kb)")
    axe.legend(loc="upper left", frameon=False, fontsize=9.5, ncol=1)
    # coverage
    rows, hs, bins, ylims = track_coverage(None, lo, hi, 2000); ymax = max(ylims)
    cols = plt.get_cmap("tab10")
    for k, ((name, _, _), h) in enumerate(zip(rows, hs)):
        ax = axs[3 + k]
        c = OTHER if name.startswith("other") else RAW
        ax.fill_between(bins[:-1], 0, np.minimum(h, ylims[k]), step="post", color=c, lw=0)
        ax.set_ylim(0, ylims[k])
        ax.set_yticks([])
        if a.per_track_scale:
            ax.text(1.002, 0.95, f"{ylims[k] * (100 if IMP is not None else 1):.2g}", transform=ax.transAxes, fontsize=7.5, color=P.neutral.dark, va="top")
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(P.neutral.light)
        label(ax, name if name.startswith("other") else f"{name} ({tf_n[name]})")
    covtxt = ("imputed signal: % of cells with the 1 kb bin (unified model)" if IMP is not None else "coverage: read 5' ends, CPM per 2 kb")
    covtxt += ("; each track scaled to its own maximum (top-right number)" if a.per_track_scale else f"; same scale 0–{ymax:.3g} for all tracks")
    axs[3 + len(rows) - 1].text(1.0, -0.35, covtxt + "; (n) = shared edges the TF is in",
                                transform=axs[3 + len(rows) - 1].transAxes, ha="right", va="top", fontsize=9, color=P.neutral.dark)
    if states:
        draw_states(axs[-2], lo, hi)
        label(axs[-2], "ChromHMM (HEK293T)")
        axs[-1].legend(handles=[Rectangle((0, 0), 1, 1, color=GCOL[g]) for g in ("enhancer", "TSS/flank", "transcribed", "repressed")]
                       + [Rectangle((0, 0), 1, 1, color=P.highlight)],
                       labels=["enhancer", "TSS/flank", "transcribed", "repressed", "band: hub element"],
                       ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False, fontsize=9, handlelength=1.2)
    # genes
    draw_genes(axs[-1], lo, hi, gpack, gnl, fs=8)
    label(axs[-1], "RefSeq genes\n(red = hub promoter)")
    axs[-1].axis("on")
    for sp in ("top", "right", "left"):
        axs[-1].spines[sp].set_visible(False)
    axs[-1].set_yticks([])
    axs[-1].set_xlim(lo, hi)
    axs[-1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.2f}"))
    axs[-1].set_xlabel(f"{chrom} position (Mb, hg38)")
    shade(axs[1:], lo, hi)
    fig.suptitle(a.title or f"{chrom}:{lo:,}-{hi:,}", x=0.125, ha="left", y=0.995, fontsize=13)
    fig.savefig(a.out.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(a.out.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---------------- zoom views ----------------
    if a.zoom:
        zp = [pack_genes(*parse(z)[1:], 7.2 * 0.75, 9) for z in a.zoom]
        NS = 1 if states else 0
        heights = [1.3] + [0.42] * nrow_cov + [0.38] * NS + [0.32 * max(n for _, n in zp) + 0.4]
        fig, axs = plt.subplots(len(heights), len(a.zoom), figsize=(7.2 * len(a.zoom), sum(heights) + 1.0),
                                gridspec_kw={"height_ratios": heights, "hspace": 0.08, "wspace": 0.12}, squeeze=False)
        for j, z in enumerate(a.zoom):
            _, zs, ze = parse(z)
            col = axs[:, j]
            for ax in col:
                ax.set_xlim(zs, ze)
            # edges leaving the window: stubs pointing to the partner
            axe = col[0]
            for r in E.itertuples():
                for p_, q_ in ((r.region1, r.region2), (r.region2, r.region1)):
                    pm = (parse(p_)[1] + parse(p_)[2]) / 2
                    qm = (parse(q_)[1] + parse(q_)[2]) / 2
                    if zs <= pm <= ze:
                        if zs <= qm <= ze:
                            if pm < qm:
                                arc(axe, pm, qm, 0.8 * (qm - pm) / (ze - zs), LOOPC if r.loop else EDGE, 0.6 + 0.3 * r.n_tfs)
                        else:
                            xe = ze if qm > ze else zs
                            axe.plot([pm, xe], [0, 0.9], color=LOOPC if r.loop else EDGE, lw=0.6 + 0.3 * r.n_tfs, alpha=0.6)
            axe.set_ylim(0, 1.0)
            axe.axis("off")
            axe.set_title(f"{chrom}:{zs:,}-{ze:,}", fontsize=11.5, loc="left")
            rows, hs, bins, ylims = track_coverage(None, zs, ze, 500); ymax = max(ylims)
            for k, ((name, _, _), h) in enumerate(zip(rows, hs)):
                ax = col[1 + k]
                c = OTHER if name.startswith("other") else RAW
                ax.fill_between(bins[:-1], 0, np.minimum(h, ylims[k]), step="post", color=c, lw=0)
                ax.set_ylim(0, ylims[k])
                ax.set_yticks([])
                for sp in ("top", "right", "left"):
                    ax.spines[sp].set_visible(False)
                ax.spines["bottom"].set_color(P.neutral.light)
                ax.set_xticks([])
                if j == 0:
                    label(ax, name)
            col[-2 - NS].text(1.0, -0.3, ("imputed, % cells per 1 kb" if IMP is not None else "CPM per 500 bp") + ("; per-track scale" if a.per_track_scale else f", 0–{ymax:.3g}"), transform=col[-2 - NS].transAxes, ha="right", va="top", fontsize=9, color=P.neutral.dark)
            if states:
                draw_states(col[-2], zs, ze)
                if j == 0:
                    label(col[-2], "ChromHMM")
            draw_genes(col[-1], zs, ze, *zp[j], fs=9)
            col[-1].axis("on")
            for sp in ("top", "right", "left"):
                col[-1].spines[sp].set_visible(False)
            col[-1].set_yticks([])
            col[-1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.3f}"))
            col[-1].set_xlabel(f"{chrom} (Mb, hg38)")
            shade(col, zs, ze)
            if j == 0:
                label(axe, "shared edges")
        fig.savefig(a.out.with_name(a.out.stem + "_zoom.png"), dpi=600, bbox_inches="tight", facecolor="white")
        fig.savefig(a.out.with_name(a.out.stem + "_zoom.svg"), bbox_inches="tight", facecolor="white")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
