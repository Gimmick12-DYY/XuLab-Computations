#!/usr/bin/env python
"""Figure 2 (QC) draft: assembled multi-panel PNG/PDF + per-panel SVGs, palette C, Arial 7 pt. Read-only on results.
  a  browser: HEK293T ChIP / sc raw / sc imputed / 79-TF pool at 2 CTCF and 2 ZNF282 loci (hek293t_2x2/browser_loci.tsv)
  b  sc vs HEK293T ChIP Pearson over summit groups at ZNF777 depth: 2x2 block + clustered 4x4 (hek293t_2x2/corr4x4_all.tsv)
  c  CTCF motif-centred aggregate profile at CTCF-unique sites (spec/T5_profile_CTCF.tsv)
  d  sensitivity: own-reference AUROC / AUPRC on 1 kb bins, raw vs imputed vs ATAC (diag/plain_*_genome_*.tsv)
  e  motif table (CTCF, MAZ; values from TF1000cells_motifs_local.xlsx)
  Supplement S1: 6x6 raw summit-group Pearson (references labelled by cell line); S2: imputed audit step table.
Outputs OUT/fig2_draft/."""
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from fig2_common import BIN, D, INP, OUT, ROOT, TFS, bin_index, load_imp7, read_bed

for f in (ROOT.parent / "fonts/arial").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402

P = rp.use()
plt.rcParams["font.sans-serif"] = ["Arial"] + plt.rcParams["font.sans-serif"]
plt.rcParams["svg.fonttype"] = "none"
INK, MID, LIGHT = P.neutral.ink, P.neutral.mid, P.neutral.light
OD = OUT / "fig2_draft"
OD.mkdir(exist_ok=True)
CELL = {"CTCF": "mixed", "MAZ": "HEK293", "ZBTB7A": "HEK293", "ZIC2": "HEK293", "ZNF282": "HEK293T", "ZNF777": "HEK293"}


def sh(*a):
    return subprocess.run(a, check=True, capture_output=True, text=True).stdout


# ---------- data for a ----------
bi, off, N = bin_index()
fr = np.load(INP / "frame.npz", allow_pickle=True)
tf79 = list(fr["tf79"])
ncell = fr["ncells_imp79"]
imp = load_imp7()
L2 = pd.read_csv(OUT / "hek293t_2x2/browser_loci.tsv", sep="\t")
BAM2 = {"CTCF": ROOT / "chip_gse103651/work/bam/gse103651.pooled.bam", "ZNF282": D / "ZNF282_HEK293.bam"}
PK = {"CTCF": D / "CTCF_HEK293T_GSE103651_peaks.bed", "ZNF282": D / "ZNF282_HEK293_peaks.bed"}
HALF, STEP = 5000, 50
edges = np.arange(-HALF, HALF + 1, STEP)
cx = (edges[:-1] + edges[1:]) / 2
five_tot, pool_hist = {}, []
mapped = {t: sum(int(l.split("\t")[2]) for l in sh("samtools", "idxstats", str(BAM2[t])).splitlines()) for t in BAM2}
tracks = []
for r in L2.itertuples():
    a0, a1 = r.pos - HALF, r.pos + HALF
    dep = sh("samtools", "depth", "-a", "-r", f"{r.chrom}:{a0 + 1}-{a1}", str(BAM2[r.tf]))
    dv = np.array([int(l.split("\t")[2]) for l in dep.splitlines()], float)
    dv = np.pad(dv, (0, 2 * HALF - len(dv)))[:2 * HALF]
    pool = np.zeros(len(cx)); ptot = 0; own = None; otot = 0
    for t in tf79:
        z = np.load(INP / "five" / f"{t}.npz")
        q = z[r.chrom]
        tot = sum(len(z[c]) for c in z.files)
        h = np.histogram(q[(q >= a0) & (q < a1)] - r.pos, edges)[0]
        pool += h; ptot += tot
        if t == r.tf:
            own, otot = h, tot
    o = off[r.chrom][0]
    b0, b1 = a0 // BIN, (a1 - 1) // BIN
    tracks.append(dict(tf=r.tf, chrom=r.chrom, pos=r.pos, bulk=dv.reshape(-1, STEP).mean(1) / mapped[r.tf] * 1e6,
                       raw=own / otot * 1e6, pool=pool / ptot * 1e6,
                       imp=imp[TFS.index(r.tf), o + b0:o + b1 + 1] / ncell[tf79.index(r.tf)] * 100,
                       impx=np.arange(b0, b1 + 2) * BIN - r.pos))
pks = {t: read_bed(PK[t]) for t in PK}


def panel_a(fig, ss):
    gs = ss.subgridspec(4, len(tracks), hspace=0.25, wspace=0.35)
    rows = [("bulk", "HEK293T\nChIP (CPM)", INK), ("raw", "sc raw\n(CPM)", P.arms.raw),
            ("imp", "sc imputed\n(% cells)", P.arms.imputed), ("pool", "79-TF pool\n(CPM)", MID)]
    for c, d in enumerate(tracks):
        same = [e for e in tracks if e["tf"] == d["tf"]]
        for ri, (nm, lab, col) in enumerate(rows):
            ax = fig.add_subplot(gs[ri, c])
            ymax = max(e[nm].max() for e in same) * 1.1 or 1
            if nm == "imp":
                ax.stairs(d["imp"], d["impx"], fill=True, color=col, lw=0)
            else:
                ax.fill_between(cx, d[nm], step="mid", color=col, lw=0)
            ax.set_ylim(0, ymax); ax.set_xlim(-HALF, HALF)
            ax.set_yticks([0, float(f"{ymax / 1.1:.2g}")])
            ax.tick_params(labelsize=5)
            if c == 0:
                ax.set_ylabel(lab, fontsize=5.5)
            if ri == 0:
                pk = pks[d["tf"]]
                pk = pk[(pk[0] == d["chrom"]) & (pk[2] > d["pos"] - HALF) & (pk[1] < d["pos"] + HALF)]
                for s, e in zip(pk[1], pk[2]):
                    ax.axvspan(s - d["pos"], e - d["pos"], color=P.highlight, lw=0, zorder=0)
                ax.set_title(f"{d['tf']}\n{d['chrom']}:{d['pos']:,}", fontsize=5.5)
            if ri < 3:
                ax.set_xticks([])
            else:
                ax.set_xticks([-HALF, 0, HALF], ["-5 kb", "summit", "+5 kb"], fontsize=5)


def heat(ax, M, xl, yl, lim, outline=True, fs=5):
    im = ax.imshow(M, cmap=P.div_cmap(), vmin=-lim, vmax=lim)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=fs,
                    color="white" if abs(M[i, j]) > 0.6 * lim else INK)
    if outline:
        for i in range(min(M.shape)):
            ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=INK, lw=0.6))
    ax.set_xticks(range(M.shape[1]), xl, rotation=90, fontsize=5.5)
    ax.set_yticks(range(M.shape[0]), yl, fontsize=5.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


R4 = pd.read_csv(OUT / "hek293t_2x2/corr4x4_all.tsv", sep="\t", index_col=0)


def panel_b(fig, ss):
    gs = ss.subgridspec(1, 2, wspace=0.9, width_ratios=[1, 1.5])
    ax = fig.add_subplot(gs[0])
    B = R4.values[2:, :2]
    heat(ax, B, ["HEK293T ChIP\nCTCF", "HEK293T ChIP\nZNF282"], ["sc CTCF", "sc ZNF282"], 1, fs=6)
    ax.set_title("sc vs HEK293T ChIP\n(Pearson r)", fontsize=6)
    g2 = gs[1].subgridspec(2, 1, height_ratios=[0.25, 1], hspace=0.02)
    axd = fig.add_subplot(g2[0]); axh = fig.add_subplot(g2[1])
    R = R4.values
    Z = linkage(squareform(np.clip(1 - R, 0, None), checks=False), "average")
    d = dendrogram(Z, ax=axd, no_labels=True, color_threshold=0, above_threshold_color=INK)
    axd.axis("off")
    o = d["leaves"]
    labs = [R4.index[i].replace("HEK293T ChIP ", "ChIP ") for i in o]
    im = heat(axh, R[np.ix_(o, o)], labs, labs, 1, outline=False)
    for t in axh.get_xticklabels() + axh.get_yticklabels():
        t.set_color(INK if t.get_text().startswith("ChIP") else P.arms.raw)
    fig.colorbar(im, ax=axh, fraction=0.05, pad=0.04).ax.tick_params(labelsize=5)


T5 = pd.read_csv(OUT / "spec/T5_profile_CTCF.tsv", sep="\t", index_col=0)


def panel_c(fig, ss):
    ax = fig.add_subplot(ss)
    x = T5.columns.astype(float)
    for t in T5.index:
        if t == "CTCF":
            continue
        ax.plot(x, T5.loc[t], color=LIGHT if t != "pooled79" else MID, lw=0.5 if t != "pooled79" else 0.8,
                ls="-" if t != "pooled79" else "--")
    ax.plot(x, T5.loc["CTCF"], color=INK, lw=1.0)
    ax.plot([], [], color=INK, lw=1, label="sc CTCF"); ax.plot([], [], color=LIGHT, lw=0.5, label="other 5 TFs")
    ax.plot([], [], color=MID, lw=0.8, ls="--", label="79-TF pool")
    ax.legend(fontsize=5, frameon=False, loc="upper right")
    ax.set_xlabel("Distance to CTCF motif (bp)", fontsize=6); ax.set_ylabel("Reads per million per site", fontsize=6)
    ax.set_title("CTCF-unique sites with CTCF motif", fontsize=6); ax.tick_params(labelsize=5)


def own(f):
    M = pd.read_csv(OUT / f"diag/{f}", sep="\t", index_col=0)
    return np.diag(M.values)


SENS = {m: {"raw": own(f"plain_raw_genome_{m}.tsv"), "imputed": own(f"plain_imputed_genome_{m}.tsv"),
            "ATAC": pd.read_csv(OUT / "diag/plain_atac_baseline.tsv", sep="\t", index_col=0).loc[TFS, f"{m}_genome"].values}
        for m in ("auroc", "auprc")}


def panel_d(fig, ss):
    gs = ss.subgridspec(1, 2, wspace=0.35)
    xl = [f"{t}\n({CELL[t]})" for t in TFS]
    for k, m in enumerate(("auroc", "auprc")):
        ax = fig.add_subplot(gs[k])
        w = 0.27
        for j, (nm, col) in enumerate((("raw", P.arms.raw), ("imputed", P.arms.imputed), ("ATAC", P.arms.atac))):
            ax.bar(np.arange(6) + (j - 1) * w, SENS[m][nm], w, color=col, label=f"sc {nm}" if nm != "ATAC" else "ATAC only")
        ax.set_xticks(range(6), xl, rotation=90, fontsize=5)
        ax.set_ylabel(m.upper() + " vs own reference", fontsize=6)
        if m == "auroc":
            ax.set_ylim(0.5, 1)
        else:
            ax.legend(fontsize=5, frameon=False, loc="upper right")
        ax.tick_params(labelsize=5)
    fig.text(ss.get_position(fig).x0, ss.get_position(fig).y1 + 0.005, "1 kb bins, genome-wide", fontsize=5.5)


MOTIF = [("CTCF", "CTCF", "1e-39", "1e-68", "1e-101"), ("MAZ", "MAZ", "1.9e-12 (AME)", "1e-8", "1e-159")]


def panel_e(fig, ss):
    ax = fig.add_subplot(ss); ax.axis("off")
    cols = ["TF", "Motif", "Raw peaks", "Raw bins", "Imputed peaks"]
    rows = [list(m) for m in MOTIF]
    tb = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(5.5); tb.scale(1, 1.4)
    for (i, j), c in tb.get_celld().items():
        c.set_linewidth(0.4); c.set_edgecolor(LIGHT)
        if i == 0:
            c.set_text_props(weight="bold")
    ax.set_title("Known-motif enrichment (HOMER p unless noted)", fontsize=6)


PANELS = [("a", panel_a, (7.2, 3.0)), ("b", panel_b, (4.2, 2.0)), ("c", panel_c, (2.6, 2.0)),
          ("d", panel_d, (4.4, 2.2)), ("e", panel_e, (3.2, 1.0))]
for nm, fn, size in PANELS:
    f = plt.figure(figsize=size)
    gs = f.add_gridspec(1, 1, left=0.12, right=0.97, bottom=0.25, top=0.85)
    fn(f, gs[0])
    for ext in ("svg", "png"):
        f.savefig(OD / f"panel_{nm}.{ext}", dpi=300)
    plt.close(f)

fig = plt.figure(figsize=(7.2, 9.4))
G = fig.add_gridspec(4, 2, height_ratios=[1.35, 1.0, 1.05, 0.55], hspace=0.65, wspace=0.35,
                     left=0.09, right=0.98, top=0.97, bottom=0.05)
regions = {"a": G[0, :], "b": G[1, 0], "c": G[1, 1], "d": G[2, :], "e": G[3, 0]}
for nm, fn, _ in PANELS:
    fn(fig, regions[nm])
    pos = regions[nm].get_position(fig)
    fig.text(pos.x0 - 0.06, pos.y1 + 0.01, nm, fontsize=9, weight="bold")
for ext in ("png", "pdf", "svg"):
    fig.savefig(OD / f"fig2_draft.{ext}", dpi=300)
plt.close(fig)

# ---------- supplement ----------
R6 = pd.read_csv(OUT / "round2/pearson/r2p_corr_preferred_top5000_znf777.tsv", sep="\t", index_col=0).values[6:, :6]
st = pd.read_csv(OUT / "imputed_audit/diagnostic_steps.tsv", sep="\t")
st = st[st.summit_set == "CTCF"].pivot_table(index="tf", columns="step", values="own_over_mean_other5").loc[TFS]
order = ["raw_positive_cell_frac", "continuous_mean_score", "frac_cells_above_threshold_premask", "unmasked_binarized_frac", "masked_final_frac"]
st = st[order]
fs = plt.figure(figsize=(7.2, 3.2))
gs = fs.add_gridspec(1, 2, wspace=0.6, left=0.12, right=0.97, bottom=0.32, top=0.85)
ax = fs.add_subplot(gs[0])
heat(ax, R6, [f"{t} ({CELL[t]})" for t in TFS], [f"sc {t}" for t in TFS], max(0.1, np.abs(R6).max()))
ax.set_title("S1  sc raw vs bulk Pearson, summit groups,\nZNF777 depth (own-reference cells outlined)", fontsize=6)
ax = fs.add_subplot(gs[1])
im = ax.imshow(np.log2(st.values), cmap=P.div_cmap(), vmin=-1.6, vmax=1.6)
for i in range(st.shape[0]):
    for j in range(st.shape[1]):
        ax.text(j, i, f"{st.values[i, j]:.2f}", ha="center", va="center", fontsize=5)
ax.set_xticks(range(5), ["raw cells", "model score", "above cutoff", "binarized", "masked final"], rotation=90, fontsize=5.5)
ax.set_yticks(range(6), [f"imputed {t}" for t in TFS], fontsize=5.5)
ax.tick_params(length=0)
ax.set_title("S2  At CTCF summits: own / mean of other 5,\nby imputation step", fontsize=6)
for ext in ("png", "pdf", "svg"):
    fs.savefig(OD / f"fig2_supplement_draft.{ext}", dpi=300)
print("done")
