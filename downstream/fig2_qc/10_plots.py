#!/usr/bin/env python
"""Draft Figure 2 QC panels (palette option C, Arial 7 pt). Writes OUT/figs/*.png + .pdf.

  fig2_ref_overlap      NTxNT bp-Jaccard + fraction of peaks overlapping, plus promoter/ATAC/unique fractions
  fig2_matrix_plain_vs_ctrl  AUROC heatmaps (diverging at 0.5): plain vs controlled, depth-matched raw and imputed
  fig2_T4_pseudo        own-TF z vs depth-matched pseudo-TF null per reference
  fig2_Pb_<tag>         clustered Pearson heatmaps with dendrogram
  fig2_T5_profiles      motif-anchored profiles (CTCF, MAZ)
Usage: 10_plots.py [--ctrl T2_matched=spec/T2_{arm}_atac_auroc.tsv] [--pb tag ...]
"""
import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram

from fig2_common import NT, OUT, ROOT, TFS

sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402

P = rp.use()
ap = argparse.ArgumentParser()
ap.add_argument("--plain", default="diag/plain_{arm}_atac_auroc.tsv")
ap.add_argument("--ctrl", default="spec/T2_{arm}_atac_auroc.tsv")
ap.add_argument("--ctrl-label", default="Unique peaks, matched negatives (T2)")
ap.add_argument("--pb", nargs="*", default=["Pb3_link_V1_plain_raw_dm", "Pb3_link_V1_plain_imputed",
                                             "Pb3_link_V2_centred_raw_dm", "Pb3_link_V3_top1000_raw_dm",
                                             "Pb3_link_V4_partial_raw_dm", "Pb_link_distinct3_union_raw_dm",
                                             "Pb_link_all6_unique_raw_dm"])
a = ap.parse_args()
fd = OUT / "figs"
fd.mkdir(exist_ok=True)
INK, GREY = P.neutral.ink, P.neutral.light


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(fd / f"{name}.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)


def heat(ax, M, cmap, vmin, vmax, fmt="{:.2f}", box=True, xl=TFS, yl=TFS):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xl)), xl, rotation=90)
    ax.set_yticks(range(len(yl)), yl)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                continue
            mid = (vmin + vmax) / 2
            dark = abs(v - mid) > 0.35 * (vmax - vmin) if cmap.name.endswith("div") else v > vmin + 0.6 * (vmax - vmin)
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=5, color="white" if dark else INK)
    if box:
        for k in range(min(M.shape)):
            ax.add_patch(Rectangle((k - .5, k - .5), 1, 1, fill=False, ec=INK, lw=0.8))
    return im


# 1. reference overlap
J = pd.read_csv(OUT / "diag/ref_jaccard_bp.tsv", sep="\t", index_col=0).values
F = pd.read_csv(OUT / "diag/ref_frac_overlap.tsv", sep="\t", index_col=0).values
ann = pd.read_csv(OUT / "diag/ref_annotation.tsv", sep="\t").set_index("reference").loc[TFS]
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
Jm = J.copy(); np.fill_diagonal(Jm, np.nan)
im = heat(ax[0], np.where(np.eye(NT, dtype=bool), np.nan, J), P.seq_cmap(), 0, np.nanmax(Jm), box=False)
ax[0].set_title("Peak overlap (bp Jaccard)")
fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.03)
im = heat(ax[1], np.where(np.eye(NT, dtype=bool), np.nan, F), P.seq_cmap(), 0, 1, box=False)
ax[1].set_title("Fraction of row peaks\noverlapping column peaks")
ax[1].set_ylabel("Reference (row)")
fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03)
y = np.arange(NT)
w = 0.27
ax[2].barh(y - w, ann.frac_promoter, w, color=P.chromhmm.promoter, label="at promoter (TSS ±1 kb)")
ax[2].barh(y, ann.frac_atac_union, w, color=P.arms.atac, label="in ATAC union")
ax[2].barh(y + w, ann.frac_unique, w, color=INK, label="unique to this reference")
ax[2].set_yticks(y, [f"{t} (n={n:,})" for t, n in zip(TFS, ann.n_peaks)])
ax[2].invert_yaxis()
ax[2].set_xlim(0, 1)
ax[2].set_xlabel("Fraction of peaks")
ax[2].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=5)
fig.tight_layout()
save(fig, "fig2_ref_overlap")

# 2. plain vs controlled matrices
arms = [("raw_dm", "sc raw (depth-matched)"), ("imputed", "sc imputed")]
summ = pd.read_csv(OUT / "summary_diagonal.tsv", sep="\t")
fig, axs = plt.subplots(2, 2, figsize=(5.6, 6.4), layout="constrained")
for r, (arm, alab) in enumerate(arms):
    for c, (pat, clab) in enumerate(((a.plain, "Plain (all peaks, ATAC bins)"), (a.ctrl, a.ctrl_label))):
        M = pd.read_csv(OUT / pat.format(arm=arm), sep="\t", index_col=0).values
        im = heat(axs[r, c], M, P.div_cmap(), 0.2, 0.8)
        dm = np.diag(M).mean() - M[~np.eye(NT, dtype=bool)].mean()
        own = sum(np.argmax(M[:, j]) == j for j in range(NT))
        axs[r, c].set_title(f"{alab}\n{clab}\ndiag−off {dm:+.3f}, own best {own}/{NT}", fontsize=6)
        axs[r, c].set_xlabel("Bulk reference")
        axs[r, c].set_ylabel("sc track")
fig.colorbar(im, ax=axs, fraction=0.03, pad=0.02, label="AUROC", shrink=0.5)
save(fig, "fig2_matrix_plain_vs_ctrl")

# 3. T4 pseudo-TF null
for scope in ("genome", "atac"):
    Z = pd.read_csv(OUT / f"spec/T4_z_vs_pseudo_{scope}.tsv", sep="\t", index_col=0)
    Lp = pd.read_csv(OUT / f"spec/T4_L_pseudo_{scope}.tsv", sep="\t", index_col=0).values
    Pc = Lp - Lp.mean(1, keepdims=True)
    Pz = (Pc - Pc.mean(0)) / Pc.std(0, ddof=1)
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    rng = np.random.default_rng(0)
    for j, tf in enumerate(TFS):
        ax.scatter(j + rng.uniform(-.18, .18, len(Pz)), Pz[:, j], s=4, color=GREY, lw=0,
                   label="pseudo-TF (pooled panel, depth-matched)" if j == 0 else None)
        oth = np.delete(Z.values[:, j], j)
        ax.scatter(np.full(NT - 1, j + 0.3), oth, s=8, facecolor="none", edgecolor=P.neutral.mid, lw=0.5,
                   label="other TFs" if j == 0 else None)
        ax.scatter([j], [Z.values[j, j]], s=18, color=INK, zorder=3, label="own TF" if j == 0 else None)
    ax.axhline(0, color=GREY, lw=0.5)
    ax.axhline(2, color=GREY, lw=0.5, ls=":")
    ax.set_xticks(range(NT), TFS, rotation=45, ha="right")
    ax.set_xlabel("Reference-unique peak set")
    ax.set_ylabel("Enrichment z vs pseudo-TFs")
    ax.set_title(f"T4 aggregate obs/exp, ZNF777 depth ({scope})")
    ax.legend(fontsize=5, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    save(fig, f"fig2_T4_pseudo_{scope}")

# 4. P-b clustered Pearson
for tag in a.pb:
    f = OUT / f"panels/{tag}.npz"
    if not f.exists():
        continue
    z = np.load(f)
    R, labs, Zl = z["R"], list(z["labs"]), z["Z"]
    n = len(labs)
    fig = plt.figure(figsize=(3.4, 3.4))
    axd = fig.add_axes([0.27, 0.78, 0.53, 0.11])
    d = dendrogram(Zl, ax=axd, no_labels=True, color_threshold=0, above_threshold_color=INK)
    axd.axis("off")
    o = d["leaves"]
    axh = fig.add_axes([0.27, 0.22, 0.53, 0.56])
    lo = max(-1, np.floor(R[np.triu_indices(n, 1)].min() * 10) / 10)
    im = axh.imshow(R[np.ix_(o, o)], cmap=P.div_cmap(), vmin=-1 if lo < 0 else -1, vmax=1, aspect="auto")
    ol = [labs[i].replace("bulk_", "Bulk ").replace("sc_", "sc ") for i in o]
    axh.set_xticks(range(n), ol, rotation=90)
    axh.set_yticks(range(n), ol)
    for t in axh.get_xticklabels() + axh.get_yticklabels():
        t.set_color(INK if t.get_text().startswith("Bulk") else P.arms.raw if "imputed" not in tag else P.arms.imputed)
    axh.tick_params(length=0)
    cax = fig.add_axes([0.83, 0.22, 0.025, 0.25])
    fig.colorbar(im, cax=cax, label="Pearson r")
    k = n // 2
    blk = R[k:, :k]
    own = sum(int(np.argmax(blk[i]) == i) for i in range(k))
    title = tag.replace("Pb3_link_", "").replace("Pb_link_", "").replace("_", " ").replace("raw dm", "raw (ZNF777 depth)")
    fig.suptitle(f"{title}\nown reference = best bulk for {own}/{k} sc tracks", fontsize=6, y=0.995)
    save(fig, f"fig2_{tag.replace('_link', '')}")

# 5. T5 profiles
fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.0))
for ax, tf in zip(axs, ("CTCF", "MAZ")):
    f = OUT / f"spec/T5_profile_{tf}.tsv"
    if not f.exists():
        continue
    df = pd.read_csv(f, sep="\t", index_col=0)
    x = df.columns.astype(float)
    for t in df.index:
        if t == tf:
            ax.plot(x, df.loc[t], color=INK, lw=1.0, label=f"{t} (own)", zorder=3)
        elif t == "pooled79":
            ax.plot(x, df.loc[t], color=P.neutral.mid, lw=0.8, ls="--", label="79-TF pool")
        else:
            ax.plot(x, df.loc[t], color=GREY, lw=0.5, label="other TFs" if t == [u for u in TFS if u != tf][0] else None)
    ax.set_xlabel(f"Distance to {tf} motif (bp)")
    ax.set_ylabel("Reads per million per site")
    ax.set_title(f"{tf}-unique peaks with motif")
    ax.legend(fontsize=5, frameon=False)
fig.tight_layout()
save(fig, "fig2_T5_profiles")
# 6. split-half noise ceiling for region-level sc-vs-bulk r
f = OUT / "panels/split_half_ceiling.tsv"
if f.exists():
    c = pd.read_csv(f, sep="\t")
    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    y = np.arange(NT)
    for k, (dep, ver, col, lab) in enumerate((("full", "plain", P.neutral.mid, "full depth, plain"),
                                             ("znf777", "plain", P.arms.raw, "ZNF777 depth, plain"),
                                             ("znf777", "centred", INK, "ZNF777 depth, TF-specific part (centred)"))):
        v = c[(c.depth == dep) & (c.version == ver)].set_index("tf").loc[TFS, "max_r_ceiling"]
        ax.barh(y + (k - 1) * 0.27, v, 0.27, color=col, label=lab)
    ax.set_yticks(y, TFS)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Max attainable r with any reference\n(split-half reliability, Spearman-Brown)")
    ax.legend(fontsize=5, loc="lower right")
    fig.tight_layout()
    save(fig, "fig2_split_half_ceiling")
print("done")
