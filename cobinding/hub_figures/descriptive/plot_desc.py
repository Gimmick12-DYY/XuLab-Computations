"""Descriptive supplementary panels A-D (before tendency/network)."""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
sys.path.insert(0, "."); from desc_tables import cls  # noqa: E402  (re-uses the RefSeq classifier)
plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
C = pd.read_csv("per_tf_counts.tsv", sep="\t").set_index("tf")
R = pd.read_csv("../goodex/tf_total_reads.tsv", sep="\t", header=None, index_col=0)[1]
W = pd.read_excel("/Users/dyy/Desktop/HEK293T-ATGTAG/Cicero_TF_loop_overlap_WT_0.7.xlsx", sheet_name=0).set_index("TF")
C["reads"] = R.reindex(C.index); C["pairs"] = W["Pairs >25 kb"].reindex(C.index)
C = C[C.cliques > 0].sort_values("reads")
CLASSES = ["promoter", "exon", "intron", "intergenic"]; CCOL = ["#B03A2E", "#E59866", "#85A9C9", "#C8C8C8"]

# ---- A: per-TF data overview -----------------------------------------------------------
cols = [("reads", "DNA reads (all cells)"), ("cicero_peaks", "Cicero input peaks"), ("pairs", "co-binding pairs >25 kb"), ("cliques", "cliques (≥3 regions)")]
fig, axs = plt.subplots(1, 4, figsize=(15, 0.2 * len(C) + 1.6), sharey=True, gridspec_kw={"wspace": 0.12})
y = np.arange(len(C))
for ax, (k, t) in zip(axs, cols):
    v = C[k].fillna(0)
    ax.barh(y, v, color="#1F4E8C" if k != "reads" else "#555", height=0.75)
    ax.set_xscale("log"); ax.set_title(t, loc="left", fontsize=11.5)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_ylim(-0.7, len(C) - 0.3)
axs[0].set_yticks(y); axs[0].set_yticklabels([f"{t}{' *' if t == 'CTCF' else ''}" for t in C.index], fontsize=8.5)
r = np.corrcoef(np.log10(C.reads), np.log10(C.cliques))[0, 1]
fig.suptitle(f"Per-TF data behind the co-binding cliques (TFs ordered by sequencing depth; log scales) · log reads vs log cliques r = {r:.2f}",
             x=0.01, ha="left", y=0.955, fontsize=13)
fig.savefig("figA_per_tf_overview.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---- B: clique size and span -------------------------------------------------------------
Z = pd.read_csv("clique_sizes.tsv.gz", sep="\t")
fig, axs = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"wspace": 0.3, "width_ratios": [1, 1, 1.4]})
vc = Z.n_regions.clip(upper=12).value_counts().sort_index()
axs[0].bar(vc.index, vc.values, color="#1F4E8C"); axs[0].set_yscale("log")
axs[0].set_xticks(range(3, 13)); axs[0].set_xticklabels([str(i) for i in range(3, 12)] + ["≥12"])
axs[0].set_xlabel("regions per clique"); axs[0].set_ylabel("cliques (log)")
axs[0].set_title(f"Clique size (n = {len(Z):,}; median {Z.n_regions.median():.0f})", loc="left", fontsize=11.5)
axs[1].hist(np.log10(Z.span_kb.clip(lower=1)), bins=40, color="#1F4E8C")
axs[1].set_xticks([0, 1, 2, 3]); axs[1].set_xticklabels(["1", "10", "100", "1,000"])
axs[1].set_xlabel("clique span (kb)"); axs[1].set_ylabel("cliques")
axs[1].set_title(f"Clique span (median {Z.span_kb.median():.0f} kb)", loc="left", fontsize=11.5)
big = Z.groupby("tf").size(); big = big[big >= 20].index
g = Z[Z.tf.isin(big)].groupby("tf").agg(ge4=("n_regions", lambda v: 100 * (v >= 4).mean()), span=("span_kb", "median"), n=("n_regions", "size"))
axs[2].scatter(g["span"], g["ge4"], s=8 + 30 * np.log10(g.n), color="#1F4E8C", alpha=0.8)
for t, rr in g.iterrows():
    if rr.n > 1000 or rr.ge4 > g.ge4.quantile(0.9) or rr.ge4 < g.ge4.quantile(0.08):
        axs[2].annotate(t, (rr["span"], rr.ge4), fontsize=8.5, xytext=(3, 2), textcoords="offset points")
axs[2].set_xlabel("median clique span (kb)"); axs[2].set_ylabel("% of cliques with ≥4 regions")
axs[2].set_title("Per TF (≥20 cliques; dot size = no. of cliques)", loc="left", fontsize=11.5)
for ax in axs:
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.savefig("figB_clique_size_span.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---- C: genomic annotation ---------------------------------------------------------------
A = pd.read_csv("annotation_counts.tsv", sep="\t")
H = pd.read_csv("../basic/hub_regions.tsv", sep="\t", usecols=["hub_region", "n_tfs"])
H["cls"] = [cls(r.split(":")[0], *map(int, r.split(":")[1].split("-"))) for r in H.hub_region]
H["bin"] = pd.cut(H.n_tfs, [0, 1, 2, 4, 9, 100], labels=["1", "2", "3–4", "5–9", "≥10"])
bars = {"all Cicero peaks": A.groupby("class").peaks.sum(), "all clique regions": A.groupby("class").clique_regions.sum()}
for b, d in H.groupby("bin", observed=True):
    bars[f"in {b} TF{'s' if b != '1' else ''}{chr(39)}{'' if b != '1' else 's'} cliques"] = d.cls.value_counts()
B = pd.DataFrame(bars).reindex(CLASSES).fillna(0); B = B / B.sum()
ns = {"all Cicero peaks": A.peaks.sum(), "all clique regions": A.clique_regions.sum(), **{f"in {b} TF{'s' if b != '1' else ''}{chr(39)}{'' if b != '1' else 's'} cliques": len(d) for b, d in H.groupby("bin", observed=True)}}
fig, axs = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"wspace": 0.35, "width_ratios": [1.25, 1]})
left = np.zeros(len(B.columns))
for k, c in zip(CLASSES, CCOL):
    axs[0].barh(range(len(B.columns)), B.loc[k], left=left, color=c, label=k, height=0.7)
    for i, (l, v) in enumerate(zip(left, B.loc[k])):
        if v > 0.06: axs[0].text(l + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=9, color="white" if k == "promoter" else "#222")
    left += B.loc[k].values
axs[0].set_yticks(range(len(B.columns))); axs[0].set_yticklabels([f"{c}\n(n = {ns[c]:,})" for c in B.columns], fontsize=9.5)
axs[0].invert_yaxis(); axs[0].set_xlim(0, 1); axs[0].set_xlabel("fraction of regions")
axs[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, fontsize=10)
axs[0].set_title("Genomic class of regions", loc="left", fontsize=12)
pt = A.pivot(index="tf", columns="class", values=["peaks", "clique_regions"])
fp = (pt["peaks"]["promoter"] / pt["peaks"].sum(1)); fc = (pt["clique_regions"]["promoter"] / pt["clique_regions"].sum(1))
ok = pt["clique_regions"].sum(1) >= 50
axs[1].scatter(fp[ok] * 100, fc[ok] * 100, s=22, color="#B03A2E")
m = max(fp[ok].max(), fc[ok].max()) * 100 + 3
axs[1].plot([0, m], [0, m], color="#999", lw=0.8, ls="--")
OFF = {"RBBP7": (4, 0), "XPA": (4, 0), "DDIT3": (-6, 4, "right"), "E2F3": (-6, 2, "right"), "NFYA": (4, -6), "NME2": (4, -2),
       "RBBP4": (4, -2), "SOX4": (-6, 2, "right"), "ZNF507": (-6, -4, "right"), "MBD3": (4, -6), "ZNF282": (5, 2), "ETS2": (5, -6)}
for t, o in OFF.items():
    if t in fp.index and ok.get(t, False):
        axs[1].annotate(t, (fp[t] * 100, fc[t] * 100), fontsize=9, xytext=o[:2], textcoords="offset points", ha=o[2] if len(o) > 2 else "left", va="center")
if "CTCF" in fp.index and not ok["CTCF"]:
    axs[1].scatter([fp["CTCF"] * 100], [fc["CTCF"] * 100], s=26, facecolor="white", edgecolor="#B03A2E", zorder=3)
    axs[1].annotate("CTCF (39 regions)", (fp["CTCF"] * 100, fc["CTCF"] * 100), fontsize=9, xytext=(6, 6), textcoords="offset points", va="center")
axs[1].set_xlabel("% of the TF's Cicero peaks at promoters"); axs[1].set_ylabel("% of the TF's clique regions at promoters")
axs[1].text(0.97, 0.05, "dashed: y = x", transform=axs[1].transAxes, ha="right", fontsize=9, color="#777")
axs[1].set_title("Promoter share per TF", loc="left", fontsize=12)
for ax in axs:
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.savefig("figC_genomic_annotation.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---- D: Hi-C support of pairs per TF -----------------------------------------------------
Wt = W[W["Pairs >25 kb"] >= 500].sort_values("% pairs matching")
fig, axs = plt.subplots(1, 2, figsize=(13, 0.22 * len(Wt) + 1.8), sharey=True, gridspec_kw={"wspace": 0.1})
yy = np.arange(len(Wt))
axs[0].barh(yy, Wt["% pairs matching"] * 100, color="#2E7D6B")
axs[0].axvline(14.5, color="#C0392B", ls="--", lw=1); axs[0].text(14.8, -0.4, "shuffle null (RBBP4, 14.5%)", color="#C0392B", fontsize=8.5, va="bottom")
axs[0].set_yticks(yy); axs[0].set_yticklabels([f"{t} ({n:,})" for t, n in zip(Wt.index, Wt["Pairs >25 kb"])], fontsize=9)
axs[0].set_xlabel("% of pairs on a 0.7 loop (±30 kb)"); axs[0].set_title("Co-binding pairs >25 kb", loc="left", fontsize=11.5)
axs[1].barh(yy, Wt["% edges on a loop"] * 100, color="#1F4E8C")
axs[1].axvline(15.5, color="#C0392B", ls="--", lw=1)
axs[1].set_xlabel("% of clique edges on a 0.7 loop"); axs[1].set_title("Clique edges", loc="left", fontsize=11.5)
for ax in axs:
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.suptitle("Hi-C support: co-binding pairs sit on wild-type HEK293T loops above chance (TFs with ≥500 pairs; pairs in brackets)", x=0.01, ha="left", y=0.945, fontsize=13)
fig.savefig("figD_hic_support_per_tf.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("TFs A:", len(C), " D:", len(Wt)); print(B.round(3).to_string())
