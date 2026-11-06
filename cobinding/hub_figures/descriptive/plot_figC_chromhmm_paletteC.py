"""Fig 3c combined (palette C): genomic class bars + promoter / enhancer share per TF.
Promoter = RefSeq Select TSS ±1 kb; other regions by HEK293T 18-state ChromHMM."""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "palette"))
import regulome_palette as rp
P = rp.use()
HERE = Path(__file__).resolve().parent
OUT = ROOT / "figures_paletteC"; OUT.mkdir(exist_ok=True)
MM = 1 / 25.4
CLASSES = ["promoter", "enhancer", "other TSS/flank", "transcribed", "repressed/het", "quiescent"]
MERGE = {"active enhancer": "enhancer", "weak/poised enhancer": "enhancer"}
CH = P.chromhmm
COL = [CH.promoter, CH.active_enh, CH.tss_flank, CH.transcribed, CH.repressed, CH.quiescent]
A = pd.read_csv(HERE / "annotation_counts_chromhmm.tsv", sep="\t"); A["class"] = A["class"].replace(MERGE); A = A.groupby(["tf", "class"], as_index=False)[["peaks", "clique_regions"]].sum()
H = pd.read_csv(HERE / "hub_regions_chromhmm.tsv.gz", sep="\t"); H["cls"] = H.cls.replace(MERGE)
H["bin"] = pd.cut(H.n_tfs, [0, 1, 2, 4, 9, 100], labels=["1", "2", "3–4", "5–9", "≥10"])
lab = lambda b: f"in {b} TF{'s' if b != '1' else ''}{chr(39)}{'' if b != '1' else 's'} cliques"
bars = {"all Cicero peaks": A.groupby("class").peaks.sum(), "all clique regions": A.groupby("class").clique_regions.sum()}
ns = {"all Cicero peaks": A.peaks.sum(), "all clique regions": A.clique_regions.sum()}
for b, d in H.groupby("bin", observed=True):
    bars[lab(b)] = d.cls.value_counts(); ns[lab(b)] = len(d)
B = pd.DataFrame(bars).reindex(CLASSES).fillna(0); B = B / B.sum()
fig = plt.figure(figsize=(180 * MM, 55 * MM))
gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.38)
ax = fig.add_subplot(gs[0]); left = np.zeros(B.shape[1])
for k, c in zip(CLASSES, COL):
    ax.barh(range(B.shape[1]), B.loc[k], left=left, color=c, label=k, height=0.72)
    for i, (l, v) in enumerate(zip(left, B.loc[k])):
        if v >= 0.07: ax.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=5.5, color="white" if k in ("promoter", "enhancer", "repressed/het") else P.neutral.ink)
    left += B.loc[k].values
ax.set_yticks(range(B.shape[1])); ax.set_yticklabels([f"{c}\n(n = {ns[c]:,})" for c in B.columns], fontsize=5.5)
ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("fraction of regions")
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
ax.set_title("Genomic class of regions", loc="left")
pt = A.pivot(index="tf", columns="class", values=["peaks", "clique_regions"])
ok = pt["clique_regions"].sum(1) >= 50
def share(cols):
    fp = pt["peaks"][cols].sum(1) / pt["peaks"].sum(1); fc = pt["clique_regions"][cols].sum(1) / pt["clique_regions"].sum(1)
    return fp * 100, fc * 100
for k, (cols, ttl, labs) in enumerate([(["promoter"], "Promoter share per TF", ("RBBP7", "XPA", "DDIT3", "SOX4", "ETS2", "ZNF282")),
                                       (["enhancer"], "Enhancer share per TF (ChromHMM Enh*)", None)]):
    a = fig.add_subplot(gs[k + 1]); fp, fc = share(cols)
    a.scatter(fp[ok], fc[ok], s=7, color=CH.promoter if k == 0 else CH.active_enh, lw=0, zorder=2)
    if "CTCF" in fp.index and not ok["CTCF"]:
        a.scatter([fp["CTCF"]], [fc["CTCF"]], s=9, facecolor="white", edgecolor=P.neutral.dark, lw=0.6, zorder=3)
        a.annotate("CTCF", (fp["CTCF"], fc["CTCF"]), fontsize=5.5, xytext=(-4, 5) if k == 0 else (4, -6), textcoords="offset points", ha="right" if k == 0 else "left")
    m = max(fp[ok].max(), fc[ok].max()) * 1.08 + 1
    a.plot([0, m], [0, m], color=P.neutral.mid, ls="--", lw=0.5); a.set_xlim(0, m); a.set_ylim(0, m)
    if labs is None:
        d = (fc - fp)[ok]; labs = ["ETS2"] + list(d.sort_values().index[-4:])
    for t in labs:
        if t in fp.index and ok.get(t, False):
            a.annotate(t, (fp[t], fc[t]), fontsize=5.5, xytext=(3, 2), textcoords="offset points")
    a.set_xlabel("% of the TF's Cicero peaks"); a.set_ylabel("% of the TF's clique regions"); a.set_title(ttl, loc="left")
    a.text(0.97, 0.04, f"above y = x: {(fc[ok] > fp[ok]).sum()}/{ok.sum()} TFs", transform=a.transAxes, ha="right", fontsize=5.5, color=P.neutral.dark)
for ext in ("svg", "png"): fig.savefig(OUT / f"figC_genomic_annotation_chromhmm_paletteC.{ext}", bbox_inches="tight")
print(B.round(3).to_string())
