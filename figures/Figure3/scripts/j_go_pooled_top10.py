"""Fig 3 panel j (user 2026-11-18: pooled GO with the new split, top 10 terms per group): TF-specific (module genes of exactly 1 TF) and conserved (modules of >=3 TFs) gene
sets, each with its OWN top-10 GO BP terms (no shared term list). Values = pooled lists from scripts/go_per_tf.py (checks/go_per_tf):
x = -log10 nominal p (same axis both sides), dashed line = p of the last term with BH q < 0.05 (0.05/m when none), teal = q < 0.05,
hollow grey = q >= 0.05, size = genes in the term; per-TF counts only in go_per_tf.xlsx. Top 10 terms per group ranked by pooled p; near-duplicates dropped (background gene-set Jaccard > 0.6 with a
better-ranked term; checks/go_per_tf/top_term_jaccard.tsv from scripts/go_term_jaccard.py).
  python Figure3/scripts/j_go_pooled_top10.py   (Mac)"""
import re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
H = Path(__file__).resolve().parents[1]; G = H / "checks/go_per_tf"; MM = 1 / 25.4
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 6, "axes.labelsize": 6.5, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "legend.frameon": False})
cm = LinearSegmentedColormap.from_list("t", ["#C9E3DC", "#2C7A6B", "#0E3B33"])
def core(n):
    n = n.lower().replace("torc1", "tor")
    for w in ("positive regulation of ", "negative regulation of ", "regulation of "): n = n.replace(w, "")
    return n.replace(" process", "").strip()
JJ = pd.read_csv(G / "top_term_jaccard.tsv", sep="\t"); JAC = {frozenset((a, b)): j for a, b, j in zip(JJ.term_a, JJ.term_b, JJ.jaccard)}   # scripts/go_term_jaccard.py
def top(grp, k=10):
    pq = pd.read_csv(G / f"pooled_{grp}.tsv", sep="\t", index_col=0).sort_values("p"); sm = pd.read_csv(G / f"summary_{grp}.tsv", sep="\t", index_col=0)
    keep = []
    for t, r in pq.head(40).iterrows():                                         # near-duplicates: background gene-set Jaccard > 0.6 with a better-ranked term
        if r.k == 0: continue
        if any(JAC.get(frozenset((t, u)), 0) > 0.6 for u in keep): continue
        keep.append(t)
        if len(keep) == k: break
    sigp = pq.p[pq.q < 0.05]; thr = float(sigp.max()) if len(sigp) else 0.05 / len(pq)   # BH: p of the last term with q < 0.05; none -> rank-1 bound 0.05/m
    T = pq.loc[keep].copy(); T["thr"] = thr; T["n_tfs_sig"] = sm.n_tfs_sig.reindex(keep).fillna(0).astype(int).values; T["n_tfs_tested"] = int(sm.n_tfs_tested.iloc[0]); return T
SHORT = {"dolichol-linked oligosaccharide biosynthetic process": "dolichol oligosaccharide biosynthesis", "regulation of monoatomic ion transmembrane transport": "reg. of ion transmembrane transport",
         "monoatomic ion transmembrane transport": "ion transmembrane transport", "monoatomic cation transmembrane transport": "cation transmembrane transport",
         "regulation of mononuclear cell migration": "reg. of mononuclear cell migration", "positive regulation of cell cycle phase transition": "pos. reg. of cell cycle phase transition",
         "positive regulation of triglyceride biosynthetic process": "pos. reg. of triglyceride biosynthesis", "positive regulation of TORC1 signaling": "pos. reg. of TORC1 signaling",
         "positive regulation of catabolic process": "pos. reg. of catabolic process", "regulation of cellular response to stress": "reg. of response to stress", "endoplasmic reticulum to Golgi vesicle-mediated transport": "ER to Golgi vesicle transport"}
cap = lambda s: (lambda t: t[0].upper() + t[1:])(SHORT.get(s, s))
SP, CO = top("specific"), top("conserved")
TEAL, GREY = "#2C7A6B", "#9E9E9E"; sz = lambda n: 4 + 0.35 * n
xmax = np.ceil(max(-np.log10(SP.p.min()), -np.log10(CO.p.min()), -np.log10(min(SP.thr.iloc[0], CO.thr.iloc[0]))) + 0.5)
fig = plt.figure(figsize=(183 * MM, 66 * MM)); W = 0.22
axs = [fig.add_axes([0.245, 0.24, W, 0.62]), fig.add_axes([0.73, 0.24, W, 0.62])]
for ax, T, ttl in ((axs[0], SP, "TF-specific module genes"), (axs[1], CO, "Conserved module genes")):
    y = np.arange(len(T)); x = -np.log10(T.p.values); sig = (T.q < 0.05).values
    ax.scatter(x[~sig], y[~sig], s=sz(T.k.values[~sig]), facecolors="white", edgecolors=GREY, linewidths=0.6, zorder=3)
    ax.scatter(x[sig], y[sig], s=sz(T.k.values[sig]), color=TEAL, lw=0, zorder=3)
    t = -np.log10(T.thr.iloc[0]); ax.axvline(t, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.text(t + 0.08, len(T) - 0.55, "q = 0.05", ha="left", va="bottom", fontsize=5, color="#4D4D4D")
    ax.set_yticks(y); ax.set_yticklabels([cap(s_) for s_ in T.name], fontsize=6); ax.set_ylim(len(T) - 0.5, -0.6); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, xmax); ax.set_xlabel("−log10 p"); ax.xaxis.grid(True, color="#EEEEEE", lw=0.5); ax.set_axisbelow(True)
    n_ = int(T.n_list.iloc[0]); ax.set_title(f"{ttl}\n({'1 TF' if 'specific' in ttl else '≥3 TFs'}, n = {n_:,})", fontsize=6.5, pad=9)
hs = [plt.scatter([], [], s=sz(n), color="#4D4D4D", lw=0) for n in (20, 50, 100)] + [plt.scatter([], [], s=sz(40), color=TEAL, lw=0), plt.scatter([], [], s=sz(40), facecolors="white", edgecolors=GREY, linewidths=0.6)]
fig.legend(hs, ["20", "50", "100 genes", "q < 0.05", "q ≥ 0.05"], loc="lower center", ncol=5, fontsize=5.5, bbox_to_anchor=(0.6, 0.0), handletextpad=0.2, columnspacing=1.2)
PS = {("normal", "normal"): "ArialMT"}
def illustrator_text(svg):
    def one(m):
        tag = m.group(0); sm = re.search(r'style="([^"]*)"', tag)
        if not sm: return tag
        keep, size = [], None
        for d in [d.strip() for d in sm.group(1).split(";") if d.strip()]:
            k, _, v = d.partition(":"); k, v = k.strip(), v.strip()
            if k == "font":
                mm_ = re.match(r"(?:\S+\s+)?([\d.]+)px", v); size = mm_.group(1) if mm_ else None
            else: keep.append(f"{k}: {v}")
        if size is None: return tag
        tag = tag.replace(sm.group(0), f'style="{"; ".join(keep)}"' if keep else "")
        return tag[:-1].rstrip() + f' font-family="Arial, ArialMT, Helvetica, sans-serif" font-size="{float(size):.3f}">'
    return re.sub(r"<(?:text|tspan)\b[^>]*>", one, svg)
stem = H / "j_gene_set_dotplot"; fig.savefig(f"{stem}.svg"); fig.savefig(f"{stem}.png", dpi=300)
p = Path(f"{stem}.svg"); p.write_text(illustrator_text(p.read_text())); print("font shorthand left:", p.read_text().count("font: "))
pd.concat([SP.assign(group="TF-specific (1 TF)"), CO.assign(group="conserved (>=3 TFs)")])[["group", "name", "k", "n_list", "K", "fold", "p", "q", "n_tfs_sig", "n_tfs_tested"]].rename(columns={"k": "genes", "n_list": "list_annotated", "K": "term_size", "n_tfs_sig": "TFs_with_q005_alone", "n_tfs_tested": "TFs_testable"}).to_csv(H / "j_gene_set_dotplot.tsv", sep="\t")
print("thresholds p:", SP.thr.iloc[0], CO.thr.iloc[0]); print(SP[["name", "k", "p", "q", "n_tfs_sig"]].round(4).to_string()); print(CO[["name", "k", "p", "q", "n_tfs_sig"]].round(4).to_string())
