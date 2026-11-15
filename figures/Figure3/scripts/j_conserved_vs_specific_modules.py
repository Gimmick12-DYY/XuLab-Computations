"""Fig 3 panel j: the original module-gene GO panel (fig3_v7/fig3_geneset_dotplot.py, Figure3/j_module_genes_go, values read unchanged from
its output fig3_v7/panels_extra/fig3_geneset_dotplot.tsv) with, beside it on the same y axis and the same terms in the same order, the same
test for TF-specific module genes.
Left, "Conserved module pairs": protein-coding genes whose promoter (TSS +-1 kb) is in a higher-order module of >=3 TFs (1,035 genes;
  pooled_ge3TF of cobinding/tf_groups/scripts/go_cliques.py). Filled dot = hypergeometric fold, BH q (q printed); hollow dot = matched for
  local peak density, as the original.
Right, "TF-specific module pairs": genes whose promoter is in a module of exactly 1 TF (n_clique == 1, same nodes_annotated table).
  Same GO files (goa_human.gaf + go-basic.obo, BP, ancestors, NOT dropped), same background (network promoter genes, 12,388; 10,934
  annotated), terms 10-500 genes, one-sided hypergeometric, BH over all terms of the list (go_cliques.py enrich(), statsmodels-free BH).
  The density-matched test has no 1-TF counterpart (it compares module counts among genes bound by >=3 TFs), so only filled dots here.
Same x range and dot-size scale on both sides. 150 x 55 mm, Arial 6-7 pt, Palette C neutrals. PNG + SVG + TSV.
  python j_conserved_vs_specific_modules.py <outdir>   (Longleaf, hic env)"""
import glob, gzip, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.stats import hypergeom
for f in glob.glob("/work/users/d/y/dyy12/fonts/arial/*.ttf"): fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 6, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 0, "legend.frameon": False})
X = Path("/work/users/d/y/dyy12/XuLab"); TG = X / "cobinding/tf_groups"; D = X / "data/go"; OUT = Path(sys.argv[1])
L = pd.read_csv(X / "figures/fig3_v7/panels_extra/fig3_geneset_dotplot.tsv", sep="\t")          # original panel, row order kept
# ---- go_cliques.py enrich(), unchanged except BH ----
par, name, cur = defaultdict(set), {}, None
for ln in open(D / "go-basic.obo"):
    ln = ln.strip()
    if ln == "[Term]": cur = None; continue
    if ln.startswith("id: GO:"): cur = ln[4:]
    elif cur and ln.startswith("name: "): name[cur] = ln[6:]
    elif cur and ln.startswith("is_a: "): par[cur].add(ln[6:16])
    elif cur and ln.startswith("relationship: part_of "): par[cur].add(ln[22:32])
anc = {}; sys.setrecursionlimit(10000)
def up(t):
    if t not in anc:
        s = {t}
        for p in par.get(t, ()): s |= up(p)
        anc[t] = s
    return anc[t]
g2t = defaultdict(set)
for ln in gzip.open(D / "goa_human.gaf.gz", "rt"):
    if ln[0] == "!": continue
    x = ln.split("\t")
    if "NOT" in x[3] or x[8] != "P": continue
    g2t[x[2]] |= up(x[4])
def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); r = p[o] * len(p) / np.arange(1, len(p) + 1); q = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out
def enrich(lst, bg):
    bg = [g for g in set(bg) if g in g2t]; Ls = set(lst) & set(bg); N, n = len(bg), len(Ls); t2g = defaultdict(set)
    for g in bg:
        for t in g2t[g]: t2g[t].add(g)
    rows = []
    for t, s in t2g.items():
        K = len(s)
        if not 10 <= K <= 500: continue
        k = len(s & Ls); rows.append({"term": t, "name": name.get(t, ""), "k": k, "n_list": n, "K": K, "N": N, "fold": (k / n) / (K / N) if k else 0, "p": hypergeom.sf(k - 1, N, K, n)})
    R = pd.DataFrame(rows); R["q"] = bh(R.p); return R.set_index("term")
Nd = pd.read_csv(TG / "results_location/nodes_annotated.tsv.gz", sep="\t", keep_default_na=False)
P = Nd[Nd.gene != ""].assign(gene=lambda d: d.gene.str.split(",")).explode("gene")
P = P[~P.gene.str.startswith("ENSG")].groupby(["tf", "gene"]).clique.max().reset_index(); universe = P.gene.unique()
cnt = P.groupby("gene").agg(n_node=("tf", "size"), n_clique=("clique", "sum"))
R3 = enrich(cnt.index[cnt.n_clique >= 3], universe); R1 = enrich(cnt.index[cnt.n_clique == 1], universe)
chk = R3.loc[L.term]; assert np.allclose(chk.fold.values, L.fold.values, atol=1e-4) and (chk.k.values == L.genes.values).all(), "left panel not reproduced"
n3, n1 = int(R3.n_list.iloc[0]), int(R1.n_list.iloc[0])
print(f"genes: >=3 TFs {int((cnt.n_clique >= 3).sum()):,} ({n3:,} annotated); exactly 1 TF {int((cnt.n_clique == 1).sum()):,} ({n1:,} annotated); background {len(universe):,}")
print("q<0.05 terms in whole list: conserved", int((R3.q < 0.05).sum()), "| TF-specific", int((R1.q < 0.05).sum()))
print("TF-specific top:", R1.sort_values("p").head(5)[["name", "k", "fold", "q"]].round(4).to_string())
T = L.copy(); T["specific_genes"] = R1.k.reindex(L.term).values; T["specific_fold"] = R1.fold.reindex(L.term).values; T["specific_q"] = R1.q.reindex(L.term).values
T["conserved_n_annotated"] = n3; T["specific_n_annotated"] = n1
T.round(5).to_csv(OUT / "j_gene_set_dotplot.tsv", sep="\t", index=False); print(T[["name", "genes", "fold", "q", "specific_genes", "specific_fold", "specific_q"]].round(3).to_string(index=False))
# ---- plot: original panel code on the left, same code (filled dots) on the right ----
MM = 1 / 25.4; INK, GREY = "#1A1A1A", "#9E9E9E"; sz = lambda n: 4 + 0.35 * n
fig = plt.figure(figsize=(150 * MM, 58 * MM)); a = fig.add_axes([0.27, 0.30, 0.27, 0.55]); b = fig.add_axes([0.66, 0.30, 0.27, 0.55], sharey=a)
y = np.arange(len(T)); xmax = max(3.4, T.fold.max() + 0.3, T.specific_fold.max() + 0.3)
a.hlines(y, T.matched_oe, T.fold, color=GREY, lw=0.6, zorder=1)
a.scatter(T.matched_oe, y, s=14, facecolors="white", edgecolors=GREY, linewidths=0.7, zorder=2)
a.scatter(T.fold, y, s=sz(T.genes), color=INK, lw=0, zorder=3)
b.scatter(T.specific_fold, y, s=sz(T.specific_genes), color=INK, lw=0, zorder=3)
for ax, qs, ttl, n in ((a, T.q, "Conserved module pairs", n3), (b, T.specific_q, "TF-specific module pairs", n1)):
    ax.axvline(1, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=0); ax.set_xlim(0.6, xmax); ax.set_xlabel("Fold enrichment")
    ax.set_title(f"{ttl}\n(promoter genes, n = {n:,})", fontsize=7, pad=4)
    for i, q in enumerate(qs): ax.text(xmax + 0.05, i, f"q = {q:.2g}" if q >= 0.001 else "q < 0.001", va="center", ha="left", fontsize=5.5, color=INK, clip_on=False)
a.set_yticks(y); a.set_yticklabels(T.name); a.set_ylim(-0.7, len(T) - 0.3)
b.tick_params(axis="y", left=False, labelleft=False); b.spines["left"].set_visible(False)
h = [plt.Line2D([], [], ls="", marker="o", ms=4, color=INK), plt.Line2D([], [], ls="", marker="o", ms=3.6, mfc="white", mec=GREY, mew=0.7)]
l1 = a.legend(h, ["Unmatched", "Matched for local peak density"], loc="upper left", bbox_to_anchor=(-0.95, -0.25), ncol=2, fontsize=6, handletextpad=0.3, columnspacing=1.0, borderaxespad=0)
a.add_artist(l1)
hs = [plt.scatter([], [], s=sz(n), color=INK, lw=0) for n in (20, 50, 80)]
a.legend(hs, ["20", "50", "80 genes"], loc="upper left", bbox_to_anchor=(-0.95, -0.40), ncol=3, fontsize=6, handletextpad=0.1, columnspacing=0.6, borderaxespad=0)
PS = {("normal", "normal"): "ArialMT", ("bold", "normal"): "Arial-BoldMT", ("normal", "italic"): "Arial-ItalicMT", ("bold", "italic"): "Arial-BoldItalicMT"}
def illustrator_text(svg):   # same rewrite as build_fig3_v7.py: no CSS font shorthand, explicit Arial attributes
    def one(m):
        tag = m.group(0); sm = re.search(r'style="([^"]*)"', tag)
        if not sm: return tag
        keep, size, wt, it = [], None, "normal", "normal"
        for d in [d.strip() for d in sm.group(1).split(";") if d.strip()]:
            k, _, v = d.partition(":"); k, v = k.strip(), v.strip()
            if k == "font":
                mm_ = re.match(r"(?:(italic|oblique)\s+)?(?:(bold|[1-9]00)\s+)?([\d.]+)px", v)
                if mm_: it = "italic" if mm_.group(1) else it; wt = "bold" if mm_.group(2) in ("bold", "700", "800", "900") else wt; size = mm_.group(3)
            else: keep.append(f"{k}: {v}")
        if size is None: return tag
        tag = tag.replace(sm.group(0), f'style="{"; ".join(keep)}"' if keep else "")
        return tag[:-1].rstrip() + f' font-family="Arial, {PS[(wt, it)]}, Helvetica, sans-serif" font-size="{float(size):.3f}"' + (' font-weight="bold"' if wt == "bold" else "") + (' font-style="italic"' if it == "italic" else "") + ">"
    return re.sub(r"<text\b[^>]*>", one, svg)
stem = OUT / "j_gene_set_dotplot"; fig.savefig(f"{stem}.svg"); fig.savefig(f"{stem}.png", dpi=300)
p = Path(f"{stem}.svg"); p.write_text(illustrator_text(p.read_text())); print("font shorthand left:", p.read_text().count("font: "))
