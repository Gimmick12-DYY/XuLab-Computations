"""Fig 3 panel j, side by side: GO BP of genes that higher-order coordinated binding modules sit on, shared by many TFs vs TF-specific.
Same inputs, gene assignment, background and test as the current j panel (cobinding/tf_groups/scripts/go_cliques.py, commit 00548af):
- nodes: cobinding/tf_groups/results_location/nodes_annotated.tsv.gz (clique_location.py): each TF's network peaks, flag = in one of
  that TF's modules; gene = protein-coding GENCODE v46 gene whose TSS lies within the peak +-1 kb (promoter overlap; Ensembl-only ids dropped).
- per gene: n_clique = number of TFs whose modules contain the gene's promoter.
- shared (left): n_clique >= 3 (the current j list). TF-specific (right): n_clique == 1 (promoter in a module of exactly one TF).
- background (both): every protein-coding gene with a promoter node in any TF's network (go_cliques.py 'universe').
- GO: go-basic.obo + goa_human.gaf (data/go, downloaded 2026-11-06), BP annotations propagated to ancestors (is_a, part_of), terms of
  10-500 background genes, one-sided hypergeometric, BH within each list.
- A size-matched check draws 1,035 genes from the specific list 200 times (seed 1) and reports how often each plotted term reaches q < 0.05.
Plot: one shared y axis = union of the top 8 terms of each side (by q, ties by raw p); x = fold enrichment; dot size = genes in term; filled = q < 0.05,
hollow grey = n.s.; q printed at the right of each half. 140 x 70 mm, Arial 6-7 pt; PNG + SVG.
  python scripts/j_shared_vs_specific.py <outdir>   (hic env on Longleaf)"""
import glob, gzip, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import hypergeom
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in glob.glob("/work/users/d/y/dyy12/fonts/arial/*.ttf"): fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 6, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "legend.frameon": False})
X = Path("/work/users/d/y/dyy12/XuLab"); TG = X / "cobinding/tf_groups"; D = X / "data/go"; OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
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
    bg = [g for g in set(bg) if g in g2t]; L = set(lst) & set(bg); N, n = len(bg), len(L); t2g = defaultdict(set)
    for g in bg:
        for t in g2t[g]: t2g[t].add(g)
    rows = []
    for t, s in t2g.items():
        K = len(s)
        if not 10 <= K <= 500: continue
        k = len(s & L); rows.append((t, name.get(t, ""), k, n, K, N, (k / n) / (K / N) if k else 0.0, hypergeom.sf(k - 1, N, K, n)))
    R = pd.DataFrame(rows, columns=["term", "name", "k", "n_list", "K", "N", "fold", "p"]); R["q"] = bh(R.p.values); return R.set_index("term")
Nd = pd.read_csv(TG / "results_location/nodes_annotated.tsv.gz", sep="\t", keep_default_na=False)
P = Nd[Nd.gene != ""].assign(gene=lambda d: d.gene.str.split(",")).explode("gene")
P = P[~P.gene.str.startswith("ENSG")].groupby(["tf", "gene"]).clique.max().reset_index()
universe = P.gene.unique(); cnt = P.groupby("gene").agg(n_node=("tf", "size"), n_clique=("clique", "sum"))
shared = cnt.index[cnt.n_clique >= 3]; specific = cnt.index[cnt.n_clique == 1]
A = enrich(shared, universe); B = enrich(specific, universe)
print(f"universe {len(universe):,} genes (with GO BP: {A.N.iloc[0]:,}); shared (>=3 TFs) {len(shared):,} (GO-annotated {A.n_list.iloc[0]:,}); specific (1 TF) {len(specific):,} (GO-annotated {B.n_list.iloc[0]:,})")
print("q<0.05 terms: shared", int((A.q < 0.05).sum()), "| specific", int((B.q < 0.05).sum()))
top = list(dict.fromkeys(list(A.sort_values(["q", "p"]).index[:8]) + list(B.sort_values(["q", "p"]).index[:8])))   # ties at q = 1 broken by raw p
rng = np.random.default_rng(1); spec_l = list(specific); hits = defaultdict(int)
for _ in range(200):
    Rm = enrich(rng.choice(spec_l, len(shared), replace=False), universe)
    for t in top: hits[t] += int(t in Rm.index and Rm.q[t] < 0.05)
T = pd.DataFrame({"name": [A.name.get(t, B.name.get(t, "")) for t in top], "shared_fold": [A.fold.get(t, np.nan) for t in top], "shared_q": [A.q.get(t, np.nan) for t in top],
                  "shared_genes": [A.k.get(t, 0) for t in top], "specific_fold": [B.fold.get(t, np.nan) for t in top], "specific_q": [B.q.get(t, np.nan) for t in top],
                  "specific_genes": [B.k.get(t, 0) for t in top], "specific_sizematched_frac_q005": [hits[t] / 200 for t in top]}, index=top)
T = T.sort_values(["shared_q", "specific_q"], ascending=[False, False]); T.round(5).to_csv(OUT / "j_gene_set_dotplot_shared_vs_specific.tsv", sep="\t")
print(T.round(3).to_string())
SHORT = {"endoplasmic reticulum to Golgi vesicle-mediated transport": "ER to Golgi vesicle transport",
         "regulation of monoatomic ion transmembrane transport": "Regulation of ion transmembrane transport",
         "monoatomic ion transmembrane transport": "Ion transmembrane transport", "monoatomic cation transport": "Cation transport",
         "dolichol-linked oligosaccharide biosynthetic process": "Dolichol oligosaccharide synthesis",
         "regulation of cellular response to stress": "Regulation of response to stress"}
cap = lambda s: SHORT.get(s, (s[0].upper() + s[1:]) if s else s)
MM = 1 / 25.4; fig = plt.figure(figsize=(140 * MM, (22 + 4.6 * len(T)) * MM)); gs = fig.add_gridspec(1, 2, wspace=0.42, left=0.30, right=0.93, top=0.9, bottom=0.2 if len(T) < 14 else 0.14)
INK, GREY, TEAL = "#1A1A1A", "#9E9E9E", "#1B5E52"; y = np.arange(len(T)); sz = lambda n: 3 + 0.35 * np.asarray(n, float)
xmax = np.nanmax([T.shared_fold.max(), T.specific_fold.max()]) * 1.08
for k, (side, lab, nn) in enumerate((("shared", "Shared by ≥3 TFs", len(shared)), ("specific", "TF-specific (1 TF)", len(specific)))):
    ax = fig.add_subplot(gs[k]); f, q, g = T[f"{side}_fold"].values, T[f"{side}_q"].values, T[f"{side}_genes"].values
    sig = q < 0.05
    ax.axvline(1, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.scatter(f[sig], y[sig], s=sz(g[sig]), color=TEAL, lw=0, zorder=3); ax.scatter(f[~sig], y[~sig], s=sz(g[~sig]), facecolors="white", edgecolors=GREY, linewidths=0.6, zorder=3)
    ax.set_ylim(-0.7, len(T) - 0.3); ax.set_xlim(0, xmax); ax.set_xlabel("Fold enrichment"); ax.set_title(f"{lab}\n(n = {nn:,} genes)", fontsize=6.5, loc="center")
    ax.set_yticks(y); ax.set_yticklabels([cap(s) for s in T.name] if k == 0 else []); ax.tick_params(axis="y", length=0)
    for i, qq in enumerate(q): ax.text(xmax * 1.02, i, ("q < 0.001" if qq < 0.001 else f"q = {qq:.2g}") if np.isfinite(qq) else "n/a", va="center", fontsize=5, color=INK if qq < 0.05 else GREY, clip_on=False)
h = [plt.Line2D([], [], ls="", marker="o", ms=3.5, color=TEAL), plt.Line2D([], [], ls="", marker="o", ms=3.5, mfc="white", mec=GREY, mew=0.6)] + [plt.scatter([], [], s=sz(v), color=INK, lw=0) for v in (20, 100, 300)]
fig.legend(h, ["q < 0.05", "n.s.", "20", "100", "300 genes"], loc="lower center", ncol=5, fontsize=6, handletextpad=0.3, columnspacing=1.0)
PS = {("normal", "normal"): "ArialMT", ("bold", "normal"): "Arial-BoldMT"}
def illustrator_text(svg):
    def one(m):
        tag = m.group(0); sm = re.search(r'style="([^"]*)"', tag)
        if not sm: return tag
        keep, size, wt = [], None, "normal"
        for d in [d.strip() for d in sm.group(1).split(";") if d.strip()]:
            kk, _, v = d.partition(":"); kk, v = kk.strip(), v.strip()
            if kk == "font":
                mm_ = re.match(r"(?:(?:italic|oblique)\s+)?(?:(bold|[1-9]00)\s+)?([\d.]+)px", v)
                if mm_: wt = "bold" if mm_.group(1) in ("bold", "700") else wt; size = mm_.group(2)
            else: keep.append(f"{kk}: {v}")
        if size is None: return tag
        tag = tag.replace(sm.group(0), f'style="{"; ".join(keep)}"' if keep else "")
        return tag[:-1].rstrip() + f' font-family="Arial, {PS[(wt, "normal")]}, Helvetica, sans-serif" font-size="{float(size):.3f}"' + (' font-weight="bold"' if wt == "bold" else "") + ">"
    return re.sub(r"<text\b[^>]*>", one, svg)
stem = OUT / "j_gene_set_dotplot_shared_vs_specific"; fig.savefig(f"{stem}.svg"); fig.savefig(f"{stem}.png", dpi=300)
p = Path(f"{stem}.svg"); p.write_text(illustrator_text(p.read_text())); print("font shorthand left:", p.read_text().count("font: "))
