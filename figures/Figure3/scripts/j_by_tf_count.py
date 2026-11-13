"""Fig 3 panel j by TF count: GO BP of genes that higher-order coordinated binding modules sit on, binned by how many TFs' modules
contain the gene's promoter (1, 2, 3-4, >=5). Same gene assignment, background and test as before (all from j_shared_vs_specific.py):
Same inputs, gene assignment, background and test as the current j panel (cobinding/tf_groups/scripts/go_cliques.py, commit 00548af):
- nodes: cobinding/tf_groups/results_location/nodes_annotated.tsv.gz (clique_location.py): each TF's network peaks, flag = in one of
  that TF's modules; gene = protein-coding GENCODE v46 gene whose TSS lies within the peak +-1 kb (promoter overlap; Ensembl-only ids dropped).
- per gene: n_clique = number of TFs whose modules contain the gene's promoter.
- bins by n_clique: 1, 2, 3-4, >=5.
- background (both): every protein-coding gene with a promoter node in any TF's network (go_cliques.py 'universe').
- GO: go-basic.obo + goa_human.gaf (data/go, downloaded 2026-11-06), BP annotations propagated to ancestors (is_a, part_of), terms of
  10-500 background genes, one-sided hypergeometric, BH within each list.
- size-matched: every bin subsampled to the smallest bin's GO-annotated size, 200 draws (seed 1); median q and fraction of draws q < 0.05 per term.
Plot: one shared y axis = union of the top 6 terms of each bin (by q, ties by raw p); x = fold enrichment; dot size = genes in term;
filled = q < 0.05, hollow grey = n.s.; q at the right of each column. 183 mm wide, Arial 6-7 pt; PNG + SVG + TSV.
  python scripts/j_by_tf_count.py <outdir>   (hic env on Longleaf)"""
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
BINS = [("1 TF", cnt.n_clique == 1), ("2 TFs", cnt.n_clique == 2), ("3-4 TFs", cnt.n_clique.between(3, 4)), ("≥5 TFs", cnt.n_clique >= 5)]
R = {}; GL = {}
for lab, m in BINS:
    GL[lab] = list(cnt.index[m]); R[lab] = enrich(GL[lab], universe)
    print(f"{lab}: {len(GL[lab]):,} genes (GO-annotated {R[lab].n_list.iloc[0]:,}); terms q<0.05: {int((R[lab].q < 0.05).sum())}")
print(f"universe {len(universe):,} genes; with GO BP {R['1 TF'].N.iloc[0]:,}")
top = []
for lab, _ in BINS: top += list(R[lab].sort_values(["q", "p"]).index[:6])
top = list(dict.fromkeys(top))
annot = {lab: [g for g in GL[lab] if g in g2t] for lab in GL}; nmin = min(len(v) for v in annot.values()); rng = np.random.default_rng(1)
SM = {lab: {t: [] for t in top} for lab in GL}
for lab in GL:
    for _ in range(200):
        Rm = enrich(rng.choice(annot[lab], nmin, replace=False), universe)
        for t in top: SM[lab][t].append(Rm.q[t] if t in Rm.index else 1.0)
rows = []
for t in top:
    r = {"term": t, "name": next(R[l].name[t] for l in R if t in R[l].index)}
    for lab in GL:
        x = R[lab]; r[f"{lab}_fold"] = x.fold.get(t, np.nan); r[f"{lab}_q"] = x.q.get(t, np.nan); r[f"{lab}_genes"] = x.k.get(t, 0)
        r[f"{lab}_sizematched_median_q"] = float(np.median(SM[lab][t])); r[f"{lab}_sizematched_frac_q005"] = float(np.mean(np.array(SM[lab][t]) < 0.05))
    rows.append(r)
T = pd.DataFrame(rows).set_index("term"); T["_o"] = [min(T.loc[t, f"{l}_q"] for l in GL) for t in T.index]; T = T.sort_values("_o", ascending=False).drop(columns="_o")
T.round(5).to_csv(OUT / "j_gene_set_dotplot_by_tf_count.tsv", sep="\t"); print(f"size-matched to {nmin} GO-annotated genes per bin")
print(T[[c for c in T.columns if c.endswith("_q") or c == "name"]].round(3).to_string())
print(T[[c for c in T.columns if "sizematched_median_q" in c]].round(3).to_string())
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
SHORT = {"endoplasmic reticulum to Golgi vesicle-mediated transport": "ER to Golgi vesicle transport",
         "regulation of monoatomic ion transmembrane transport": "Regulation of ion transmembrane transport",
         "monoatomic ion transmembrane transport": "Ion transmembrane transport", "monoatomic cation transport": "Cation transport",
         "dolichol-linked oligosaccharide biosynthetic process": "Dolichol oligosaccharide synthesis",
         "regulation of cellular response to stress": "Regulation of response to stress",
         "positive regulation of triglyceride biosynthetic process": "Pos. reg. of triglyceride biosynthesis",
         "signal transduction in response to DNA damage": "DNA damage signal transduction",
         "negative regulation of protein kinase activity": "Neg. reg. of protein kinase activity"}
cap = lambda s: SHORT.get(s, (s[0].upper() + s[1:]) if s else s)
MM = 1 / 25.4; fig = plt.figure(figsize=(183 * MM, (24 + 4.4 * len(T)) * MM)); gs = fig.add_gridspec(1, 4, wspace=0.55, left=0.26, right=0.95, top=0.92, bottom=0.12)
INK, GREY, TEAL = "#1A1A1A", "#9E9E9E", "#1B5E52"; y = np.arange(len(T)); sz = lambda n: 3 + 0.35 * np.asarray(n, float)
XCAP = 6.0; xmax = XCAP * 1.05   # folds above 6 (few-gene terms) drawn at the edge as >
for k, (lab, _) in enumerate(BINS):
    ax = fig.add_subplot(gs[k]); f, q, g = T[f"{lab}_fold"].values, T[f"{lab}_q"].values, T[f"{lab}_genes"].values; sig = q < 0.05
    ax.axvline(1, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=0)
    over = f > XCAP; fc = np.minimum(f, XCAP)
    ax.scatter(fc[sig & ~over], y[sig & ~over], s=sz(g[sig & ~over]), color=TEAL, lw=0, zorder=3); ax.scatter(fc[~sig & ~over], y[~sig & ~over], s=sz(g[~sig & ~over]), facecolors="white", edgecolors=GREY, linewidths=0.6, zorder=3)
    ax.scatter(fc[over], y[over], marker=">", s=12, color=[TEAL if v else GREY for v in sig[over]], lw=0, zorder=3)
    ax.set_ylim(-0.7, len(T) - 0.3); ax.set_xlim(0, xmax); ax.set_xticks([0, 1, 2, 4, 6]); ax.set_xlabel("Fold enrichment"); ax.set_title(f"{lab}\n(n = {len(GL[lab]):,} genes)", fontsize=6.5)
    ax.set_yticks(y); ax.set_yticklabels([cap(s) for s in T.name] if k == 0 else []); ax.tick_params(axis="y", length=0)
    for i, qq in enumerate(q): ax.text(xmax * 1.02, i, ("<0.001" if qq < 0.001 else f"{qq:.2g}"), va="center", fontsize=5, color=INK if qq < 0.05 else GREY, clip_on=False)
    ax.text(xmax * 1.02, len(T) - 0.3, "q", fontsize=5.5, color=INK, va="bottom")
h = [plt.Line2D([], [], ls="", marker="o", ms=3.5, color=TEAL), plt.Line2D([], [], ls="", marker="o", ms=3.5, mfc="white", mec=GREY, mew=0.6)] + [plt.scatter([], [], s=sz(v), color=INK, lw=0) for v in (20, 100, 300)]
fig.legend(h, ["q < 0.05", "n.s.", "20", "100", "300 genes"], loc="lower center", ncol=5, fontsize=6, handletextpad=0.3, columnspacing=1.0)
stem = OUT / "j_gene_set_dotplot_by_tf_count"; fig.savefig(f"{stem}.svg"); fig.savefig(f"{stem}.png", dpi=300)
p = Path(f"{stem}.svg"); p.write_text(illustrator_text(p.read_text())); print("font shorthand left:", p.read_text().count("font: "))
