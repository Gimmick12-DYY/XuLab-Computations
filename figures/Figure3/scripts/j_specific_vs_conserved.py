"""Fig 3 panel j: GO BP of genes at TF-specific vs conserved coordinated binding pairs, for module pairs (left) and all pairs (right),
compareCluster-style dot plot with one shared y axis.
Pairs: first-step Cicero pairs (p <= 0.05, >25 kb, blacklist dropped) of 68 TFs from cobinding/tf_groups/scripts/build_incidence.py:
  all pairs = results/ (375,053 pairs); module pairs = results_cliques/ (172,311 pairs inside one TF's higher-order module).
  Matching across TFs: peaks of all TFs merged by overlap into one region set; a pair is the same pair when both ends fall in the
  same two merged regions. TF-specific = pair found by exactly 1 TF; conserved = found by >= 2 TFs.
Genes: protein-coding GENCODE v46 genes whose TSS lies within a pair end +-1 kb (same promoter rule as before). A gene touched by any
  conserved pair goes to Conserved; Specific = genes touched only by TF-specific pairs (groups disjoint within each panel).
Background, GO annotation and test as before: 12,388 genes with a promoter peak in any TF's network; goa_human.gaf + go-basic.obo,
  BP, propagated to ancestors, terms of 10-500 background genes, one-sided hypergeometric, BH within each list.
Size-matched check (TSV only): the larger group of each panel subsampled to the smaller group's annotated size, 200 draws (seed 1).
Terms: union of the q < 0.05 terms of all four columns (top 6 per column), near duplicates merged (background gene-set Jaccard > 0.6,
  better q kept). Dots only where q < 0.05; size = gene ratio, colour = -log10 q. 150 mm wide, Arial 6-7 pt; PNG + SVG + TSV.
  python scripts/j_specific_vs_conserved.py <outdir>   (hic env on Longleaf)"""
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
import re as _re
import scipy.sparse as sp
TSS = defaultdict(list)
for ln in gzip.open(X / "data/gencode_v46/gencode.v46.basic.annotation.gtf.gz", "rt"):
    if ln[0] == "#": continue
    x = ln.split("\t")
    if x[2] != "gene" or 'gene_type "protein_coding"' not in x[8]: continue
    TSS[x[0]].append((int(x[3]) if x[6] == "+" else int(x[4]), _re.search('gene_name "([^"]+)"', x[8]).group(1)))
TSS = {c: (np.array([a for a, _ in sorted(v)]), [b for _, b in sorted(v)]) for c, v in TSS.items()}
def genes_of(c, s, e):
    if c not in TSS: return []
    p, g = TSS[c]; lo, hi = np.searchsorted(p, s - 1000), np.searchsorted(p, e + 1000, side="right"); return g[lo:hi]
Nd = pd.read_csv(TG / "results_location/nodes_annotated.tsv.gz", sep="\t", keep_default_na=False)
Pn = Nd[Nd.gene != ""].assign(gene=lambda d: d.gene.str.split(",")).explode("gene"); Pn = Pn[~Pn.gene.str.startswith("ENSG")]
universe = Pn.gene.unique(); U = set(universe)
LISTS = {}
for panel, d in (("Module pairs", "results_cliques"), ("All pairs", "results")):
    inc = sp.load_npz(TG / d / "inc_pair.npz").tocsc(); ntf = np.diff(inc.indptr)
    pr = pd.read_csv(TG / d / "pairs.tsv", sep="\t"); rg = pd.read_csv(TG / d / "regions.tsv", sep="\t")
    rgen = [genes_of(c, s, e) for c, s, e in zip(rg.chrom, rg.start, rg.end)]
    gc, gs = set(), set()
    for r1, r2, n in zip(pr.r1.values, pr.r2.values, ntf):
        tgt = gc if n >= 2 else gs
        tgt.update(rgen[r1]); tgt.update(rgen[r2])
    gs -= gc; gc &= U; gs &= U
    LISTS[(panel, "TF-specific")] = sorted(gs); LISTS[(panel, "Conserved")] = sorted(gc)
    print(f"{panel}: pairs {len(pr):,}; conserved (>=2 TFs) {int((ntf >= 2).sum()):,}; genes TF-specific {len(gs):,}, conserved {len(gc):,}")
R = {k: enrich(v, universe) for k, v in LISTS.items()}
for k, r in R.items(): print(k, "GO-annotated", int(r.n_list.iloc[0]), "q<0.05 terms", int((r.q < 0.05).sum()))
t2g = defaultdict(set)
for g in universe:
    for t in g2t.get(g, ()): t2g[t].add(g)
cand = []
for k, r in R.items(): cand += list(r[r.q < 0.05].sort_values(["q", "p"]).index[:6])
cand = list(dict.fromkeys(cand))
if len(cand) < 4:
    for k, r in R.items(): cand += list(r.sort_values(["q", "p"]).index[:2])
    cand = list(dict.fromkeys(cand))
bestq = {t: min(R[k].q.get(t, 1) for k in R) for t in cand}; keep = []
for t in sorted(cand, key=lambda t: bestq[t]):
    if all(len(t2g[t] & t2g[u]) / max(len(t2g[t] | t2g[u]), 1) <= 0.6 for u in keep): keep.append(t)
rng = np.random.default_rng(1); SM = {}
for panel in ("Module pairs", "All pairs"):
    a = [g for g in LISTS[(panel, "TF-specific")] if g in g2t]; b = [g for g in LISTS[(panel, "Conserved")] if g in g2t]
    big, small, bigk = (a, b, "TF-specific") if len(a) > len(b) else (b, a, "Conserved")
    qs = {t: [] for t in keep}
    for _ in range(200):
        Rm = enrich(list(rng.choice(big, len(small), replace=False)), universe)
        for t in keep: qs[t].append(Rm.q[t] if t in Rm.index else 1.0)
    SM[panel] = (bigk, len(small), qs)
COLS = [("Module pairs", "TF-specific"), ("Module pairs", "Conserved"), ("All pairs", "TF-specific"), ("All pairs", "Conserved")]
rows = []
for t in keep:
    r = {"term": t, "name": name.get(t, "")}
    for k in COLS:
        x = R[k]; lab = f"{k[0]} | {k[1]}"
        r[f"{lab} q"] = x.q.get(t, np.nan); r[f"{lab} fold"] = x.fold.get(t, np.nan); r[f"{lab} genes"] = x.k.get(t, 0); r[f"{lab} gene_ratio"] = x.k.get(t, 0) / x.n_list.iloc[0]
    for panel in SM:
        bigk, n, qs = SM[panel]; r[f"{panel} | {bigk} size-matched to {n}: median q"] = float(np.median(qs[t])); r[f"{panel} | {bigk} size-matched to {n}: frac q<0.05"] = float(np.mean(np.array(qs[t]) < 0.05))
    rows.append(r)
T = pd.DataFrame(rows).set_index("term")
best = np.argmin(T[[f"{a} | {b} q" for a, b in COLS]].values, 1); T = T.iloc[np.lexsort((-T[[f"{a} | {b} q" for a, b in COLS]].min(1).values, best))]
T.round(5).to_csv(OUT / "j_gene_set_dotplot.tsv", sep="\t"); print(T[[c for c in T.columns if c.endswith(" q") or c == "name"]].round(3).to_string())
for panel in SM: print(panel, "size-matched", SM[panel][0], "->", SM[panel][1], {name[t][:30]: round(float(np.mean(np.array(SM[panel][2][t]) < 0.05)), 2) for t in keep})
SHORT = {"regulation of cellular response to stress": "Regulation of response to stress", "signal transduction in response to DNA damage": "DNA damage signal transduction",
         "endoplasmic reticulum to Golgi vesicle-mediated transport": "ER to Golgi vesicle transport",
         "detection of chemical stimulus involved in sensory perception": "Detection of chemical stimulus (sensory)", "mRNA metabolic process": "mRNA metabolic process",
         "regulation of mRNA metabolic process": "Regulation of mRNA metabolic process", "G protein-coupled receptor signaling pathway": "GPCR signaling pathway"}
cap = lambda s: SHORT.get(s, (s[0].upper() + s[1:]) if s else s)
from matplotlib.colors import LinearSegmentedColormap
cm = LinearSegmentedColormap.from_list("t", ["#C9E3DC", "#2C7A6B", "#0E3B33"]); MM = 1 / 25.4
allq = T[[f"{a} | {b} q" for a, b in COLS]].values; lq = -np.log10(np.clip(allq[allq < 0.05], 1e-12, 1)); vmin, vmax = 1.3, max(2.5, float(lq.max()) if lq.size else 2.5)
fig = plt.figure(figsize=(150 * MM, (26 + 4.4 * len(T)) * MM)); gs = fig.add_gridspec(1, 2, wspace=0.08, left=0.30, right=0.80, top=0.9, bottom=0.2)
for k, panel in enumerate(("Module pairs", "All pairs")):
    ax = fig.add_subplot(gs[k])
    for j, grp in enumerate(("TF-specific", "Conserved")):
        x = R[(panel, grp)]
        for i, t in enumerate(T.index):
            q = x.q.get(t, 1.0)
            if q < 0.05: ax.scatter(j, i, s=8 + 900 * x.k[t] / x.n_list.iloc[0], c=[[-np.log10(q)]], cmap=cm, vmin=vmin, vmax=vmax, lw=0, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"{g}\n({int(R[(panel, g)].n_list.iloc[0]):,})" for g in ("TF-specific", "Conserved")])
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(len(T) - 0.4, -0.6); ax.grid(color="#EEEEEE", lw=0.5); ax.set_axisbelow(True); ax.tick_params(length=0)
    ax.set_yticks(range(len(T))); ax.set_yticklabels([cap(s) for s in T.name] if k == 0 else []); ax.set_title(panel, fontsize=7)
    if k == 1: ax.spines["left"].set_visible(False)
    if not any((R[(panel, g)].q < 0.05).any() for g in ("TF-specific", "Conserved")): ax.text(0.5, (len(T) - 1) / 2, "No term\nq < 0.05", ha="center", va="center", fontsize=6, color="#8C8C8C")
cax = fig.add_axes([0.85, 0.58, 0.018, 0.28]); cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin, vmax)), cax=cax); cb.set_label("−log10 q", fontsize=6); cb.ax.tick_params(labelsize=5)
lax = fig.add_axes([0.83, 0.2, 0.15, 0.3]); lax.set_axis_off()
for v in (0.02, 0.05, 0.1): lax.scatter([], [], s=8 + 900 * v, color="#555555", label=f"{v:.0%}")
lax.legend(title="Gene ratio", loc="upper left", fontsize=5.5, title_fontsize=6, labelspacing=1.2, borderaxespad=0)
fig.text(0.58, 0.04, "Genes with a promoter at a pair end (n = annotated genes); empty = q ≥ 0.05", ha="center", fontsize=5.5, color="#4D4D4D")
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
stem = OUT / "j_gene_set_dotplot"; fig.savefig(f"{stem}.svg"); fig.savefig(f"{stem}.png", dpi=300)
p = Path(f"{stem}.svg"); p.write_text(illustrator_text(p.read_text())); print("font shorthand left:", p.read_text().count("font: "))
