"""Fig 3 v7 extra panel: functions of the genes that higher-order co-binding modules sit on (promoter genes, TSS +-1 kb, in modules
of >=3 TFs; 1,035 genes) vs all network promoter genes (10,934), and how much of it survives matching for local peak density.
Unmatched (filled dot): hypergeometric GO BP fold and BH q from cobinding/tf_groups/scripts/go_cliques.py (commit 00548af,
results_go/go_cliques_universe.tsv.gz, list pooled_ge3TF). Matched (hollow dot): the density-matched pooled test of
cobinding/tf_groups/scripts/geneset_panel.py part (C) (commit 1c951c7), recomputed here for the GO terms shown: per gene, observed
module TFs vs expected from each TF's module rate in its local peak-density decile (+-500 kb), genes with peaks in >=3 TFs;
value = (O/E in the term) / (O/E of all genes); 2,000 random gene sets of the same size within n-TF bins give p (two-sided), BH.
Terms: q < 0.05 unmatched, plus DNA repair, RNA splicing regulation and ER-Golgi transport (q < 0.1); near-duplicate
terms dropped (mitotic cell cycle process, regulation of TORC1 signaling); chromatin organization (>500 genes) was outside the 10-500
term-size range of the GO run, so it has no unmatched fold and is not shown; heat-response / unfolded-protein terms left out (heat-shock genes, exon caution; see legend notes).
Final size 85 x 55 mm, Arial, 6-7 pt, Palette C neutrals; SVG text written as explicit Arial attributes (Illustrator-safe).
Runs on Longleaf:  python fig3_geneset_dotplot.py <outdir>"""
import glob, gzip, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in glob.glob("/work/users/d/y/dyy12/fonts/arial/*.ttf"): fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 6, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 0, "legend.frameon": False})
X = Path("/work/users/d/y/dyy12/XuLab"); TG = X / "cobinding/tf_groups"; D = X / "data/go"; OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
TERMS = [("GO:0000278", "Mitotic cell cycle"), ("GO:0051052", "Regulation of DNA metabolic process"), ("GO:0006282", "Regulation of DNA repair"),
         ("GO:0006281", "DNA repair"), ("GO:0080135", "Regulation of cellular response to stress"),
         ("GO:1904263", "Positive regulation of TORC1 signaling"), ("GO:0043484", "Regulation of RNA splicing"),
         ("GO:0006888", "ER to Golgi vesicle transport")]
U = pd.read_csv(TG / "results_go/go_cliques_universe.tsv.gz", sep="\t"); U = U[U.list == "pooled_ge3TF"].set_index("term")
# ---- density-matched O/E (geneset_panel.py part C) for these terms ----
par, cur = defaultdict(set), None
for ln in open(D / "go-basic.obo"):
    ln = ln.strip()
    if ln == "[Term]": cur = None; continue
    if ln.startswith("id: GO:"): cur = ln[4:]
    elif cur and ln.startswith("is_a: "): par[cur].add(ln[6:16])
    elif cur and ln.startswith("relationship: part_of "): par[cur].add(ln[22:32])
anc = {}; sys.setrecursionlimit(10000)
def up(t):
    if t not in anc:
        s = {t}
        for p in par.get(t, ()): s |= up(p)
        anc[t] = s
    return anc[t]
want = {t for t, _ in TERMS}; members = defaultdict(set); annotated = set()
for ln in gzip.open(D / "goa_human.gaf.gz", "rt"):
    if ln[0] == "!": continue
    x = ln.split("\t")
    if "NOT" in x[3] or x[8] != "P": continue
    annotated.add(x[2])
    for t in up(x[4]) & want: members[t].add(x[2])
N = pd.read_csv(TG / "results_location/nodes_annotated.tsv.gz", sep="\t", keep_default_na=False); N["mid"] = (N.start + N.end) // 2; dens = np.zeros(len(N))
for (tf, c), idx in N.groupby(["tf", "chrom"]).groups.items():
    m = np.sort(N.mid.values[idx]); x = N.mid.values[idx]; dens[idx] = np.searchsorted(m, x + 5e5, side="right") - np.searchsorted(m, x - 5e5)
N["dens"] = dens; tfc = N.groupby("tf").clique.agg(["sum", "size"]); tfs = tfc.index[(tfc["sum"] >= 30) & (tfc["size"] - tfc["sum"] >= 30)]
N = N[N.tf.isin(tfs)].copy(); N["dec"] = N.groupby("tf").dens.transform(lambda v: pd.qcut(v.rank(method="first"), 10, labels=False))
N["p_exp"] = N.groupby(["tf", "dec"]).clique.transform("mean")
Q = N[N.gene != ""].copy(); Q["genes"] = Q.gene.str.split(",").apply(lambda g: [x for x in g if x in annotated]); Q = Q[Q.genes.str.len() > 0].explode("genes")
Q = Q.groupby(["genes", "tf"]).agg(o=("clique", "max"), e=("p_exp", "max")).reset_index()
GE = Q.groupby("genes").agg(n_node=("tf", "size"), O=("o", "sum"), E=("e", "sum")); GE = GE[GE.n_node >= 3]
GE["bin"] = pd.qcut(GE.n_node.rank(method="first"), 10, labels=False); base = GE.O.sum() / GE.E.sum(); rng = np.random.default_rng(2)
byb = {b: g.index.values for b, g in GE.groupby("bin")}; rows = []
def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); r = p[o] * len(p) / np.arange(1, len(p) + 1); q = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out
for t, nm in TERMS:
    inset = GE.index.isin(members[t]); sub = GE[inset]; r = (sub.O.sum() / sub.E.sum()) / base; cnts = sub.bin.value_counts(); null = []
    for _ in range(2000):
        idx = np.concatenate([rng.choice(byb[b], n, replace=False) for b, n in cnts.items()]); s2 = GE.loc[idx]; null.append((s2.O.sum() / s2.E.sum()) / base)
    null = np.array(null); u = U.loc[t]
    rows.append(dict(term=t, name=nm, genes=int(u.k), fold=float(u.fold), q=float(u.q), matched_oe=r, matched_lo=np.quantile(null, .025), matched_hi=np.quantile(null, .975),
                     matched_p=min(1, 2 * min((np.sum(null >= r) + 1) / 2001, (np.sum(null <= r) + 1) / 2001)), matched_n=len(sub)))
R = pd.DataFrame(rows); R["matched_q"] = bh(R.matched_p); R = R.sort_values("fold").reset_index(drop=True)
R.round(5).to_csv(OUT / "fig3_geneset_dotplot.tsv", sep="\t", index=False); print(R[["name", "genes", "fold", "q", "matched_oe", "matched_q"]].round(3).to_string(index=False))
# ---- plot, final size 85 x 55 mm ----
MM = 1 / 25.4; INK, GREY = "#1A1A1A", "#9E9E9E"
fig = plt.figure(figsize=(85 * MM, 55 * MM)); ax = fig.add_axes([0.47, 0.33, 0.40, 0.65])
y = np.arange(len(R)); sz = lambda n: 4 + 0.35 * n
ax.hlines(y, R.matched_oe, R.fold, color=GREY, lw=0.6, zorder=1)
ax.scatter(R.matched_oe, y, s=14, facecolors="white", edgecolors=GREY, linewidths=0.7, zorder=2)
ax.scatter(R.fold, y, s=sz(R.genes), color=INK, lw=0, zorder=3)
ax.axvline(1, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=0)
ax.set_yticks(y); ax.set_yticklabels(R.name); ax.set_ylim(-0.7, len(R) - 0.3)
xmax = max(3.4, R.fold.max() + 0.3); ax.set_xlim(0.6, xmax); ax.set_xlabel("Fold enrichment")
for i, q in enumerate(R.q): ax.text(xmax + 0.05, i, f"q = {q:.2g}" if q >= 0.001 else "q < 0.001", va="center", ha="left", fontsize=5.5, color=INK, clip_on=False)
h = [plt.Line2D([], [], ls="", marker="o", ms=4, color=INK), plt.Line2D([], [], ls="", marker="o", ms=3.6, mfc="white", mec=GREY, mew=0.7)]
l1 = ax.legend(h, ["Unmatched", "Matched for local peak density"], loc="upper left", bbox_to_anchor=(-1.15, -0.22), ncol=2, fontsize=6, handletextpad=0.3, columnspacing=1.0, borderaxespad=0)
ax.add_artist(l1)
hs = [plt.scatter([], [], s=sz(n), color=INK, lw=0) for n in (20, 50, 80)]
ax.legend(hs, ["20", "50", "80 genes"], loc="upper left", bbox_to_anchor=(-1.15, -0.36), ncol=3, fontsize=6, handletextpad=0.1, columnspacing=0.6, borderaxespad=0)
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
fig.savefig(OUT / "fig3_geneset_dotplot.svg"); fig.savefig(OUT / "fig3_geneset_dotplot.pdf"); fig.savefig(OUT / "fig3_geneset_dotplot.png", dpi=300)
p = OUT / "fig3_geneset_dotplot.svg"; p.write_text(illustrator_text(p.read_text())); print("font shorthand left:", p.read_text().count("font: "))
