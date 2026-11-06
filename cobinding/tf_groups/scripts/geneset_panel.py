"""Functional gene-set panel for clique target genes (protein-coding promoters, TSS +-1 kb).
(A) Universe = promoter peaks of GO-BP-annotated genes. Per TF x gene set: clique vs non-clique promoter peaks of the same TF, Mantel-Haenszel over local peak-density deciles
    (same strata as clique_location_density.py) -> log2 OR, p, BH over all TF x set tests. Tests whether a TF's cliques
    concentrate on a function beyond the TF's accessible promoters.
(B) Pooled: per gene set, genes with clique promoters in >=k TFs vs genes with promoter peaks in >=k TFs (k = 3, 5, 10),
    Fisher; and the same for each TF's clique genes vs all network-promoter genes (descriptive "what is captured").
args: location_dir outdir"""
import sys, gzip, numpy as np, pandas as pd
from collections import defaultdict
from pathlib import Path
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import StratifiedTable
LOC, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True); D = "/work/users/d/y/dyy12/XuLab/data/go"
SETS = {"GO:0000278": "Mitotic cell cycle", "GO:0044770": "Cell cycle phase transition", "GO:0006260": "DNA replication", "GO:0006281": "DNA repair",
        "GO:0008380": "RNA splicing", "GO:0006412": "Translation", "GO:0042254": "Ribosome biogenesis", "GO:0006457": "Protein folding",
        "GO:0009408": "Response to heat", "GO:0008610": "Lipid biosynthesis", "GO:0006119": "Oxidative phosphorylation", "GO:0006325": "Chromatin organization",
        "GO:0006511": "Ubiquitin-dependent catabolism", "GO:0006915": "Apoptotic process", "GO:0030154": "Cell differentiation", "GO:0007399": "Nervous system development",
        "GO:0009952": "Anterior/posterior patterning", "GO:0001525": "Angiogenesis", "GO:0006955": "Immune response", "GO:0016055": "Wnt signaling",
        "GO:0007165": "Signal transduction", "GO:0006357": "Pol II transcription regulation"}
par, cur = defaultdict(set), None
for ln in open(f"{D}/go-basic.obo"):
    ln = ln.strip()
    if ln == "[Term]": cur = None; continue
    if ln.startswith("id: GO:"): cur = ln[4:]
    elif cur and ln.startswith("is_a: "): par[cur].add(ln[6:16])
    elif cur and ln.startswith("relationship: part_of "): par[cur].add(ln[22:32])
anc = {}
def up(t):
    if t not in anc:
        s = {t}
        for p in par.get(t, ()): s |= up(p)
        anc[t] = s
    return anc[t]
sys.setrecursionlimit(10000); members = defaultdict(set); annotated = set()
for ln in gzip.open(f"{D}/goa_human.gaf.gz", "rt"):
    if ln[0] == "!": continue
    x = ln.split("\t")
    if "NOT" in x[3] or x[8] != "P": continue
    annotated.add(x[2])
    for t in up(x[4]) & SETS.keys(): members[t].add(x[2])
N = pd.read_csv(LOC / "nodes_annotated.tsv.gz", sep="\t", keep_default_na=False); N["mid"] = (N.start + N.end) // 2; dens = np.zeros(len(N))
for (tf, c), idx in N.groupby(["tf", "chrom"]).groups.items():
    m = np.sort(N.mid.values[idx]); x = N.mid.values[idx]; dens[idx] = np.searchsorted(m, x + 5e5, side="right") - np.searchsorted(m, x - 5e5)
N["dens"] = dens
tfc = N.groupby("tf").clique.agg(["sum", "size"]); tfs = tfc.index[(tfc["sum"] >= 30) & (tfc["size"] - tfc["sum"] >= 30)]
N = N[N.tf.isin(tfs)].copy(); N["dec"] = N.groupby("tf").dens.transform(lambda v: pd.qcut(v.rank(method="first"), 10, labels=False))
P = N[N.gene != ""].copy(); P["genes"] = P.gene.str.split(",").apply(lambda g: [x for x in g if x in annotated])
P = P[P.genes.str.len() > 0].copy()   # universe = promoters of genes with any GO BP annotation (well-annotated genes are more often clique promoters)
rows = []
for t, nm in SETS.items():
    S = members[t]; P["hit"] = P.genes.apply(lambda g: any(x in S for x in g))
    for tf, g in P.groupby("tf"):
        tabs = []
        for _, h in g.groupby("dec"):
            a = (h.hit & (h.clique == 1)).sum(); b = (~h.hit & (h.clique == 1)).sum(); c = (h.hit & (h.clique == 0)).sum(); d = (~h.hit & (h.clique == 0)).sum()
            if min(a + b, c + d, a + c, b + d) > 0: tabs.append(np.array([[a, b], [c, d]]))
        if len(tabs) < 2: continue
        st = StratifiedTable(tabs)
        rows.append({"set": nm, "tf": tf, "log2OR": np.log2(st.oddsratio_pooled), "p": st.test_null_odds().pvalue,
                     "clique_genes_in_set": int((g.hit & (g.clique == 1)).sum()), "clique_genes": int((g.clique == 1).sum())})
A = pd.DataFrame(rows); A["q"] = multipletests(A.p, method="fdr_bh")[1]; A.to_csv(OUT / "panel_tf_x_set_density_matched.tsv", sep="\t", index=False)
print(f"(A) {len(A)} TF x set tests; q<0.05: {(A.q < 0.05).sum()}; q<0.1: {(A.q < 0.1).sum()}")
print(A.sort_values("p").head(15).round(4).to_string(index=False))
M = A.pivot(index="tf", columns="set", values="log2OR"); M.to_csv(OUT / "panel_log2OR_matrix.tsv", sep="\t")
print("median log2OR per set:"); print(M.median().sort_values().round(3).to_string())
# (B) pooled, no density matching (descriptive)
G = P.explode("genes").groupby(["genes", "tf"]).clique.max().reset_index()
cnt = G.groupby("genes").agg(n_node=("tf", "size"), n_clique=("clique", "sum"))
brows = []
for t, nm in SETS.items():
    inset = cnt.index.isin(members[t])
    for k in (3, 5, 10):
        bg = cnt.n_node >= k; fg = cnt.n_clique >= k
        a = (fg & inset).sum(); b = (fg & ~inset).sum(); c = (bg & ~fg & inset).sum(); d = (bg & ~fg & ~inset).sum()
        orr, p = fisher_exact([[a, b], [c, d]])
        brows.append({"set": nm, "k": k, "genes_clique_geK": a, "frac_in_set_clique": a / max(a + b, 1), "frac_in_set_bg": (a + c) / max(a + b + c + d, 1), "OR": orr, "p": p})
B = pd.DataFrame(brows); B["q"] = multipletests(B.p, method="fdr_bh")[1]; B.to_csv(OUT / "panel_pooled_geK.tsv", sep="\t", index=False)
print("\n(B) pooled clique genes in >=k TFs vs genes with peaks in >=k TFs"); print(B.sort_values("p").head(15).round(4).to_string(index=False))
# (C) pooled, density-matched: per gene, observed clique TFs vs expected (sum over TFs of the TF x density-decile clique
# rate for that promoter peak). Per set: (O/E in set) / (O/E all genes); p from 2,000 random gene sets of the same size
# drawn within the same n_node bins.
N["p_exp"] = N.groupby(["tf", "dec"]).clique.transform("mean")
Q = N[N.gene != ""].copy(); Q["genes"] = Q.gene.str.split(",").apply(lambda g: [x for x in g if x in annotated]); Q = Q[Q.genes.str.len() > 0].explode("genes")
Q = Q.groupby(["genes", "tf"]).agg(o=("clique", "max"), e=("p_exp", "max")).reset_index()
GE = Q.groupby("genes").agg(n_node=("tf", "size"), O=("o", "sum"), E=("e", "sum")); GE = GE[GE.n_node >= 3]
GE["bin"] = pd.qcut(GE.n_node.rank(method="first"), 10, labels=False); base = GE.O.sum() / GE.E.sum(); rng = np.random.default_rng(2)
byb = {b: g.index.values for b, g in GE.groupby("bin")}; crow = []
for t, nm in SETS.items():
    inset = GE.index.isin(members[t]); sub = GE[inset]
    if len(sub) < 10: continue
    r = (sub.O.sum() / sub.E.sum()) / base; cnts = sub.bin.value_counts()
    null = []
    for _ in range(2000):
        idx = np.concatenate([rng.choice(byb[b], n, replace=False) for b, n in cnts.items()]); s2 = GE.loc[idx]; null.append((s2.O.sum() / s2.E.sum()) / base)
    null = np.array(null); crow.append({"set": nm, "n_genes": len(sub), "OE_ratio": r, "null_lo": np.quantile(null, .025), "null_hi": np.quantile(null, .975),
                                        "p_two_sided": min(1, 2 * min((np.sum(null >= r) + 1) / 2001, (np.sum(null <= r) + 1) / 2001))})
C = pd.DataFrame(crow); C["q"] = multipletests(C.p_two_sided, method="fdr_bh")[1]; C = C.sort_values("OE_ratio", ascending=False)
C.to_csv(OUT / "panel_pooled_density_matched.tsv", sep="\t", index=False)
print("\n(C) pooled, density-matched (genes with peaks in >=3 TFs)"); print(C.round(3).to_string(index=False))
