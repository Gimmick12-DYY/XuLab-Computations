"""Where do cliques sit? Per TF, compare its clique regions with its other network nodes (peaks with >=1 FDR<=0.05 Cicero
edge but in no clique), so accessibility/co-accessibility is matched by construction.
(1) chromatin class (protein-coding TSS +-1 kb, else HEK293T ChromHMM18 class) and WT Hi-C A/B (100 kb E1): per-TF odds
    ratio clique vs non-clique node, pooled across TFs (median, sign test).
(2) 1 Mb hotspots: per TF expected clique nodes in a bin = its non-clique-matched rate x its nodes there; bins scored by
    number of TFs with binomial p<0.01 and by TF-averaged log2 O/E; a bin's genes listed.
(3) promoter genes: per gene, number of TFs whose cliques contain its promoter vs expected from TF clique rates.
args: outdir"""
import sys, gzip, re, numpy as np, pandas as pd
from collections import defaultdict
from pathlib import Path
from scipy.stats import binom, binomtest
ROOT = Path("/work/users/d/y/dyy12/XuLab"); OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
KEEP = {f"chr{i}" for i in range(1, 23)} | {"chrX"}
DIRS = {p.name: p for p in sorted((ROOT / "cobinding/results/ourrun_k50").iterdir()) if (p / "nodes.tsv").exists()}
DIRS["RBBP4"] = ROOT / "cobinding/results/RBBP4_sweep_k50_i10000_s555"
def iv(r): c, se = r.split(":"); s, e = se.split("-"); return c, int(s), int(e)
# annotation tables
tss = defaultdict(list)
for ln in gzip.open(ROOT / "data/gencode_v46/gencode.v46.basic.annotation.gtf.gz", "rt"):
    if ln[0] == "#": continue
    x = ln.split("\t")
    if x[2] != "gene" or 'gene_type "protein_coding"' not in x[8]: continue
    tss[x[0]].append((int(x[3]) if x[6] == "+" else int(x[4]), re.search('gene_name "([^"]+)"', x[8]).group(1)))
tss = {c: (np.array([a for a, _ in sorted(v)]), [b for _, b in sorted(v)]) for c, v in tss.items()}
CL = {"TssA": "TSS-flank", "TssFlnk": "TSS-flank", "TssFlnkU": "TSS-flank", "TssFlnkD": "TSS-flank", "EnhA1": "Enh-active", "EnhA2": "Enh-active",
      "EnhG1": "Enh-active", "EnhG2": "Enh-active", "EnhWk": "Enh-weak", "TssBiv": "Bivalent", "EnhBiv": "Bivalent", "Tx": "Transcribed", "TxWk": "Transcribed",
      "ReprPC": "Polycomb", "ReprPCWk": "Polycomb", "Het": "Het/ZNF", "ZNF/Rpts": "Het/ZNF", "Quies": "Quiescent"}
hmm = defaultdict(list)
for ln in gzip.open(ROOT / "data/HEK293T_chromHMM18.bed.gz", "rt"):
    x = ln.split("\t"); hmm[x[0]].append((int(x[1]), int(x[2]), x[3]))
hmm = {c: (np.array([a for a, _, _ in v]), np.array([b for _, b, _ in v]), [s for _, _, s in v]) for c, v in hmm.items()}
E1 = {}
ef = ROOT / "hic/work_wt293t/compartments/compartments_100000.E1.bedGraph"
if ef.exists():
    for ln in open(ef):
        x = ln.split()
        if len(x) == 4 and x[3] not in ("nan", ""): E1[(x[0], int(x[1]) // 100000)] = float(x[3])
def annot(c, s, e):
    g = ""
    if c in tss:
        P, N = tss[c]; lo, hi = np.searchsorted(P, s - 1000), np.searchsorted(P, e + 1000)
        if hi > lo: g = ",".join(sorted(set(N[lo:hi])))
    if g: cl = "Promoter"
    else:
        S, Ee, St = hmm.get(c, (np.array([0]), np.array([0]), ["Quies"])); k0 = np.searchsorted(Ee, s, side="right"); best, st = 0, "Quies"
        for k in range(k0, min(k0 + 20, len(S))):
            if S[k] >= e: break
            ov = min(e, Ee[k]) - max(s, S[k])
            if ov > best: best, st = ov, St[k]
        cl = CL[st]
    ab = E1.get((c, ((s + e) // 2) // 100000), np.nan)
    return cl, g, ("A" if ab > 0 else "B") if ab == ab else "NA"
rows = []
for tf, d in DIRS.items():
    n = pd.read_csv(d / "nodes.tsv", sep="\t", usecols=["peak", "n_cliques"])
    for p, k in zip(n.peak, n.n_cliques):
        c, s, e = iv(p)
        if c in KEEP: rows.append((tf, c, s, e, int(k > 0)))
N = pd.DataFrame(rows, columns=["tf", "chrom", "start", "end", "clique"])
cache = {}
ann = [cache.setdefault((c, s, e), annot(c, s, e)) for c, s, e in zip(N.chrom, N.start, N.end)]
N["cls"], N["gene"], N["AB"] = zip(*ann)
N.to_csv(OUT / "nodes_annotated.tsv.gz", sep="\t", index=False)
tfc = N.groupby("tf").clique.agg(["sum", "size"]); tfs = tfc.index[(tfc["sum"] >= 30) & (tfc["size"] - tfc["sum"] >= 30)].tolist()
print(len(tfs), "TFs with >=30 clique and >=30 non-clique nodes")
# (1) class enrichment
res = []
for cat, col in [(c, "cls") for c in sorted(N.cls.unique())] + [("A compartment", "AB")]:
    for tf in tfs:
        x = N[N.tf == tf]
        if col == "AB": x = x[x.AB != "NA"]; hit = x.AB == "A"
        else: hit = x.cls == cat
        a = (hit & (x.clique == 1)).sum(); b = (~hit & (x.clique == 1)).sum(); c_ = (hit & (x.clique == 0)).sum(); d_ = (~hit & (x.clique == 0)).sum()
        res.append({"category": cat, "tf": tf, "frac_clique": a / max(a + b, 1), "frac_nonclique": c_ / max(c_ + d_, 1), "log2OR": np.log2(((a + .5) * (d_ + .5)) / ((b + .5) * (c_ + .5)))})
R = pd.DataFrame(res); R.to_csv(OUT / "class_enrichment_per_tf.tsv", sep="\t", index=False)
S = R.groupby("category").agg(median_log2OR=("log2OR", "median"), n_tf_up=("log2OR", lambda v: (v > 0).sum()), n_tf=("log2OR", "size"),
                               median_frac_clique=("frac_clique", "median"), median_frac_nonclique=("frac_nonclique", "median"))
S["sign_test_p"] = [binomtest(int(u), int(n)).pvalue for u, n in zip(S.n_tf_up, S.n_tf)]
S.sort_values("median_log2OR", ascending=False).to_csv(OUT / "class_enrichment_summary.tsv", sep="\t"); print(S.sort_values("median_log2OR", ascending=False).round(3).to_string())
# (2) 1 Mb hotspots
N["bin"] = N.chrom + ":" + ((N.start + N.end) // 2 // 1_000_000).astype(str)
X = N[N.tf.isin(tfs)]; rate = X.groupby("tf").clique.mean()
g = X.groupby(["bin", "tf"]).clique.agg(["sum", "size"]).reset_index(); g["exp"] = g["size"] * g.tf.map(rate)
g["p"] = binom.sf(g["sum"] - 1, g["size"], g.tf.map(rate)); g["l2oe"] = np.log2((g["sum"] + .5) / (g["exp"] + .5))
H = g.groupby("bin").agg(n_tf_present=("tf", "size"), n_tf_sig=("p", lambda v: (v < 0.01).sum()), mean_l2oe=("l2oe", "mean"), obs=("sum", "sum"), exp=("exp", "sum"))
H["exp_n_tf_sig"] = H.n_tf_present * 0.01; H = H[H.n_tf_present >= 10]
H["p_bin"] = binom.sf(H.n_tf_sig - 1, H.n_tf_present, 0.01)
from statsmodels.stats.multitest import multipletests
H["q_bin"] = multipletests(H.p_bin, method="fdr_bh")[1]
genes = X[X.clique == 1].groupby("bin").gene.apply(lambda s: ",".join(sorted(set(x for v in s if v for x in v.split(",")))[:25]))
H["clique_promoter_genes"] = genes.reindex(H.index).fillna("")
H.sort_values(["q_bin", "mean_l2oe"], ascending=[True, False]).to_csv(OUT / "hotspots_1Mb.tsv", sep="\t")
print((H.q_bin < 0.05).sum(), "of", len(H), "1 Mb bins with excess TFs enriched for cliques (q<0.05)")
print(H.sort_values(["q_bin", "mean_l2oe"], ascending=[True, False]).head(25)[["n_tf_present", "n_tf_sig", "mean_l2oe", "q_bin", "clique_promoter_genes"]].to_string())
# (3) promoter genes
P = X[X.gene != ""].assign(gene=lambda d: d.gene.str.split(",")).explode("gene")
P = P.groupby(["gene", "tf"]).clique.max().reset_index(); P["r"] = P.tf.map(rate)
GG = P.groupby("gene").agg(n_tf_node=("tf", "size"), n_tf_clique=("clique", "sum"), exp=("r", "sum"))
GG["p"] = [binomtest(int(k), int(n), e / n, alternative="greater").pvalue if n else 1 for k, n, e in zip(GG.n_tf_clique, GG.n_tf_node, GG.exp)]
GG["q"] = multipletests(GG.p, method="fdr_bh")[1]
GG.sort_values("p").to_csv(OUT / "promoter_genes_clique_excess.tsv", sep="\t")
(OUT / "go").mkdir(exist_ok=True); (OUT / "go/background.txt").write_text("\n".join(GG.index) + "\n")
(OUT / "go/clique_excess_q05.txt").write_text("\n".join(GG.index[GG.q < 0.05]) + "\n")
print((GG.q < 0.05).sum(), "promoter genes with excess clique membership across TFs (q<0.05) of", len(GG)); print(GG.sort_values("p").head(30).round(4).to_string())
