"""GO step 1 (user 2026-11-17): how many TFs' higher-order coordinated binding modules each gene sits in, to choose the TF-specific vs
conserved cutoff. Module gene of a TF = protein-coding gene whose TSS +-1 kb overlaps an element of one of that TF's modules (>=3 elements,
all pairs linked, FDR <= 0.05; ourrun_k50, RBBP4 sweep run): clique == 1 in cobinding/tf_groups/results_location/nodes_annotated.tsv.gz, as
panel j / go_cliques.py. GO-annotated = has a BP annotation in goa_human.gaf (NOT dropped), the files of panel j.
Writes go_gene_tf_counts.png/.svg, one .tsv per sheet and go_gene_tf_counts.xlsx when openpyxl is present (sheets: distribution, per_TF, summary, candidate_cutoffs, genes).
  python scripts/go_gene_tf_counts.py <outdir>   (Longleaf, hic env)"""
import glob, gzip, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in glob.glob("/work/users/d/y/dyy12/fonts/arial/*.ttf"): fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 6, "axes.labelsize": 7, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.linewidth": 0.6, "legend.frameon": False})
X = Path("/work/users/d/y/dyy12/XuLab"); TG = X / "cobinding/tf_groups"; OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ann = set()
for ln in gzip.open(X / "data/go/goa_human.gaf.gz", "rt"):
    if ln[0] == "!": continue
    x = ln.split("\t")
    if "NOT" in x[3] or x[8] != "P": continue
    ann.add(x[2])
N = pd.read_csv(TG / "results_location/nodes_annotated.tsv.gz", sep="\t", keep_default_na=False)
P = N[N.gene != ""].assign(gene=lambda d: d.gene.str.split(",")).explode("gene")
P = P[~P.gene.str.startswith("ENSG")].groupby(["tf", "gene"]).clique.max().reset_index()
MG = P[P.clique == 1]; cnt = MG.groupby("gene").tf.nunique().rename("n_tfs").to_frame(); cnt["go_annotated"] = cnt.index.isin(ann)
cnt["tfs"] = MG.groupby("gene").tf.apply(lambda s: ",".join(sorted(s)))
mx = int(cnt.n_tfs.max()); k = np.arange(1, mx + 1)
n = cnt.n_tfs.value_counts().reindex(k, fill_value=0); na = cnt[cnt.go_annotated].n_tfs.value_counts().reindex(k, fill_value=0)
D = pd.DataFrame({"TFs_per_gene": k, "n_genes": n.values, "n_annotated": na.values})
D["cum_pct_genes"] = (100 * D.n_genes.cumsum() / D.n_genes.sum()).round(2); D["cum_pct_annotated"] = (100 * D.n_annotated.cumsum() / D.n_annotated.sum()).round(2)
D["pct_ge_this"] = (100 * D.n_genes[::-1].cumsum()[::-1] / D.n_genes.sum()).round(2)
q = cnt.n_tfs.quantile([.5, .75, .9, .95, .99]); qa = cnt[cnt.go_annotated].n_tfs.quantile([.5, .75, .9, .95, .99])
S = pd.DataFrame({"stat": ["genes", "GO-annotated genes", "TFs with modules", "network promoter genes (background)", "median", "75%", "90%", "95%", "99%", "max"],
                  "all module genes": [len(cnt), int(cnt.go_annotated.sum()), MG.tf.nunique(), P.gene.nunique(), *q.values, mx],
                  "GO-annotated": [int(cnt.go_annotated.sum()), int(cnt.go_annotated.sum()), MG.tf.nunique(), P.gene.nunique(), *qa.values, int(cnt[cnt.go_annotated].n_tfs.max())]})
cls = pd.cut(cnt.n_tfs, [0, 1, 2, 4, 9, 10 ** 6], labels=["1", "2", "3-4", "5-9", ">=10"])
T = MG.join(cls.rename("cls"), on="gene").groupby(["tf", "cls"], observed=False).size().unstack(fill_value=0)
T["total"] = T.sum(1); T = T.sort_values("total", ascending=False)
for c in ["1", "2", "3-4", "5-9", ">=10"]: T[f"% {c}"] = (100 * T[c] / T.total).round(1)
def side(lo_max, hi_min):
    a = cnt[cnt.n_tfs <= lo_max]; b = cnt[cnt.n_tfs >= hi_min]
    return [len(a), int(a.go_annotated.sum()), len(b), int(b.go_annotated.sum()), len(cnt) - len(a) - len(b)]
p90 = int(D.TFs_per_gene[D.pct_ge_this <= 10].min())   # smallest TF count whose tail holds <= 10% of genes
C = pd.DataFrame([["A: specific = 1 TF vs conserved >= 2 TFs", *side(1, 2)], ["B: specific = 1 TF vs conserved >= 3 TFs (gap: 2 TFs left out)", *side(1, 3)],
                  [f"C: specific = 1 TF vs conserved = top 10% (>= {p90} TFs)", *side(1, p90)], ["D: specific = 1 TF vs conserved >= 5 TFs", *side(1, 5)]],
                 columns=["cutoff", "specific genes", "specific annotated", "conserved genes", "conserved annotated", "genes left out"])
SH = {"distribution": (D, False), "per_TF": (T, True), "summary": (S, False), "candidate_cutoffs": (C, False), "genes": (cnt.sort_values("n_tfs", ascending=False), True)}
for nm, (df, ix) in SH.items(): df.to_csv(OUT / f"{nm}.tsv", sep="\t", index=ix)
try:
    with pd.ExcelWriter(OUT / "go_gene_tf_counts.xlsx") as w:
        for nm, (df, ix) in SH.items(): df.to_excel(w, sheet_name=nm, index=ix)
except ImportError: print("no openpyxl here: sheets written as .tsv (xlsx built on the Mac)")
MM = 1 / 25.4; fig, ax = plt.subplots(figsize=(120 * MM, 60 * MM)); fig.subplots_adjust(left=0.1, right=0.88, bottom=0.18, top=0.95)
ax.bar(k - 0.2, D.n_genes, 0.4, color="#4D4D4D", label="All module genes"); ax.bar(k + 0.2, D.n_annotated, 0.4, color="#BDBDBD", label="GO-annotated")
ax.set_yscale("log"); ax.set_xlabel("Number of TFs whose modules contain the gene (TSS ± 1 kb)"); ax.set_ylabel("Genes"); ax.set_xlim(0.3, mx + 0.7)
ax.set_xticks(k[(k - 1) % max(1, mx // 15) == 0] if mx > 20 else k)
a2 = ax.twinx(); a2.plot(k, D.cum_pct_genes, color="#B83A2E", lw=0.8, marker="o", ms=1.8); a2.set_ylim(0, 100); a2.set_ylabel("Cumulative % of genes", color="#B83A2E")
a2.tick_params(axis="y", colors="#B83A2E"); a2.spines["right"].set_color("#B83A2E"); a2.spines["top"].set_visible(False)
ax.legend(loc="upper right", bbox_to_anchor=(0.97, 0.75), fontsize=6)
fig.savefig(OUT / "go_gene_tf_counts.png", dpi=300); fig.savefig(OUT / "go_gene_tf_counts.svg")
print(D.to_string(index=False)); print(S.to_string(index=False)); print(C.to_string(index=False)); print(T.head(8).to_string())
