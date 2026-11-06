"""GO BP of clique target genes (protein-coding promoters, TSS +-1 kb, that are clique nodes).
Two backgrounds: (a) "network": genes whose promoter is a network node of the same TF(s) (accessibility/co-accessibility
matched); (b) "universe": all protein-coding genes with a promoter node in any TF. Lists:
  per TF (>=30 clique promoter genes); pooled genes in cliques of >=3 / >=5 / >=10 TFs (background: genes whose
  promoter is a node in >=3/5/10 TFs for (a)).
Hypergeometric, terms 10-500 genes in background, BH within list. Writes go_cliques_{bg}.tsv and a recurrence summary.
args: location_dir (nodes_annotated.tsv.gz from clique_location.py) outdir"""
import sys, gzip, numpy as np, pandas as pd
from collections import defaultdict
from pathlib import Path
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
LOC = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True); D = "/work/users/d/y/dyy12/XuLab/data/go"
par, name, cur = defaultdict(set), {}, None
for ln in open(f"{D}/go-basic.obo"):
    ln = ln.strip()
    if ln == "[Term]": cur = None; continue
    if ln.startswith("id: GO:"): cur = ln[4:]
    elif cur and ln.startswith("name: "): name[cur] = ln[6:]
    elif cur and ln.startswith("is_a: "): par[cur].add(ln[6:16])
    elif cur and ln.startswith("relationship: part_of "): par[cur].add(ln[22:32])
anc = {}
def up(t):
    if t not in anc:
        s = {t}
        for p in par.get(t, ()): s |= up(p)
        anc[t] = s
    return anc[t]
sys.setrecursionlimit(10000); g2t = defaultdict(set)
for ln in gzip.open(f"{D}/goa_human.gaf.gz", "rt"):
    if ln[0] == "!": continue
    x = ln.split("\t")
    if "NOT" in x[3] or x[8] != "P": continue
    g2t[x[2]] |= up(x[4])
def enrich(lst, bg, label):
    bg = [g for g in set(bg) if g in g2t]; L = set(lst) & set(bg); N, n = len(bg), len(L)
    if n < 10: return []
    t2g = defaultdict(set)
    for g in bg:
        for t in g2t[g]: t2g[t].add(g)
    rows = []
    for t, s in t2g.items():
        K = len(s)
        if not 10 <= K <= 500: continue
        k = len(s & L)
        rows.append({"list": label, "term": t, "name": name.get(t, ""), "k": k, "n_list": n, "K": K, "N": N, "fold": (k / n) / (K / N) if k else 0,
                     "p": hypergeom.sf(k - 1, N, K, n), "genes": ",".join(sorted(s & L)[:25])})
    if rows:
        q = multipletests([r["p"] for r in rows], method="fdr_bh")[1]
        for r, v in zip(rows, q): r["q"] = v
    return rows
N = pd.read_csv(LOC / "nodes_annotated.tsv.gz", sep="\t", keep_default_na=False)
P = N[N.gene != ""].assign(gene=lambda d: d.gene.str.split(",")).explode("gene")
P = P[~P.gene.str.startswith("ENSG")].groupby(["tf", "gene"]).clique.max().reset_index()
universe = P.gene.unique()
out = {"network": [], "universe": []}
for tf, g in P.groupby("tf"):
    cl = g.gene[g.clique == 1]
    if len(cl) < 30: continue
    out["network"] += enrich(cl, g.gene, tf); out["universe"] += enrich(cl, universe, tf)
cnt = P.groupby("gene").agg(n_node=("tf", "size"), n_clique=("clique", "sum"))
for k in (3, 5, 10):
    lst = cnt.index[cnt.n_clique >= k]
    out["network"] += enrich(lst, cnt.index[cnt.n_node >= k], f"pooled_ge{k}TF"); out["universe"] += enrich(lst, universe, f"pooled_ge{k}TF")
    (OUT / f"pooled_ge{k}TF_genes.txt").write_text("\n".join(sorted(lst)) + "\n")
for bg, rows in out.items():
    R = pd.DataFrame(rows).sort_values(["list", "p"]); R.to_csv(OUT / f"go_cliques_{bg}.tsv.gz", sep="\t", index=False)
    S = R[R.q < 0.05]; per = S.groupby("list").size()
    tfl = [l for l in R.list.unique() if not l.startswith("pooled")]
    rec = S[~S.list.str.startswith("pooled")].groupby(["term", "name"]).list.agg(["size", lambda s: ",".join(sorted(s))]).reset_index()
    rec.columns = ["term", "name", "n_tf", "tfs"]; rec = rec.sort_values("n_tf", ascending=False); rec.to_csv(OUT / f"go_recurrence_{bg}.tsv", sep="\t", index=False)
    print(f"\n=== background: {bg} === TFs tested {len(tfl)}; TFs with >=1 term q<0.05: {sum(per.reindex(tfl).fillna(0) > 0)}")
    print(per.to_string())
    print("recurrent terms:"); print(rec.head(25)[["name", "n_tf", "tfs"]].to_string(index=False))
    for l in [x for x in R.list.unique() if x.startswith("pooled")]:
        print(l); print(R[(R.list == l) & (R.q < 0.05)].head(10)[["name", "k", "n_list", "fold", "q"]].round(4).to_string(index=False))
