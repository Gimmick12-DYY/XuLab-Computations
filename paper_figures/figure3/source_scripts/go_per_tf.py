"""GO steps 2-3 (user 2026-11-17): GO BP of TF-specific vs conserved module genes, per TF and summarised across TFs.
Module gene of a TF: protein-coding gene whose TSS +-1 kb overlaps an element of one of that TF's higher-order coordinated binding modules
(>=3 elements, all pairs linked; nodes_annotated.tsv.gz clique == 1, as panel j). n_tfs(gene) = number of TFs whose modules contain it.
Split (user 2026-11-18, strict split at 3, no buffer): conserved = n_tfs >= 3; TF-specific = n_tfs 1-2. Every module gene is in one group.
Sensitivity: TF-specific = n_tfs == 1 only (2-TF genes left out).
Step 2: per TF, its TF-specific genes and its conserved genes, each tested separately. Step 3: per group, terms ranked by the number of TFs
with q < 0.05 (ties: pooled q), plus the pooled lists (all TF-specific genes; all conserved genes).
Test (as panel j / go_cliques.py): goa_human.gaf + go-basic.obo, BP, propagated to ancestors, NOT dropped; background = 12,388 network
promoter genes; terms of 10-500 background genes; one-sided hypergeometric; BH within each list. Lists with < 20 annotated genes: n/a.
Near-duplicate terms merged for the top lists: a term is dropped if its background gene set has Jaccard > 0.6 with a better-ranked term.
Writes per_TF_specific.tsv, per_TF_conserved.tsv, summary_specific.tsv, summary_conserved.tsv, sensitivity.tsv, pooled_*.tsv, tested_lists.tsv.
  python scripts/go_per_tf.py <outdir>   (Longleaf, hic env)"""
import gzip, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import hypergeom
X = Path("/work/users/d/y/dyy12/XuLab"); TG = X / "cobinding/tf_groups"; D = X / "data/go"; OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
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
N = pd.read_csv(TG / "results_location/nodes_annotated.tsv.gz", sep="\t", keep_default_na=False)
P = N[N.gene != ""].assign(gene=lambda d: d.gene.str.split(",")).explode("gene")
P = P[~P.gene.str.startswith("ENSG")].groupby(["tf", "gene"]).clique.max().reset_index(); universe = P.gene.unique()
bg = [g for g in set(universe) if g in g2t]; BG = set(bg); Nbg = len(bg); t2g = defaultdict(set)
for g in bg:
    for t in g2t[g]: t2g[t].add(g)
TERMS = [t for t, s in t2g.items() if 10 <= len(s) <= 500]; K = np.array([len(t2g[t]) for t in TERMS])
def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); r = p[o] * len(p) / np.arange(1, len(p) + 1); q = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out
def enrich(lst):
    L = set(lst) & BG; n = len(L)
    k = np.array([len(t2g[t] & L) for t in TERMS]); p = hypergeom.sf(k - 1, Nbg, K, n)
    return pd.DataFrame({"term": TERMS, "name": [name.get(t, "") for t in TERMS], "k": k, "n_list": n, "K": K, "fold": np.where(k > 0, (k / max(n, 1)) / (K / Nbg), 0), "p": p, "q": bh(p)}).set_index("term")
def dedup(order):
    keep = []
    for t in order:
        if all(len(t2g[t] & t2g[u]) / len(t2g[t] | t2g[u]) <= 0.6 for u in keep): keep.append(t)
    return keep
MG = P[P.clique == 1]; ntf = MG.groupby("gene").tf.nunique()
def run(spec_set, tag):
    groups = {"specific": spec_set, "conserved": set(ntf.index[ntf >= 3])}
    per, tested, pooled = {}, [], {}
    for grp, gs in groups.items():
        rows = []
        for tf, d in MG.groupby("tf"):
            lst = sorted(set(d.gene) & gs); na = len(set(lst) & BG)
            tested.append(dict(set=tag, group=grp, tf=tf, genes=len(lst), annotated=na, tested=na >= 20))
            if na < 20: continue
            R = enrich(lst); R["tf"] = tf; rows.append(R.reset_index())
        per[grp] = pd.concat(rows) if rows else pd.DataFrame()
        pooled[grp] = enrich(sorted(gs))
    S = {}
    for grp in groups:
        A = per[grp]; sig = A[A.q < 0.05]
        s = sig.groupby("term").agg(n_tfs_sig=("tf", "size"), median_fold=("fold", "median"), tfs=("tf", lambda x: ",".join(sorted(x))))
        pq = pooled[grp]; s = pq[["name", "k", "n_list", "fold", "q"]].rename(columns={"k": "pooled_genes", "n_list": "pooled_n", "fold": "pooled_fold", "q": "pooled_q"}).join(s, how="left")
        s["n_tfs_sig"] = s.n_tfs_sig.fillna(0).astype(int); s["n_tfs_tested"] = int(A.tf.nunique()) if len(A) else 0
        s = s.sort_values(["n_tfs_sig", "pooled_q"], ascending=[False, True]); s["top10"] = s.index.isin(dedup([t for t in s.index if s.n_tfs_sig[t] > 0 or s.pooled_q[t] < 0.05])[:10])
        S[grp] = s
    return per, S, pooled, pd.DataFrame(tested)
per, S, pooled, tested = run(set(ntf.index[ntf <= 2]), "main (specific = 1-2 TFs)")
per2, S2, pooled2, tested2 = run(set(ntf.index[ntf == 1]), "sensitivity (specific = 1 TF)")
top = lambda s, k=10: s[s.top10].head(k)
for grp in ("specific", "conserved"):
    cols = ["tf", "term", "name", "k", "n_list", "fold", "q"]
    A = per[grp]; A[A.q < 0.05].sort_values(["tf", "q"])[cols].to_csv(OUT / f"per_TF_{grp}.tsv", sep="\t", index=False)
    S[grp].to_csv(OUT / f"summary_{grp}.tsv", sep="\t"); pooled[grp].sort_values("p").to_csv(OUT / f"pooled_{grp}.tsv", sep="\t")
t1, t2 = top(S["specific"]), top(S2["specific"])
sens = pd.DataFrame({"main_specific_top10": list(t1.name) + [""] * (10 - len(t1)), "sensitivity_specific_top10": list(t2.name) + [""] * (10 - len(t2)),
                     "main_conserved_top10": list(top(S["conserved"]).name) + [""] * (10 - len(top(S["conserved"]))), "sensitivity_conserved_top10": list(top(S2["conserved"]).name) + [""] * (10 - len(top(S2["conserved"])))})
sens.to_csv(OUT / "sensitivity.tsv", sep="\t", index=False); S2["specific"].to_csv(OUT / "summary_specific_sensitivity.tsv", sep="\t")
pd.concat([tested, tested2]).to_csv(OUT / "tested_lists.tsv", sep="\t", index=False)
for grp in ("specific", "conserved"):
    print(f"== {grp}: TFs tested {S[grp].n_tfs_tested.iloc[0]}; pooled genes {pooled[grp].n_list.iloc[0]}; pooled q<0.05 terms {(pooled[grp].q < 0.05).sum()}; terms sig in >=1 TF {(S[grp].n_tfs_sig > 0).sum()}")
    print(top(S[grp])[["name", "n_tfs_sig", "median_fold", "pooled_fold", "pooled_q"]].round(4).to_string())
print("== sensitivity"); print(sens.to_string(index=False))
print("specific top10 overlap main vs sensitivity:", len(set(t1.index) & set(t2.index)), "of", len(t1))
