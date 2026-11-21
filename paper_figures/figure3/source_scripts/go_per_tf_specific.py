"""GO per TF on TF-specific module genes (user 2026-11-18): for each TF, its module genes that sit in modules of 1-2 TFs in total (strict
split at 3). Same GO files, background (12,388 network promoter genes), terms 10-500, one-sided hypergeometric, BH within each TF's list.
TFs with >= 20 annotated genes tested. Writes per TF the top 3 terms by p (with q) and the number of q < 0.05 terms, plus a paralog-cluster
check of the hit genes (OR*, PCDH*, KRTAP*, ZNF*, HIST*/H2*, UGT*, TAS2R*) for every listed term.
  python scripts/go_per_tf_specific.py <outdir>   (Longleaf, hic env)"""
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
import re as _re
MG = P[P.clique == 1]; ntf = MG.groupby("gene").tf.nunique(); spec = set(ntf.index[ntf <= 2])
CL = [("OR", r"^OR\d"), ("PCDH", r"^PCDH"), ("KRTAP", r"^KRTAP"), ("ZNF", r"^ZNF"), ("histone", r"^(HIST|H2A|H2B|H3C|H4C|H1-)"), ("UGT", r"^UGT"), ("TAS2R", r"^TAS2R")]
rows, tested = [], []
for tf, d in MG.groupby("tf"):
    lst = sorted(set(d.gene) & spec); L = set(lst) & BG; na = len(L)
    tested.append(dict(tf=tf, specific_genes=len(lst), annotated=na, tested=na >= 20))
    if na < 20: continue
    R = enrich(lst).sort_values("p"); nsig = int((R.q < 0.05).sum())
    for rank, (t, r) in enumerate(R.head(3).iterrows(), 1):
        hits = sorted(t2g[t] & L); cc = {c: sum(bool(_re.match(p, g)) for g in hits) for c, p in CL}; top_c = max(cc, key=cc.get)
        rows.append(dict(tf=tf, annotated=na, n_terms_q005=nsig, rank=rank, term=t, name=r["name"], genes=int(r.k), fold=float(r.fold), p=float(r.p), q=float(r.q),
                         top_cluster=top_c if cc[top_c] else "", cluster_frac=(cc[top_c] / len(hits)) if hits else 0.0, cluster_artifact=bool(hits) and cc[top_c] / len(hits) > 0.5,
                         hit_genes=",".join(hits[:40])))
T = pd.DataFrame(rows); D = pd.DataFrame(tested).sort_values("annotated", ascending=False)
T.to_csv(OUT / "per_TF_top3.tsv", sep="\t", index=False); D.to_csv(OUT / "tested.tsv", sep="\t", index=False)
print("TFs tested", int(D.tested.sum()), "of", len(D), "; TFs with >=1 term q<0.05:", int((T.groupby("tf").n_terms_q005.first() > 0).sum()))
print("min q over all TFs:", T.q.min()); print(T[T["rank"] == 1].sort_values("q").head(10)[["tf", "annotated", "name", "genes", "fold", "p", "q", "top_cluster", "cluster_frac"]].round(4).to_string(index=False))
