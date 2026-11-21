"""Leave-one-TF-out robustness of the pooled GO result (user 2026-11-18). Pooled lists as panel j draft: conserved = genes in modules of
>=3 TFs, TF-specific = genes in modules of 1-2 TFs (strict split at 3; union across TFs, each gene once). For each TF with modules, that TF is removed
from the panel, TFs per gene are recounted over the remaining TFs, both pooled lists rebuilt and tested again (same GO files, background
12,388 network promoter genes, terms 10-500, one-sided hypergeometric, BH). Recorded: fold/q of the 5 conserved terms of the draft, and
the number of terms at q < 0.05 in each list.
Writes loo_conserved_terms.tsv, loo_summary.tsv, loo_term_keep.tsv, go_pooled_robustness.png.
  python scripts/go_pooled_loo.py <outdir>   (Longleaf, hic env)"""
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
MG = P[P.clique == 1]
FOCUS = {"GO:1903047": "Mitotic cell cycle process", "GO:0051052": "Regulation of DNA metabolic process", "GO:0006282": "Regulation of DNA repair",
         "GO:0080135": "Regulation of response to stress", "GO:1904263": "Pos. reg. of TORC1 signaling"}
rows, lrows = [], []
for drop in ["(none)"] + sorted(MG.tf.unique()):
    M = MG[MG.tf != drop]; ntf = M.groupby("gene").tf.nunique()
    cons, spec = sorted(ntf.index[ntf >= 3]), sorted(ntf.index[ntf <= 2])
    Rc, Rs = enrich(cons), enrich(spec)
    lrows.append(dict(left_out=drop, conserved_genes=len(cons), conserved_annotated=int(Rc.n_list.iloc[0]), conserved_terms_q005=int((Rc.q < 0.05).sum()),
                      specific_genes=len(spec), specific_annotated=int(Rs.n_list.iloc[0]), specific_terms_q005=int((Rs.q < 0.05).sum()),
                      **{f"{FOCUS[t]} q": float(Rc.q[t]) for t in FOCUS}))
    for t in FOCUS: rows.append(dict(left_out=drop, term=t, name=FOCUS[t], k=int(Rc.k[t]), fold=float(Rc.fold[t]), q=float(Rc.q[t]), kept=bool(Rc.q[t] < 0.05)))
S = pd.DataFrame(lrows); R = pd.DataFrame(rows)
S.to_csv(OUT / "loo_summary.tsv", sep="\t", index=False); R.to_csv(OUT / "loo_conserved_terms.tsv", sep="\t", index=False)
L1 = R[R.left_out != "(none)"]
K = L1.groupby("name").agg(leave_outs=("kept", "size"), still_q005=("kept", "sum"), killed_by=("left_out", lambda s: ",".join(s[~L1.loc[s.index, "kept"]])),
                           q_min=("q", "min"), q_max=("q", "max")).reindex(list(FOCUS.values()))
K.to_csv(OUT / "loo_term_keep.tsv", sep="\t")
print(S[S.left_out.isin(["(none)", "SOX4"])].T.to_string()); print(K.to_string())
print("specific terms q<0.05 across leave-outs: max", S.specific_terms_q005.max())
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, glob as _g
from matplotlib import font_manager as fm
for f in _g.glob("/work/users/d/y/dyy12/fonts/arial/*.ttf"): fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 6, "svg.fonttype": "none"})
tfs = [d for d in S.left_out if d != "(none)"]; MM = 1 / 25.4
fig, ax = plt.subplots(figsize=(183 * MM, 46 * MM)); fig.subplots_adjust(left=0.23, right=0.99, bottom=0.36, top=0.9)
for i, t in enumerate(FOCUS):
    for j, d in enumerate(tfs):
        r = R[(R.left_out == d) & (R.term == t)].iloc[0]
        ax.scatter(j, i, s=12, color="#2C7A6B" if r.kept else "white", edgecolors="#2C7A6B" if r.kept else "#9E9E9E", linewidths=0.5)
ax.set_yticks(range(len(FOCUS))); ax.set_yticklabels(list(FOCUS.values())); ax.set_ylim(len(FOCUS) - 0.5, -0.5)
ax.set_xticks(range(len(tfs))); ax.set_xticklabels(tfs, rotation=90, fontsize=4.5); ax.set_xlim(-0.7, len(tfs) - 0.3)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.set_xlabel("TF left out (pooled conserved list rebuilt without it)"); ax.set_title("Filled = term still q < 0.05 after leaving the TF out", fontsize=6, loc="left")
fig.savefig(OUT / "go_pooled_robustness.png", dpi=300); fig.savefig(OUT / "go_pooled_robustness.svg")
