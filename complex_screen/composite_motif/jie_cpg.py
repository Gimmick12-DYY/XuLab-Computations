# CpG evidence for EZH2-MTF2 (MTF2's EH domain binds unmethylated CpG; Li et al. 2017). Union-ATAC elements with each TF's Gate 1 residual: co-bound = Btilde > 0 for both; single = one only.
# Background: elements with neither, matched to the co-bound set on GC x ATAC strata. Stats: CpG o/e, GCG per kb, CpG-island-like fraction (GC >= 0.5, CpG o/e >= 0.6). Control: the same enrichment for 40 non-partner pairs of similar depth, same library group.
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/complex_screen/composite_motif")
from cm_common import *
OUT = f"{X}/complex_screen/jie_motif"; E = load_elements(); C = pd.read_csv(f"{U}/covariates.tsv", sep="\t"); n = len(E)
sq = {}; nm = None
for l in open(f"{U}/union.fa"):
    l = l.strip()
    if l.startswith(">"): nm = l[1:].split("::")[0]; sq[nm] = []
    else: sq[nm].append(l.upper())
S = ["".join(sq[e]) for e in E.eid]; L = np.array([len(s) for s in S], float); cg = np.array([s.count("CG") for s in S], float); c_ = np.array([s.count("C") for s in S], float); g_ = np.array([s.count("G") for s in S], float); gcg = np.array([s.count("GCG") + s.count("CGC") for s in S], float)
oe = cg * L / np.maximum(c_ * g_, 1); gc = (c_ + g_) / L; dens = gcg / L * 1000; cgi = ((gc >= 0.5) & (oe >= 0.6)).astype(float)
np.savez_compressed(f"{OUT}/cpg_stats.npz", oe=oe, gc=gc, gcg_per_kb=dens, cgi=cgi)
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]; Yz = {t: np.load(f"{U}/counts/{t}.npz") for t in tfs}; tot = sum(Yz[t]["y"] for t in tfs); NN = sum(int(Yz[t]["total"]) for t in tfs); B = {}
def bp(t):
    if t not in B:
        y = Yz[t]["y"].astype(float); R = int(Yz[t]["total"]); mu = residual(y, R, tot - y, NN - R); B[t] = np.log2((y + .5) / (mu + .5)) > 0
    return B[t]
stra = np.minimum((pd.Series(gc).rank(pct=True).values * 20).astype(int), 19) * 10 + np.minimum((pd.Series(C.atac.values).rank(pct=True).values * 10).astype(int), 9)
def matched(fg, pool, rng):
    ix = []
    for k, m in zip(*np.unique(stra[fg], return_counts=True)):
        p = np.flatnonzero(pool & (stra == k))
        if len(p): ix.append(rng.choice(p, m, replace=len(p) < m))
    return np.concatenate(ix)
def test(a, b, boot=500, seed=1):
    r = np.random.default_rng(seed); A, Bb = bp(a), bp(b); co = A & Bb; si = A ^ Bb; no = ~A & ~Bb; d = dict(a=a, b=b, n_co=int(co.sum()), n_single=int(si.sum()), n_none=int(no.sum())); co_i = np.flatnonzero(co); ids = matched(co, no, r); res = {}
    for nm_, v in (("cpg_oe", oe), ("gcg_per_kb", dens), ("cgi_frac", cgi)):
        res[nm_] = (v[co].mean(), v[si].mean(), v[ids].mean()); d.update({f"{nm_}_co": round(v[co].mean(), 4), f"{nm_}_single": round(v[si].mean(), 4), f"{nm_}_gcmatched": round(v[ids].mean(), 4), f"{nm_}_ratio": round(v[co].mean() / v[ids].mean(), 4)})
        if boot:
            b_ = [v[r.choice(co_i, len(co_i))].mean() / v[r.choice(ids, len(ids))].mean() for _ in range(boot)]; d[f"{nm_}_lo"], d[f"{nm_}_hi"] = round(np.percentile(b_, 2.5), 4), round(np.percentile(b_, 97.5), 4)
    return d
F = pd.read_csv(f"{X}/regulatome/m1_binding/v2/frac_reads_by_class.tsv", sep="\t", index_col=0); grp = (F.all_elements > 0.22).to_dict(); rd = pd.Series({t: int(Yz[t]["total"]) for t in tfs}); rk = rd.rank()
R0 = [test("EZH2", "MTF2"), test("MLX", "SREBF2")]; ctl = []; r = np.random.default_rng(9); P = pd.read_csv(f"{X}/regulatome/r1_tftf/v2_final/r1_pairs_frozen.tsv", sep="\t", keep_default_na=False, na_values=[""]); edge = set(map(tuple, P[P["class"] != "."][["a", "b"]].values)) | set((b, a) for a, b in P[P["class"] != "."][["a", "b"]].values)
pool = lambda t: [x for x in tfs if abs(rk[x] - rk[t]) <= 8 and grp[x] == grp[t] and x != t]
for pr in (("EZH2", "MTF2"), ("MLX", "SREBF2")):
    k = 0
    while k < 40:
        a, b = r.choice(pool(pr[0])), r.choice(pool(pr[1]))
        if a == b or (a, b) in edge: continue
        ctl.append(dict(test(a, b, boot=0), control_for="-".join(pr))); k += 1
Ct = pd.DataFrame(ctl); pd.DataFrame(R0).to_csv(f"{OUT}/cpg_pair_results.tsv", sep="\t", index=False); Ct.to_csv(f"{OUT}/cpg_control_pairs.tsv", sep="\t", index=False); pd.set_option("display.width", 200)
print(pd.DataFrame(R0).T.to_string())
for r0 in R0:
    k = f"{r0['a']}-{r0['b']}"; c = Ct[Ct.control_for == k]; print(k, {m: dict(obs=r0[f"{m}_ratio"], ctl_median=round(c[f"{m}_ratio"].median(), 3), ctl_p5=round(c[f"{m}_ratio"].quantile(.05), 3), ctl_p95=round(c[f"{m}_ratio"].quantile(.95), 3), rank_pct=round((c[f"{m}_ratio"] < r0[f"{m}_ratio"]).mean(), 3)) for m in ("cpg_oe", "gcg_per_kb", "cgi_frac")})
