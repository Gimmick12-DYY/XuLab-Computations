# R2b specific-residual target lists, v5: identical to ../r2b_residual/r2b_residual.py (v4) except that the number of null draws (n_draws),
# the output folder and the manifest label are parameters: argv 1-3. Spec: SPEC.md (v4) plus README.md here.
# Per TF: gene residual X = log2((S_obs+.5)/(S_exp+.5))
# over power-law ABC links (kappa 0), expected reads from the Gate 1 v2.1 GLM, Poisson null (200 draws), BH per TF, split-half check; list = q <= 0.10
# and both halves > 0, top 1,000 by X. Writes per_tf/<TF>.tsv, r2b_summary.tsv, then manifest_v4.sha256 (outputs + script). No Perturb-seq input.
import glob, hashlib, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
X = "/work/users/d/y/dyy12/XuLab"; G = f"{X}/regulatome/gate1"; V = f"{G}/v2"; O = f"{X}/regulatome/r2b_residual"
NNULL = int(sys.argv[1]) if len(sys.argv) > 1 else 200  # n_draws: v4 used 200, v5 uses 10,000
O = sys.argv[2] if len(sys.argv) > 2 else O; MV = sys.argv[3] if len(sys.argv) > 3 else "v4"
EMIN, HMIN, QMAX, CAP = 5.0, 2.5, 0.10, 1000
t0 = time.time(); os.makedirs(f"{O}/per_tf", exist_ok=True)
assert not os.path.exists(f"{O}/manifest_{MV}.sha256"), f"manifest {MV} exists; outputs are frozen"
E = pd.read_csv(f"{X}/regulatome/m0_universe/elements_v2.bed", sep="\t", header=None, names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"])
act = np.flatnonzero(E.cls.isin(["promoter", "distal_active", "bivalent"]).values); Ea = E.iloc[act].reset_index(drop=True); nE = len(Ea)
gc = pd.read_csv(f"{V}/gc.tsv", sep="\t", header=None, index_col=0)[1].reindex(Ea.eid).values
mp = pd.read_csv(f"{V}/map.tsv", sep="\t", header=None, index_col=0)[1].reindex(Ea.eid).values
assert not np.isnan(gc).any() and not np.isnan(mp).any()
atac = np.load(f"{X}/regulatome/m1_binding/v2/counts/atac_counts.npz")["atac"][act].astype(float); atac_cpm = atac / atac.sum() * 1e6
logw = np.log(Ea.end.values - Ea.start.values)
coord = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(Ea.chrom, Ea.start, Ea.end))}
A = pd.read_csv(f"{X}/regulatome/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene", "ABC.Score"])
nlinks = len(A); A["e"] = (A.chr + ":" + A.start.astype(str) + "-" + A.end.astype(str)).map(coord); A = A.dropna(subset=["e"]); A["e"] = A.e.astype(int)
genes = np.array(sorted(A.TargetGene.unique())); A["g"] = A.TargetGene.map({g: i for i, g in enumerate(genes)})
W = sp.csr_matrix((A["ABC.Score"].values, (A.e.values, A.g.values)), shape=(nE, len(genes))); Wt = W.T.tocsr()
print(f"elements {nE}, ABC links {nlinks}, mapped to active elements {len(A)}, genes {len(genes)}", flush=True)
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]
# per-TF element sums and reads (Gate 1 v2.1 inputs), loaded one at a time
tf_sum, tf_reads = {}, {}
for t in tfs:
    tf_sum[t] = np.asarray(sp.load_npz(f"{G}/cellmats/{t}.raw.npz").sum(0)).ravel(); tf_reads[t] = np.load(f"{G}/cellmats/{t}.cells.npz")["reads"].astype(float).sum()
tot_sum = sum(tf_sum.values()); tot_reads = sum(tf_reads.values())
def design(pool_sum, pool_reads):  # from gate1/v2_1/gate1_v21.py
    pc = pool_sum / pool_reads * 1e6
    return np.column_stack([np.ones(nE), np.log1p(atac_cpm), np.log1p(pc), logw, gc, gc ** 2, mp])
def glm_mu(y, off, Xd):  # from gate1/v2_1/gate1_v21.py
    b = np.zeros(Xd.shape[1]); b[0] = np.log(y.sum() / np.exp(off).sum())
    for _ in range(30):
        eta = off + Xd @ b; mu = np.exp(eta); z = eta - off + (y - mu) / mu
        XtW = Xd.T * mu; bn = np.linalg.solve(XtW @ Xd + 1e-6 * np.eye(len(b)), XtW @ z)
        if np.max(np.abs(bn - b)) < 1e-7: b = bn; break
        b = bn
    return np.exp(off + Xd @ b)
def bh(p):
    n = len(p); o = np.argsort(p); q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]; return np.minimum(q, 1)
lx = lambda o, e: np.log2((o + 0.5) / (e + 0.5))
R2 = pd.read_csv(f"{X}/regulatome/r2_targets/r2_summary.tsv", sep="\t").set_index("tf"); rows = []
for t in tfs:
    Rm = sp.load_npz(f"{G}/cellmats/{t}.raw.npz").tocsr(); rd = np.load(f"{G}/cellmats/{t}.cells.npz")["reads"].astype(float)
    y = np.asarray(Rm.sum(0)).ravel(); N = rd.sum()
    mu = glm_mu(y, np.full(nE, np.log(N / 1e6)), design(tot_sum - tf_sum[t], tot_reads - tf_reads[t]))
    So, Se = Wt @ y, Wt @ mu; Xg = lx(So, Se); sc = Se >= EMIN
    rng = np.random.default_rng(20261029 + tfs.index(t)); h = rng.permutation(Rm.shape[0]) < Rm.shape[0] // 2  # Gate 1 v2.1 first split
    half = {}
    for nm, m in (("A", h), ("B", ~h)):
        o = np.asarray(Rm[m].sum(0)).ravel(); e = Wt @ (mu * rd[m].sum() / N); half[nm] = np.where(e >= HMIN, lx(Wt @ o, e), np.nan)
    rs = np.random.default_rng(7000 + tfs.index(t)); ge = np.zeros(sc.sum())
    for k in range(0, NNULL, 50):
        D = rs.poisson(mu[:, None], size=(nE, min(50, NNULL - k))).astype(float); ge += (lx(Wt @ D, Se[:, None])[sc] >= Xg[sc][:, None]).sum(1)
    p = np.full(len(genes), np.nan); q = np.full(len(genes), np.nan); p[sc] = (1 + ge) / (NNULL + 1); q[sc] = bh(p[sc])
    ok = sc & (q <= QMAX) & (half["A"] > 0) & (half["B"] > 0)
    T = pd.DataFrame(dict(gene=genes, S_obs=So, S_exp=Se, X=Xg, p=p, q=q, X_A=half["A"], X_B=half["B"], pass_=ok))[sc].rename(columns={"pass_": "pass"})
    T = T.sort_values(["pass", "X"], ascending=[False, False]); T["rank"] = np.where(T["pass"], np.arange(1, len(T) + 1), np.nan)
    T.to_csv(f"{O}/per_tf/{t}.tsv", sep="\t", index=False, float_format="%.5g")
    L = T[T["pass"]].head(CAP); lst = set(L.gene)
    d1 = set(pd.read_csv(f"{X}/regulatome/r2_targets/top1000/{t}.tsv", sep="\t").gene) if os.path.exists(f"{X}/regulatome/r2_targets/top1000/{t}.tsv") else set()
    P = pd.read_csv(f"{X}/regulatome/r2_targets/per_tf/{t}.tsv.gz", sep="\t", usecols=["gene", "S_atac"]); at = set(P.sort_values("S_atac", ascending=False).head(1000).gene)
    rows.append(dict(tf=t, tier=R2.tier.get(t, "."), n_scored=int(sc.sum()), n_pass=int(ok.sum()), n_list=len(L), median_X_list=float(L.X.median()) if len(L) else np.nan,
                     ovl_direct_top1000=len(lst & d1), ovl_atac_top1000=len(lst & at)))
    print(t, rows[-1], f"{time.time() - t0:.0f}s", flush=True)
pd.DataFrame(rows).to_csv(f"{O}/r2b_summary.tsv", sep="\t", index=False, float_format="%.4g")
fs = sorted(glob.glob(f"{O}/per_tf/*.tsv")) + [f"{O}/r2b_summary.tsv", os.path.abspath(__file__), f"{X}/regulatome/r2b_residual/SPEC.md"] + ([f"{O}/README.md"] if os.path.exists(f"{O}/README.md") else [])
open(f"{O}/manifest_{MV}.sha256", "w").write("".join(f"{hashlib.sha256(open(f, 'rb').read()).hexdigest()}  {os.path.relpath(f, X)}\n" for f in fs))
print(f"manifest {MV}", hashlib.sha256(open(f"{O}/manifest_{MV}.sha256", "rb").read()).hexdigest(), f"total {time.time() - t0:.0f}s")
