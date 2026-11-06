# Gate 1 specificity test for one TF: split-half reproducibility of gene-level binding scores vs pseudo-TF nulls.
# Representation (per element, per half): raw obs/exp residual, imputed residual, blend, raw uncorrected (accessibility
# control), ATAC-only (identical halves; must fail). Gene score S = residual @ W, with W = power-law ABC (>= 0.02),
# promoter-only (self-promoter links), or ABC + Cicero (kappa = 1, the TF's own step-1 pairs > 25 kb, element -> promoter).
# Statistic = Spearman(S_A, S_B) across genes, mean over splits. Null = pseudo-TFs with the same cell count drawn from the
# pooled cells of the other TFs in the same library group (25 high vs 54 low in-element fraction).
import json, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import rankdata

G = "/work/users/d/y/dyy12/XuLab/regulatome/gate1"
X = "/work/users/d/y/dyy12/XuLab"
tf, n_split, n_null, n_null_split = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
seed0 = 20261028
t0 = time.time()

# ---- shared inputs ------------------------------------------------------------------------------------------------
E = pd.read_csv(f"{X}/regulatome/m0_universe/elements_v2.bed", sep="\t", header=None,
                names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"])
act = np.flatnonzero(E.cls.isin(["promoter", "distal_active", "bivalent"]).values)
Ea = E.iloc[act].reset_index(drop=True); nE = len(Ea)
atac = np.load(f"{X}/regulatome/m1_binding/v2/counts/atac_counts.npz")["atac"][act].astype(float)
logw = np.log(Ea.end.values - Ea.start.values)
coord = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(Ea.chrom, Ea.start, Ea.end))}
A = pd.read_csv(f"{X}/regulatome/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t",
                usecols=["chr", "start", "end", "TargetGene", "TargetGeneTSS", "isSelfPromoter", "ABC.Score"])
A["e"] = (A.chr + ":" + A.start.astype(str) + "-" + A.end.astype(str)).map(coord)
A = A.dropna(subset=["e"]); A["e"] = A.e.astype(int)
genes = np.array(sorted(A.TargetGene.unique())); gi = {g: i for i, g in enumerate(genes)}
A["g"] = A.TargetGene.map(gi)
W = sp.csr_matrix((A["ABC.Score"].values, (A.e.values, A.g.values)), shape=(nE, len(genes)))
P = A[A.isSelfPromoter.astype(bool)]
Wp = sp.csr_matrix((np.ones(len(P)), (P.e.values, P.g.values)), shape=(nE, len(genes)))

def cicero_W(t):
    """kappa=1 links: element overlapping one end of a TF Cicero pair (> 25 kb) -> gene whose TSS +/- 1 kb overlaps the other end"""
    f = f"{X}/cobinding/work/ourrun_k50/{t}/{t}.peak_edges.tsv"
    try: C = pd.read_csv(f, sep="\t")
    except FileNotFoundError: return None
    def parse(s):
        p = s.str.extract(r"^(\w+):(\d+)-(\d+)$"); return p[0].values, p[1].astype(int).values, p[2].astype(int).values
    c1, s1, e1 = parse(C.peak1); c2, s2, e2 = parse(C.peak2)
    keep = np.abs((s1 + e1) / 2 - (s2 + e2) / 2) > 25000
    tss = A.drop_duplicates("TargetGene")[["chr", "TargetGeneTSS", "g"]]
    tss_by = {c: (d.TargetGeneTSS.values.astype(int), d.g.values) for c, d in tss.groupby("chr")}
    el_by = {c: (d.start.values, d.end.values, d.index.values) for c, d in Ea.groupby("chrom", sort=False)}
    rows, cols = [], []
    for ca, sa, ea, cb, sb, eb in zip(c1[keep], s1[keep], e1[keep], c2[keep], s2[keep], e2[keep]):
        for (xc, xs, xe), (pc, ps, pe) in [((ca, sa, ea), (cb, sb, eb)), ((cb, sb, eb), (ca, sa, ea))]:
            if pc not in tss_by or xc not in el_by: continue
            t_, g_ = tss_by[pc]; gs = g_[(t_ >= ps - 1000) & (t_ <= pe + 1000)]
            if not len(gs): continue
            s_, e_, ix = el_by[xc]; es = ix[(s_ < xe) & (e_ > xs)]
            for e in es:
                for g in gs: rows.append(e); cols.append(g)
    if not rows: return None
    C = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(nE, len(genes))); C.data[:] = 1.0
    return C

# ---- cell matrices: focal TF + pool (other TFs, same library group) -------------------------------------------------
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]
F = pd.read_csv(f"{X}/regulatome/m1_binding/v2/frac_reads_by_class.tsv", sep="\t", index_col=0)
group = (F.all_elements > 0.22).map({True: "high", False: "low"})
def load(t):
    return (sp.load_npz(f"{G}/cellmats/{t}.raw.npz").tocsr(), sp.load_npz(f"{G}/cellmats/{t}.imp.npz").tocsr(),
            np.load(f"{G}/cellmats/{t}.cells.npz")["reads"].astype(float))
R_t, I_t, rd_t = load(tf)
pool_raw = np.zeros(nE); pool_reads = 0.0; pool_imp = np.zeros(nE); pool_cells = 0
PR, PI, PD = [], [], []
for t in tfs:
    if t == tf: continue
    r, i, d = load(t)
    pool_raw += np.asarray(r.sum(0)).ravel(); pool_reads += d.sum()
    pool_imp += np.asarray(i.sum(0)).ravel(); pool_cells += r.shape[0]
    if group[t] == group[tf]: PR.append(r); PI.append(i); PD.append(d)
PR, PI, PD = sp.vstack(PR).tocsr(), sp.vstack(PI).tocsr(), np.concatenate(PD)
pool_cpm = pool_raw / pool_reads * 1e6; pool_f = pool_imp / pool_cells
atac_cpm = atac / atac.sum() * 1e6
Xd = np.column_stack([np.ones(nE), np.log1p(atac_cpm), np.log1p(pool_cpm), logw])
n_cells = R_t.shape[0]
w_blend = n_cells / (n_cells + 10000.0)
Wc_t = cicero_W(tf)
arms = {"abc": W, "promoter": Wp}
if Wc_t is not None: arms["abc_cicero"] = W + Wc_t
print(f"{tf}: {n_cells} cells, group {group[tf]}, pool {PR.shape[0]} cells, genes {len(genes)}, cicero links "
      f"{0 if Wc_t is None else Wc_t.nnz}, load {time.time() - t0:.0f}s", flush=True)

def glm_mu(y, off):
    """Poisson GLM by IRLS: log mu = off + Xd b"""
    b = np.zeros(Xd.shape[1]); b[0] = np.log(y.sum() / np.exp(off).sum())
    for _ in range(25):
        eta = off + Xd @ b; mu = np.exp(eta); z = eta - off + (y - mu) / mu
        XtW = Xd.T * mu; bn = np.linalg.solve(XtW @ Xd + 1e-6 * np.eye(len(b)), XtW @ z)
        if np.max(np.abs(bn - b)) < 1e-6: b = bn; break
        b = bn
    return np.exp(off + Xd @ b)

def z(v): s = v.std(); return (v - v.mean()) / s if s > 0 else v * 0

def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])

def reproducibility(Rm, Im, rd, splits, rng):
    """mean split-half Spearman per representation x arm for one (pseudo-)TF"""
    y = np.asarray(Rm.sum(0)).ravel(); N = rd.sum()
    mu = glm_mu(y, np.full(nE, np.log(N / 1e6)))
    out = {}
    for _ in range(splits):
        h = rng.permutation(Rm.shape[0]) < Rm.shape[0] // 2
        reps = {}
        for name, m in (("A", h), ("B", ~h)):
            o = np.asarray(Rm[m].sum(0)).ravel(); muh = mu * rd[m].sum() / N
            raw = np.log2((o + 0.5) / (muh + 0.5))
            f = np.asarray(Im[m].sum(0)).ravel() / m.sum()
            imp = np.log2((f + 1e-3) / (pool_f + 1e-3))
            reps[name] = {"raw": raw, "imputed": imp, "blend": w_blend * z(raw) + (1 - w_blend) * z(imp),
                          "uncorrected": np.log2(o / rd[m].sum() * 1e6 + 1), "atac_only": np.log1p(atac_cpm)}
        for rep in reps["A"]:
            for arm, Wm in arms.items():
                if arm == "abc_cicero" and rep != "raw": continue
                sa, sb = Wm.T @ reps["A"][rep], Wm.T @ reps["B"][rep]
                out.setdefault(f"{rep}|{arm}", []).append(spearman(sa, sb))
    return {k: float(np.mean(v)) for k, v in out.items()}

rng = np.random.default_rng(seed0 + tfs.index(tf))
obs = reproducibility(R_t, I_t, rd_t, n_split, rng)
print("observed", {k: round(v, 3) for k, v in obs.items()}, f"{time.time() - t0:.0f}s", flush=True)
null = []
for k in range(n_null):
    cells = rng.choice(PR.shape[0], size=n_cells, replace=n_cells > PR.shape[0])
    null.append(reproducibility(PR[cells], PI[cells], PD[cells], n_null_split, rng))
    if k % 10 == 9: print(f"null {k + 1}/{n_null} {time.time() - t0:.0f}s", flush=True)
Nl = pd.DataFrame(null)
rows = []
for key, v in obs.items():
    nd = Nl[key].values
    rows.append(dict(tf=tf, rep=key.split("|")[0], arm=key.split("|")[1], n_cells=n_cells, group=group[tf], obs=v,
                     null_mean=nd.mean(), null_p90=np.percentile(nd, 90), null_max=nd.max(),
                     p=(1 + (nd >= v).sum()) / (1 + len(nd)), excess=v - np.median(nd)))
pd.DataFrame(rows).to_csv(f"{G}/results/{tf}.tsv", sep="\t", index=False, float_format="%.4f")
Nl.to_csv(f"{G}/results/{tf}.null.tsv", sep="\t", index=False, float_format="%.4f")
json.dump(dict(tf=tf, n_split=n_split, n_null=n_null, n_null_split=n_null_split, seed=seed0 + tfs.index(tf),
               secs=time.time() - t0), open(f"{G}/results/{tf}.json", "w"))
print(pd.DataFrame(rows).round(3).to_string(index=False), f"\ntotal {time.time() - t0:.0f}s")
