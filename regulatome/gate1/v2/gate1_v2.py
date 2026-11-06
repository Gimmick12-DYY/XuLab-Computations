# Gate 1 v2 (batch-aware), raw reads only. Unit = one TF, or a module (members' cells pooled), optionally minus members.
# Residual model: Poisson GLM, log mu = log(reads) + b0 + b1 log1p(ATAC CPM) + b2 log1p(leave-unit-out pooled panel CPM)
#   + b3 log(width) + b4 GC + b5 GC^2 + b6 k50 mappable fraction.
# Null: each pseudo-unit draws, for each member, n cells from ONE other TF's library (a donor in the member's library group,
#   not in the unit), so the null carries a real library signature; the pseudo-unit's pool excludes its donors.
# Depth: a member with more cells than the 3rd-largest eligible donor is downsampled to that size (n_eff) for both the
#   unit and its null, so the null never needs sampling with replacement.
# Arms: raw x ABC (power-law), raw promoter-only, raw x (ABC + Cicero, kappa=1; single TFs with Cicero pairs), ATAC-only.
# Also saves the raw x ABC half-profiles (gene scores) per split for the cross-TF specificity margin.
#   usage: gate1_v2.py tf <TF> | module <MODULE> [EXCL1,EXCL2]
import json, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import rankdata

G = "/work/users/d/y/dyy12/XuLab/regulatome/gate1"
V = f"{G}/v2"
X = "/work/users/d/y/dyy12/XuLab"
N_SPLIT, N_NULL, N_NULL_SPLIT = 20, 100, 5
mode, name = sys.argv[1], sys.argv[2]
excl = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else []
t0 = time.time()

E = pd.read_csv(f"{X}/regulatome/m0_universe/elements_v2.bed", sep="\t", header=None,
                names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"])
act = np.flatnonzero(E.cls.isin(["promoter", "distal_active", "bivalent"]).values)
Ea = E.iloc[act].reset_index(drop=True); nE = len(Ea)
gc = pd.read_csv(f"{V}/gc.tsv", sep="\t", header=None, index_col=0)[1].reindex(Ea.eid).values
mp = pd.read_csv(f"{V}/map.tsv", sep="\t", header=None, index_col=0)[1].reindex(Ea.eid).values
assert not np.isnan(gc).any() and not np.isnan(mp).any()
atac = np.load(f"{X}/regulatome/m1_binding/v2/counts/atac_counts.npz")["atac"][act].astype(float)
atac_cpm = atac / atac.sum() * 1e6
logw = np.log(Ea.end.values - Ea.start.values)
coord = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(Ea.chrom, Ea.start, Ea.end))}
A = pd.read_csv(f"{X}/regulatome/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t",
                usecols=["chr", "start", "end", "TargetGene", "TargetGeneTSS", "isSelfPromoter", "ABC.Score"])
A["e"] = (A.chr + ":" + A.start.astype(str) + "-" + A.end.astype(str)).map(coord)
A = A.dropna(subset=["e"]); A["e"] = A.e.astype(int)
genes = np.array(sorted(A.TargetGene.unique())); A["g"] = A.TargetGene.map({g: i for i, g in enumerate(genes)})
W = sp.csr_matrix((A["ABC.Score"].values, (A.e.values, A.g.values)), shape=(nE, len(genes)))
P = A[A.isSelfPromoter.astype(bool)]
Wp = sp.csr_matrix((np.ones(len(P)), (P.e.values, P.g.values)), shape=(nE, len(genes)))

def cicero_W(t):
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
            s_, e_, ix = el_by[xc]
            for e in ix[(s_ < xe) & (e_ > xs)]:
                for g in gs: rows.append(e); cols.append(g)
    if not rows: return None
    C = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(nE, len(genes))); C.data[:] = 1.0
    return C

tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]
F = pd.read_csv(f"{X}/regulatome/m1_binding/v2/frac_reads_by_class.tsv", sep="\t", index_col=0)
group = (F.all_elements > 0.22).map({True: "high", False: "low"})
if mode == "tf":
    members = [name]
else:
    Mt = pd.read_csv(f"{X}/regulatome/r1_tftf/r1_modules_provisional.tsv", sep="\t").set_index("module")
    members = Mt.loc[name, "members"].split(",")
    if str(Mt.loc[name, "extended_by_cobinding"]) not in (".", "nan"): members += Mt.loc[name, "extended_by_cobinding"].split(",")
members = [m for m in members if m not in excl]
label = name + (f"__minus_{'_'.join(excl)}" if excl else "")

raw = {t: sp.load_npz(f"{G}/cellmats/{t}.raw.npz").tocsr() for t in tfs}
reads = {t: np.load(f"{G}/cellmats/{t}.cells.npz")["reads"].astype(float) for t in tfs}
tf_sum = {t: np.asarray(raw[t].sum(0)).ravel() for t in tfs}; tf_reads = {t: reads[t].sum() for t in tfs}
tot_sum = sum(tf_sum.values()); tot_reads = sum(tf_reads.values())
ncell = {t: raw[t].shape[0] for t in tfs}

# eligible donors and effective size per member
plan = {}
for m in members:
    donors = sorted([d for d in tfs if d not in members and group[d] == group[m]], key=lambda d: -ncell[d])
    cap = ncell[donors[2]] if len(donors) >= 3 else ncell[donors[-1]]
    n_eff = min(ncell[m], cap)
    plan[m] = dict(n_eff=n_eff, donors=[d for d in donors if ncell[d] >= n_eff])
arms = {"abc": W, "promoter": Wp}
if mode == "tf":
    Wc = cicero_W(members[0])
    if Wc is not None: arms["abc_cicero"] = W + Wc
neff_txt = ", ".join(f"{m}: {plan[m]['n_eff']}/{ncell[m]}" for m in members)
print(f"{label}: members {members}, n_eff {{{neff_txt}}}, donors/member {[len(plan[m]['donors']) for m in members]}, "
      f"load {time.time() - t0:.0f}s", flush=True)

def design(pool_sum, pool_reads):
    pc = pool_sum / pool_reads * 1e6
    return np.column_stack([np.ones(nE), np.log1p(atac_cpm), np.log1p(pc), logw, gc, gc ** 2, mp])

def glm_mu(y, off, Xd):
    b = np.zeros(Xd.shape[1]); b[0] = np.log(y.sum() / np.exp(off).sum())
    for _ in range(30):
        eta = off + Xd @ b; mu = np.exp(eta); z = eta - off + (y - mu) / mu
        XtW = Xd.T * mu; bn = np.linalg.solve(XtW @ Xd + 1e-6 * np.eye(len(b)), XtW @ z)
        if np.max(np.abs(bn - b)) < 1e-7: b = bn; break
        b = bn
    return np.exp(off + Xd @ b)

def spearman(a, b): return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])

def unit_stats(parts, excluded, splits, rng, keep_profiles=False):
    """parts: list of (csr cells x elements, reads per cell) already at effective size; excluded: TFs left out of the pool"""
    Rm = sp.vstack([p[0] for p in parts]).tocsr(); rd = np.concatenate([p[1] for p in parts])
    pool_sum = tot_sum - sum(tf_sum[t] for t in excluded); pool_reads = tot_reads - sum(tf_reads[t] for t in excluded)
    Xd = design(pool_sum, pool_reads)
    y = np.asarray(Rm.sum(0)).ravel(); N = rd.sum()
    mu = glm_mu(y, np.full(nE, np.log(N / 1e6)), Xd)
    out, prof = {}, []
    for _ in range(splits):
        h = rng.permutation(Rm.shape[0]) < Rm.shape[0] // 2
        res = {}
        for nm, msk in (("A", h), ("B", ~h)):
            o = np.asarray(Rm[msk].sum(0)).ravel(); muh = mu * rd[msk].sum() / N
            res[nm] = {"raw": np.log2((o + 0.5) / (muh + 0.5)), "atac_only": np.log1p(atac_cpm)}
        for rep in ("raw", "atac_only"):
            for arm, Wm in arms.items():
                if rep == "atac_only" and arm == "abc_cicero": continue
                sa, sb = Wm.T @ res["A"][rep], Wm.T @ res["B"][rep]
                out.setdefault(f"{rep}|{arm}", []).append(spearman(sa, sb))
                if keep_profiles and rep == "raw" and arm == "abc": prof.append((sa.astype(np.float32), sb.astype(np.float32)))
    return {k: float(np.mean(v)) for k, v in out.items()}, prof

def draw(rng, src, n):
    c = rng.choice(raw[src].shape[0], size=n, replace=False); return raw[src][c], reads[src][c]

seed = 20261029 + (tfs.index(members[0]) if mode == "tf" else 500 + sum(map(ord, label)) % 1000)
rng = np.random.default_rng(seed)
# observed: members at n_eff; members above n_eff get a fresh subsample per split (split loop below handles one draw;
# for downsampled units the statistic is averaged over 4 independent subsamples x 5 splits)
down = any(plan[m]["n_eff"] < ncell[m] for m in members)
obs_runs, profs = [], []
for rep in range(4 if down else 1):
    parts = [draw(rng, m, plan[m]["n_eff"]) for m in members]
    s, pr = unit_stats(parts, members, N_SPLIT // (4 if down else 1), rng, keep_profiles=(mode == "tf"))
    obs_runs.append(s); profs += pr
obs = {k: float(np.mean([o[k] for o in obs_runs])) for k in obs_runs[0]}
print("observed", {k: round(v, 3) for k, v in obs.items()}, f"{time.time() - t0:.0f}s", flush=True)
null, donors_used = [], []
for k in range(N_NULL):
    ds = [plan[m]["donors"][rng.integers(len(plan[m]["donors"]))] for m in members]
    parts = [draw(rng, d, plan[m]["n_eff"]) for d, m in zip(ds, members)]
    s, _ = unit_stats(parts, sorted(set(ds)), N_NULL_SPLIT, rng)
    null.append(s); donors_used.append(",".join(ds))
    if k % 20 == 19: print(f"null {k + 1}/{N_NULL} {time.time() - t0:.0f}s", flush=True)
Nl = pd.DataFrame(null); Nl["donors"] = donors_used
rows = []
for key, v in obs.items():
    nd = Nl[key].values
    rows.append(dict(unit=label, mode=mode, members=",".join(members), rep=key.split("|")[0], arm=key.split("|")[1],
                     n_cells=sum(ncell[m] for m in members), n_eff=sum(plan[m]["n_eff"] for m in members),
                     groups=",".join(sorted(set(group[m] for m in members))), obs=v, null_mean=nd.mean(),
                     null_p90=np.percentile(nd, 90), null_max=nd.max(), p=(1 + (nd >= v).sum()) / (1 + len(nd)),
                     excess=v - np.median(nd)))
outdir = f"{V}/results" if mode == "tf" else f"{V}/results_modules"
pd.DataFrame(rows).to_csv(f"{outdir}/{label}.tsv", sep="\t", index=False, float_format="%.4f")
Nl.to_csv(f"{outdir}/{label}.null.tsv", sep="\t", index=False, float_format="%.4f")
if profs:
    np.savez_compressed(f"{V}/results/{label}.profiles.npz", A=np.stack([p[0] for p in profs]), B=np.stack([p[1] for p in profs]))
json.dump(dict(unit=label, members=members, plan=plan, seed=seed, secs=time.time() - t0), open(f"{outdir}/{label}.json", "w"))
print(pd.DataFrame(rows).round(3).to_string(index=False), f"\ntotal {time.time() - t0:.0f}s")
