# Stage B graded empirical-Bayes Regulome (R3, manifest v6). Spec: SPEC.md here. usage: r3_eb.py fit | validate [--test TF1,TF2]
#  fit:      Gate 1 v2.1 Poisson GLM refit once with log1p(h_e) (HOT count, leave-TF-out); NB phi by moments; b, s; ashr per TF and module;
#            genes (S_B, S_ABC, list rule); writes elements/, genes/, modules/, tf_summary.tsv (depth label filled by validate).
#  validate: binomial thinning (10/25/50/75%) for TFs with >= 20,000 reads and depth labels; split-half PPC; AUPR vs bulk ChIP (7 TFs);
#            HPA HEK293 nTPM Spearman; HOT stratification; ATAC-only proximal baseline; then manifest_v6.sha256 (written last).
# No Perturb-seq input anywhere.
import glob, hashlib, json, os, shutil, subprocess, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import spearmanr
X = "/work/users/d/y/dyy12/XuLab"; G = f"{X}/regulatome/gate1"; V = f"{G}/v2"; R3 = f"{X}/regulatome/r3_eb"
mode = sys.argv[1]; TEST = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[2] == "--test" else None
O = f"{R3}/tmp/test" if TEST else R3
assert not os.path.exists(f"{R3}/manifest_v6.sha256"), "manifest v6 exists; R3 outputs are frozen"
CORES = int(os.environ.get("SLURM_CPUS_PER_TASK", "1")); t0 = time.time()
def log(*a): print(*a, f"{time.time() - t0:.0f}s", flush=True)
# ---------- inputs (same element set and covariates as Gate 1 v2.1 / R2b) ----------
E = pd.read_csv(f"{X}/regulatome/m0_universe/elements_v2.bed", sep="\t", header=None, names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"])
act = np.flatnonzero(E.cls.isin(["promoter", "distal_active", "bivalent"]).values); Ea = E.iloc[act].reset_index(drop=True); nE = len(Ea)
gc = pd.read_csv(f"{V}/gc.tsv", sep="\t", header=None, index_col=0)[1].reindex(Ea.eid).values; mp = pd.read_csv(f"{V}/map.tsv", sep="\t", header=None, index_col=0)[1].reindex(Ea.eid).values
atac = np.load(f"{X}/regulatome/m1_binding/v2/counts/atac_counts.npz")["atac"][act].astype(float); atac_cpm = atac / atac.sum() * 1e6; logw = np.log(Ea.end.values - Ea.start.values)
mid = ((Ea.start.values + Ea.end.values) // 2).astype(np.int64)
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]
tf_sum, tf_reads = {}, {}
for t in tfs: tf_sum[t] = np.asarray(sp.load_npz(f"{G}/cellmats/{t}.raw.npz").sum(0)).ravel(); tf_reads[t] = np.load(f"{G}/cellmats/{t}.cells.npz")["reads"].astype(float).sum()
assert len(tf_sum[tfs[0]]) == nE
tot_sum = sum(tf_sum.values()); tot_reads = sum(tf_reads.values())
cpm = {t: tf_sum[t] / tf_reads[t] * 1e6 for t in tfs}; hot_ind = {t: cpm[t] > np.percentile(cpm[t], 90) for t in tfs}; hot_all = sum(hot_ind[t].astype(int) for t in tfs)
def h_loo(t): return hot_all - hot_ind[t].astype(int)
def design(pool_sum, pool_reads, h=None):  # Gate 1 v2.1 design, plus log1p(h) when given
    pc = pool_sum / pool_reads * 1e6; cols = [np.ones(nE), np.log1p(atac_cpm), np.log1p(pc), logw, gc, gc ** 2, mp]
    if h is not None: cols.append(np.log1p(h))
    return np.column_stack(cols)
def glm_mu(y, off, Xd, se=False):  # Gate 1 v2.1 IRLS; optionally returns coefficients and SEs
    b = np.zeros(Xd.shape[1]); b[0] = np.log(y.sum() / np.exp(off).sum())
    for _ in range(30):
        eta = off + Xd @ b; mu = np.exp(eta); z = eta - off + (y - mu) / mu
        XtW = Xd.T * mu; bn = np.linalg.solve(XtW @ Xd + 1e-6 * np.eye(len(b)), XtW @ z)
        if np.max(np.abs(bn - b)) < 1e-7: b = bn; break
        b = bn
    mu = np.exp(off + Xd @ b)
    if not se: return mu
    cov = np.linalg.inv((Xd.T * mu) @ Xd + 1e-6 * np.eye(len(b))); return mu, b, np.sqrt(np.diag(cov))
def fit_tf(y, N, t):
    Xd = design(tot_sum - tf_sum[t], tot_reads - tf_reads[t], h_loo(t)); return glm_mu(y, np.full(nE, np.log(N / 1e6)), Xd, se=True)
def phi_mom(y, mu):
    k = mu >= 5
    if k.sum() < 10: return 0.0
    v = float((((y[k] - mu[k]) ** 2) - mu[k]).sum() / (mu[k] ** 2).sum()); return max(v, 0.0)
def bs(y, mu, phi):
    k = mu >= 1; return k, np.log((y[k] + .5) / (mu[k] + .5)), np.sqrt(1 / (y[k] + .5) + phi)
def write_ash_in(d, unit, k, b, s):
    os.makedirs(f"{d}/in", exist_ok=True); pd.DataFrame(dict(eid=np.flatnonzero(k), b=b, s=s)).to_csv(f"{d}/in/{unit}.tsv", sep="\t", index=False, float_format="%.6g")
def run_ash(d):
    r = subprocess.run(["Rscript", f"{R3}/r3_ash.R", d, str(CORES)], capture_output=True, text=True); print(r.stdout.strip(), r.stderr.strip()[-500:], flush=True); assert r.returncode == 0
def read_ash(d, unit):
    o = pd.read_csv(f"{d}/out/{unit}.tsv", sep="\t"); g = pd.read_csv(f"{d}/out/{unit}.g.tsv", sep="\t").iloc[0]
    beta = np.zeros(nE); psd = np.zeros(nE); lfsr = np.ones(nE); ix = o.eid.values.astype(np.int64); beta[ix] = o.beta.values; psd[ix] = o.psd.values; lfsr[ix] = o.lfsr.values
    return beta, psd, lfsr, float(g.pi0), float(g.tau)
# ---------- element-gene pairs: <= 100 kb; > 20 kb only with an ABC link or a Hi-C loop (Peakachu 0.7, 5 kb and 10 kb) to the promoter ----------
A = pd.read_csv(f"{X}/regulatome/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene", "TargetGeneTSS", "ABC.Score"])
coord = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(Ea.chrom, Ea.start, Ea.end))}; A["e"] = (A.chr + ":" + A.start.astype(str) + "-" + A.end.astype(str)).map(coord); A = A.dropna(subset=["e"]); A["e"] = A.e.astype(int)
TS = A.drop_duplicates("TargetGene")[["TargetGene", "chr", "TargetGeneTSS"]].sort_values("TargetGene").reset_index(drop=True); genes = TS.TargetGene.values; ng = len(genes); gix = {g: i for i, g in enumerate(genes)}; A["g"] = A.TargetGene.map(gix)
abc_pair = {(e, g): s for e, g, s in zip(A.e.values, A.g.values, A["ABC.Score"].values)}
loops = pd.concat([pd.read_csv(f"{X}/hic/work_wt293t/loops/peakachu_{r}.loops.0.7.bedpe", sep="\t", header=None, usecols=[0, 1, 2, 3, 4, 5]) for r in (5000, 10000)])
ebych = {c: (d.start.values, d.end.values, d.index.values) for c, d in Ea.groupby("chrom", sort=False)}; tbych = {c: (d.TargetGeneTSS.values.astype(np.int64), d.index.values) for c, d in TS.sort_values("TargetGeneTSS").groupby("chr")}
loop_pair = set()
for c1, s1, e1, c2, s2, e2 in loops.itertuples(index=False):
    if c1 != c2 or c1 not in ebych or c1 not in tbych: continue
    es, ee, ei = ebych[c1]; ts, ti = tbych[c1]
    for (as_, ae), (bs_, be) in (((s1, e1), (s2, e2)), ((s2, e2), (s1, e1))):
        lo = np.searchsorted(ee, as_, side="right"); hi = np.searchsorted(es, ae, side="left"); el = ei[lo:hi]
        if not len(el): continue
        gl = ti[np.searchsorted(ts, bs_ - 1000, side="right"):np.searchsorted(ts, be + 1000, side="left")]
        for e in el:
            for g in gl: loop_pair.add((int(e), int(g)))
PE, PG, PD = [], [], []
for c, (ts, ti) in tbych.items():
    if c not in ebych: continue
    es, ee, ei = ebych[c]; m = ((es + ee) // 2).astype(np.int64)
    for t_, g in zip(ts, ti):
        lo = np.searchsorted(m, t_ - 100000, side="left"); hi = np.searchsorted(m, t_ + 100000, side="right"); el = ei[lo:hi]; d = np.abs(m[lo:hi] - t_)
        PE.append(el); PG.append(np.full(len(el), g)); PD.append(d)
PE = np.concatenate(PE); PG = np.concatenate(PG); PD = np.concatenate(PD).astype(float)
abcv = np.array([abc_pair.get((int(e), int(g)), 0.0) for e, g in zip(PE, PG)]); lp = np.array([(int(e), int(g)) in loop_pair for e, g in zip(PE, PG)])
keep = (PD <= 20000) | (abcv > 0) | lp; PE, PG, PD, abcv, lp = PE[keep], PG[keep], PD[keep], abcv[keep], lp[keep]; PW = np.exp(-PD / 10000); P5 = PD <= 5000
log(f"elements {nE}; genes {ng}; ABC links mapped {len(A)}; loops {len(loops)}; element-gene pairs kept {len(PE)} (ABC {int((abcv > 0).sum())}, loop {int(lp.sum())})")
def gene_table(beta, lfsr, excl=None):
    bp = np.maximum(beta, 0.0)
    if excl is not None: bp = np.where(excl, 0.0, bp)
    sup = (beta > 0) & (lfsr <= 0.10) & (True if excl is None else ~excl); c = bp[PE] * PW; SB = np.bincount(PG, c, ng); sp_ = sup[PE]
    csup = np.where(sp_, c, 0.0); mx = np.zeros(ng); np.maximum.at(mx, PG, csup); nsup = np.bincount(PG, sp_.astype(float), ng).astype(int); w5 = np.bincount(PG, (sp_ & P5).astype(float), ng) > 0
    top = np.full(ng, -1); order = np.lexsort((-csup, PG)); first = np.r_[True, PG[order][1:] != PG[order][:-1]]; top[PG[order][first]] = PE[order][first]
    SA = np.bincount(A.g.values, np.maximum(beta[A.e.values], 0) * A["ABC.Score"].values, ng)
    share = np.divide(mx, SB, out=np.zeros(ng), where=SB > 0); lst = (SB > 0) & (share >= 0.5) & ((nsup >= 2) | w5)
    T = pd.DataFrame(dict(gene=genes, S_B=SB, S_ABC=SA, n_supporting=nsup, top_share=share, top_element=top, top_lfsr=np.where(top >= 0, lfsr[np.maximum(top, 0)], np.nan), supporting_within_5kb=w5, listed=lst))
    T = T[T.S_B > 0].sort_values(["listed", "S_B"], ascending=[False, False]); T["rank"] = np.where(T.listed, np.arange(1, len(T) + 1), np.nan); return T
units = TEST if TEST else tfs
MODS = {"PRC2_core": ["EZH2", "MTF2", "PCGF6"], "NuRD_MBD3core": ["MBD3", "RBBP4", "RBBP7"], "WIZ_ZNF644_group": ["DR1", "THAP6", "CHAMP1", "WIZ", "ZNF644", "SRCAP"]}
if mode == "fit":
    for d in ("elements", "genes", "modules"): os.makedirs(f"{O}/{d}", exist_ok=True)
    W = f"{O}/tmp/ash_main"; shutil.rmtree(W, ignore_errors=True); fitd = {}; rows = []
    for t in units:
        y = tf_sum[t]; N = tf_reads[t]; mu, b_, se_ = fit_tf(y, N, t); phi = phi_mom(y, mu); k, b, s = bs(y, mu, phi); write_ash_in(W, t, k, b, s); fitd[t] = (y, mu, phi, b_[-1], se_[-1], int(k.sum()))
    for m, mem in MODS.items():
        mem = [x for x in mem if x in fitd] if TEST else mem
        if not mem: continue
        y = sum(fitd[x][0] for x in mem); mu = sum(fitd[x][1] for x in mem); phi = phi_mom(y, mu); k, b, s = bs(y, mu, phi); write_ash_in(W, m, k, b, s); fitd[m] = (y, mu, phi, np.nan, np.nan, int(k.sum()))
    log("GLM + inputs done"); run_ash(W); log("ash done")
    for u, (y, mu, phi, hc, hse, nsc) in fitd.items():
        beta, psd, lfsr, pi0, tau = read_ash(W, u); h = h_loo(u) if u in tf_sum else hot_all
        sup = (beta > 0) & (lfsr <= .10); neg = (beta < 0) & (lfsr <= .10); sc = mu >= 1
        el = pd.DataFrame(dict(eid=Ea.eid, chrom=Ea.chrom, start=Ea.start, end=Ea.end, y=y, mu=mu, h=h, beta=beta, psd=psd, lfsr=lfsr, call=np.where(sup, 1, np.where(neg, -1, 0))))[sc]
        isMod = u in MODS; d = "modules" if isMod else "elements"; el.to_csv(f"{O}/{d}/{u}.elements.tsv.gz" if isMod else f"{O}/elements/{u}.tsv.gz", sep="\t", index=False, float_format="%.5g")
        T = gene_table(beta, lfsr); L = T[T.listed]; T.to_csv(f"{O}/{'modules' if isMod else 'genes'}/{u}{'.genes' if isMod else ''}.tsv", sep="\t", index=False, float_format="%.5g")
        rows.append(dict(unit=u, kind="module" if isMod else "tf", members=",".join(MODS[u]) if isMod else u, flag="not_PRC2_enriched" if u == "PRC2_core" else ".", reads=sum(tf_reads[x] for x in MODS[u]) if isMod else tf_reads[u],
                         n_scored=nsc, phi=round(phi, 5), hot_coef=hc, hot_se=hse, pi0=pi0, one_minus_pi0=1 - pi0, tau=tau, n_supporting=int(sup.sum()), n_negative=int(neg.sum()),
                         n_list_uncapped=len(L), n_list=min(len(L), 1000), median_lfsr_list=float(L.top_lfsr.median()) if len(L) else np.nan))
    pd.DataFrame(rows).to_csv(f"{O}/tf_summary_fit.tsv", sep="\t", index=False, float_format="%.5g"); log("fit written", len(rows))
if mode == "validate":
    S = pd.read_csv(f"{O}/tf_summary_fit.tsv", sep="\t"); St = S[S.kind == "tf"].set_index("unit"); os.makedirs(f"{O}/validation", exist_ok=True)
    W = f"{O}/tmp/ash_val"; shutil.rmtree(W, ignore_errors=True); rng = np.random.default_rng(20261106); meta = {}
    deep = [t for t in units if tf_reads[t] >= 20000]
    for t in deep:
        for f in (.10, .25, .50, .75):
            y = rng.binomial(tf_sum[t].astype(np.int64), f).astype(float); mu, _, _ = fit_tf(y, tf_reads[t] * f, t); phi = phi_mom(y, mu); k, b, s = bs(y, mu, phi); u = f"thin__{t}__{int(f * 100)}"; write_ash_in(W, u, k, b, s); meta[u] = (t, f)
    for t in units:  # split-half PPC: Gate 1 v2.1 first split; fit on half A, predict half B
        Rm = sp.load_npz(f"{G}/cellmats/{t}.raw.npz").tocsr(); rd = np.load(f"{G}/cellmats/{t}.cells.npz")["reads"].astype(float)
        hh = np.random.default_rng(20261029 + tfs.index(t)).permutation(Rm.shape[0]) < Rm.shape[0] // 2
        el = pd.read_csv(f"{O}/elements/{t}.tsv.gz", sep="\t"); mu = np.zeros(nE); mu[el.index if False else Ea.eid.isin(el.eid).values] = el.mu.values
        yA = np.asarray(Rm[hh].sum(0)).ravel(); yB = np.asarray(Rm[~hh].sum(0)).ravel(); muA = mu * rd[hh].sum() / rd.sum(); muB = mu * rd[~hh].sum() / rd.sum()
        phiA = phi_mom(yA, muA); k, b, s = bs(yA, muA, phiA); write_ash_in(W, f"half__{t}", k, b, s); meta[f"half__{t}"] = (t, (yB, muB, phiA, b, k))
    log("validation inputs done", len(meta)); run_ash(W); log("validation ash done")
    TH = []
    for u, (t, f) in meta.items():
        if not u.startswith("thin__"): continue
        beta, psd, lfsr, pi0, tau = read_ash(W, u); TH.append(dict(tf=t, frac=f, reads=tf_reads[t] * f, one_minus_pi0=1 - pi0, tau=tau, n_supporting=int(((beta > 0) & (lfsr <= .1)).sum())))
    TH = pd.DataFrame(TH); full = pd.DataFrame(dict(tf=St.index, frac=1.0, reads=St.reads.values, one_minus_pi0=St.one_minus_pi0.values, tau=St.tau.values, n_supporting=St.n_supporting.values))
    D = pd.concat([full, TH], ignore_index=True); D.to_csv(f"{O}/validation/depth_thinning.tsv", sep="\t", index=False, float_format="%.5g")
    from sklearn.isotonic import IsotonicRegression
    lr = np.log10(D.reads.values.clip(1)); iso = IsotonicRegression(increasing=True).fit(lr, D.one_minus_pi0.values); grid = np.linspace(lr.min(), lr.max(), 2000); fit_ = iso.predict(grid)
    thr = 2 * float(np.median(full.one_minus_pi0)); above = np.flatnonzero(fit_ > thr); r_star = float(10 ** grid[above[0]]) if len(above) else np.inf
    lab = {t: ("undetectable at this depth" if St.reads[t] < r_star else ("accessibility-equivalent" if St.one_minus_pi0[t] < 0.01 else "specific (graded)")) for t in St.index}
    json.dump(dict(curve="isotonic regression of 1 - pi0 on log10 reads, full-depth TFs and thinned fits", threshold_one_minus_pi0=thr, reads_star=r_star, n_points=len(D)), open(f"{O}/validation/depth_curve.json", "w"), indent=1)
    PP = []
    for u, v in meta.items():
        if not u.startswith("half__"): continue
        t = v[0]; yB, muB, phiA, bA, k = v[1]; beta, psd, lfsr, pi0, tau = read_ash(W, u); kk = k & (muB >= 1)
        rs = np.random.default_rng(7); bb = rs.normal(beta[kk][None, :], np.maximum(psd[kk], 1e-9)[None, :], (200, kk.sum())); m = muB[kk][None, :] * np.exp(bb)
        draws = rs.poisson(m) if phiA <= 0 else rs.negative_binomial(1 / phiA, 1 / (1 + phiA * m)); lo, hi = np.percentile(draws, 5, axis=0), np.percentile(draws, 95, axis=0); inside = (yB[kk] >= lo) & (yB[kk] <= hi)
        bB = np.log((yB[kk] + .5) / (muB[kk] + .5)); sl = float(np.polyfit(beta[kk], bB, 1)[0]) if beta[kk].std() > 0 else np.nan; supA = (beta[kk] > 0) & (lfsr[kk] <= .1)
        PP.append(dict(tf=t, n_elements=int(kk.sum()), coverage_90=float(inside.mean()), coverage_90_supportingA=float(inside[supA].mean()) if supA.any() else np.nan, slope_bB_on_betaA=sl, n_supporting_A=int(supA.sum()), pi0_A=pi0))
    pd.DataFrame(PP).to_csv(f"{O}/validation/ppc_split_half.tsv", sep="\t", index=False, float_format="%.4g")
    TRUTH = {"CTCF": "CTCF_majority2of3.bed", "MAZ": "MAZ_HEK293_peaks.bed", "NFYA": "NFYA_K562_peaks.bed", "ZBTB7A": "ZBTB7A_HEK293_peaks.bed", "ZIC2": "ZIC2_HEK293_peaks.bed", "ZNF282": "ZNF282_HEK293_peaks.bed", "ZNF777": "ZNF777_HEK293_peaks.bed"}
    from sklearn.metrics import average_precision_score
    AU = []
    for t, f in TRUTH.items():
        if t not in units: continue
        pk = pd.read_csv(f"{X}/data/{f}", sep="\t", header=None, usecols=[0, 1, 2], comment="#"); pos = np.zeros(nE, bool)
        for c, d in pk.groupby(0):
            if c not in ebych: continue
            es, ee, ei = ebych[c]; ss = np.sort(d[1].values); se_ = np.sort(d[2].values)
            pos[ei[(np.searchsorted(ss, ee, side="left") - np.searchsorted(se_, es, side="right")) > 0]] = True
        el = pd.read_csv(f"{O}/elements/{t}.tsv.gz", sep="\t"); beta = np.zeros(nE); m_ = Ea.eid.isin(el.eid).values; beta[m_] = el.beta.values
        y = tf_sum[t]; mu21 = glm_mu(y, np.full(nE, np.log(tf_reads[t] / 1e6)), design(tot_sum - tf_sum[t], tot_reads - tf_reads[t])); res = np.log2((y + .5) / (mu21 + .5))
        AU.append(dict(tf=t, truth=f, n_pos=int(pos.sum()), n_elements=nE, base_rate=float(pos.mean()), aupr_beta=average_precision_score(pos, beta), aupr_raw_cpm=average_precision_score(pos, cpm[t]), aupr_gate1_residual=average_precision_score(pos, res)))
    pd.DataFrame(AU).to_csv(f"{O}/validation/aupr_bulk_chip.tsv", sep="\t", index=False, float_format="%.4g")
    hp = subprocess.run(f"unzip -p {X}/data/hpa/rna_celline.tsv.zip | awk -F'\\t' '$3==\"HEK293\"'", shell=True, capture_output=True, text=True).stdout
    H = pd.DataFrame([l.split("\t") for l in hp.strip().split("\n")], columns=["ens", "gene", "line", "TPM", "pTPM", "nTPM"]); nt = H.drop_duplicates("gene").set_index("gene").nTPM.astype(float)
    HP, HO, AT = [], [], []
    for t in units:
        el = pd.read_csv(f"{O}/elements/{t}.tsv.gz", sep="\t"); beta = np.zeros(nE); lfsr = np.ones(nE); m_ = Ea.eid.isin(el.eid).values; beta[m_] = el.beta.values; lfsr[m_] = el.lfsr.values
        T = pd.read_csv(f"{O}/genes/{t}.tsv", sep="\t"); T["listed"] = T.listed.astype(bool); L = T[T.listed].head(1000); x = T.set_index("gene").S_B.reindex(nt.index).dropna(); r = spearmanr(x, nt.reindex(x.index))[0] if len(x) > 10 else np.nan
        HP.append(dict(tf=t, n_genes=len(x), spearman_SB_nTPM=r, median_nTPM_list=float(nt.reindex(L.gene).median()) if len(L) else np.nan, median_nTPM_universe=float(nt.reindex(genes).median())))
        h = h_loo(t); sup = (beta > 0) & (lfsr <= .1); sc = Ea.eid.isin(el.eid).values; T2 = gene_table(beta, lfsr, excl=h >= 20); L2 = T2[T2.listed].head(1000); a_, b_ = set(L.gene), set(L2.gene)
        HO.append(dict(tf=t, share_supporting_hot20=float((h[sup] >= 20).mean()) if sup.any() else np.nan, share_scored_hot20=float((h[sc] >= 20).mean()), n_list=len(L), n_list_without_hot=len(L2), jaccard_without_hot=len(a_ & b_) / len(a_ | b_) if a_ | b_ else np.nan))
        cA = atac_cpm[PE] * PW; SA = np.bincount(PG, cA, ng); top = np.argsort(-SA)[:max(len(L), 1)]; base = set(genes[top]) if len(L) else set()
        AT.append(dict(tf=t, n_list=len(L), atac_baseline_overlap=len(a_ & base), frac_list_in_atac_baseline=len(a_ & base) / len(a_) if a_ else np.nan))
    pd.DataFrame(HP).to_csv(f"{O}/validation/hpa_spearman.tsv", sep="\t", index=False, float_format="%.4g"); pd.DataFrame(HO).to_csv(f"{O}/validation/hot_stratification.tsv", sep="\t", index=False, float_format="%.4g"); pd.DataFrame(AT).to_csv(f"{O}/validation/atac_proximal_baseline.tsv", sep="\t", index=False, float_format="%.4g")
    S["depth_label"] = [lab.get(u, "module (not labelled)") for u in S.unit]; S.to_csv(f"{O}/tf_summary.tsv", sep="\t", index=False, float_format="%.5g"); os.remove(f"{O}/tf_summary_fit.tsv"); log("validation written")
    if not TEST:
        shutil.rmtree(f"{R3}/tmp", ignore_errors=True)
        fs = sorted(p for p in glob.glob(f"{R3}/**/*", recursive=True) if os.path.isfile(p) and not p.endswith("manifest_v6.sha256") and "/logs/" not in p)
        open(f"{R3}/manifest_v6.sha256", "w").write("".join(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {os.path.relpath(p, X)}\n" for p in fs)); log("manifest v6", hashlib.sha256(open(f"{R3}/manifest_v6.sha256", "rb").read()).hexdigest(), len(fs))
