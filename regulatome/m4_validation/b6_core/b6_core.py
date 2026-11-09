# B6 (third Perturb-seq read): R5 core lists (a)-(e) from b6_lists.py scored against the existing kd_response/out_v2 matrices, per SPEC.md
# (incl. Addendum 1). Effect = B3/B4 coded definition (b3b4.py score(): mean |double-centred response| over list genes minus the mean over 1,000
# expression-matched random sets (20 NTC-mean bins), / SD of the random sets). load(), score() and bstat() are copied from b3b4.py unchanged
# except the seed. Writes pooled.tsv, per_tf.tsv, diagonal.tsv to m4_validation/b6_core/; README.md and manifest_b6.sha256 are written by the sbatch.
import hashlib, os, subprocess
import numpy as np, pandas as pd
from scipy import stats
X = "/work/users/d/y/dyy12/XuLab"; M4 = f"{X}/regulatome/m4_validation"; O = f"{M4}/b6_core"
assert not os.path.exists(f"{O}/manifest_b6.sha256") and not os.path.exists(f"{O}/pooled.tsv"), "B6 outputs exist; frozen"
v = subprocess.run("sha256sum regulatome/r5_core/manifest_v7.sha256 && sha256sum -c --quiet regulatome/r5_core/manifest_v7.sha256", shell=True, cwd=X, capture_output=True, text=True)
assert v.returncode == 0 and v.stdout.startswith("b89b483bba64b4e54cd39ed4f116510528702e3d35ebd9b282b7fe01f45f76ac"), "manifest v7 check failed: " + v.stdout + v.stderr
print("manifest v7 verified", v.stdout.split()[0], flush=True)
SEED = 20261109; NR = 1000; NB = 10000; ARMS = {"arm1": "arm1_high_knockdown", "arm2": "arm2_all_cell"}; ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; panel = lambda t: ALIAS.get(t, t)
LISTS = ["a_core", "b_promoter_only", "c_distal_only", "d_atac_only", "e_binding_only"]; MIN_A = 50; MIN_L = 10
Expr = pd.read_csv(f"{M4}/expression_flag/hpa_hek293_flag.tsv", sep="\t"); NOTEXP = set(Expr[~Expr.expressed_nTPM_ge_1].panel_tf)
CEIL = pd.read_csv(f"{M4}/b3b4/reliability_ceilings.tsv", sep="\t")
LST = {}
for f in sorted(os.listdir(f"{O}/lists")):
    T_ = pd.read_csv(f"{O}/lists/{f}", sep="\t"); LST[f[:-4]] = {k: list(T_.gene[T_.list == k]) for k in LISTS}
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
INPUTS = {p: sha(f"{M4}/{p}") for p in [f"kd_response/out_v2/{ARMS[a]}/{x}" for a in ARMS for x in ("kd_response.npz", "guide_qc.tsv")] + ["b3b4/reliability_ceilings.tsv", "expression_flag/hpa_hek293_flag.tsv"]}
pd.Series(INPUTS, name="sha256").rename_axis("file").to_csv(f"{O}/inputs_sha256.tsv", sep="\t")
def load(a):  # b3b4.py load(), unchanged
    d = np.load(f"{M4}/kd_response/out_v2/{ARMS[a]}/kd_response.npz", allow_pickle=True); q = pd.read_csv(f"{M4}/kd_response/out_v2/{ARMS[a]}/guide_qc.tsv", sep="\t"); u = d["units"].tolist(); ui = {x: k for k, x in enumerate(u)}; uni = d["ntc_detect"] >= .1; g = d["genes"][uni]; Z = d["z"][:, uni].astype(float); kp = q[q.kept == 1]; prim = kp[kp.kind.isin(["all", "construct"])]; tfs = sorted(prim.tf.unique())
    P_ = np.vstack([np.nanmean(Z[[ui[x] for x in prim.unit[prim.tf == t]]], axis=0) for t in tfs]); P_ = np.nan_to_num(P_); gm = P_.mean(0); Rr = P_ - gm; tm = Rr.mean(1, keepdims=True); Rr = Rr - tm
    return dict(g=g, gi={x: k for k, x in enumerate(g)}, R=Rr, tfs=tfs, mean=d["ntc_mean"][uni].astype(float), q=q)
def bstat(e, w=None, rng=None):  # b3b4.py bstat(), unchanged
    e = np.asarray(e, float); n = len(e)
    if n == 0: return dict(n=0)
    w = np.ones(n) if w is None else np.asarray(w, float); mean = np.average(e, weights=w); o = dict(n=n, mean=round(mean, 3))
    if n < 2: return o
    ix = rng.integers(0, n, (NB, n)); ww = w[ix]; bs = np.where(ww.sum(1) > 0, (e[ix] * ww).sum(1) / np.maximum(ww.sum(1), 1e-12), np.nan); pos = int((e > 0).sum()); nz = int((e != 0).sum()); o.update(ci_lo=round(np.nanpercentile(bs, 2.5), 3), ci_hi=round(np.nanpercentile(bs, 97.5), 3), n_pos=pos, sign_p=float(f"{stats.binomtest(pos, nz, .5).pvalue:.3g}")); return o
PER, DIAG = [], []
for a in ARMS:
    D = load(a); rng = np.random.default_rng(SEED + (a == "arm2")); g, gi, Rr, tfs = D["g"], D["gi"], D["R"], D["tfs"]; A_ = np.abs(Rr); pts = sorted({panel(t) for t in tfs}); Rp = np.vstack([Rr[[i for i, t in enumerate(tfs) if panel(t) == p]].mean(0) for p in pts]); Ap = np.abs(Rp); pi = {p: k for k, p in enumerate(pts)}
    bins = pd.qcut(D["mean"], 20, labels=False, duplicates="drop"); BG = [np.flatnonzero(bins == b) for b in range(bins.max() + 1)]
    def rand_sets(ix):
        cnt = np.bincount(bins[ix], minlength=len(BG)); return np.hstack([BG[b][rng.integers(0, len(BG[b]), (NR, c))] for b, c in enumerate(cnt) if c])
    def score(ab, ix, rs=None):
        rs = rand_sets(ix) if rs is None else rs; m = ab[rs].mean(1); return ab[ix].mean() - m.mean(), (ab[ix].mean() - m.mean()) / m.std()
    uni_idx = lambda genes: np.array(sorted({gi[x] for x in genes if x in gi}), int)
    W = CEIL[CEIL.arm == a].groupby("panel_tf").weight.mean()
    entered = []
    for p in pts:
        if p not in LST: continue
        ix = {k: uni_idx(LST[p][k]) for k in LISTS}
        if len(ix["a_core"]) < MIN_A: continue
        entered.append(p); r = dict(arm=a, tf=p, weight=W.get(p, np.nan), not_expressed=p in NOTEXP); sd = Rp[pi[p]].std()
        for k in LISTS:
            r[f"n_{k}"] = len(ix[k])
            if len(ix[k]) >= MIN_L: _, e = score(Ap[pi[p]], ix[k]); r[f"effect_{k}"] = round(e, 4); r[f"signed_{k}"] = round(-Rp[pi[p]][ix[k]].mean() / sd, 4)
            else: r[f"effect_{k}"] = np.nan; r[f"signed_{k}"] = np.nan
        # Addendum 2: responders = top 5% (10%) of |R| for this TF's knockdown in the arm universe; Fisher OR of list membership among responders vs all other universe genes
        ab = Ap[pi[p]]; ng = len(ab)
        for fr in (5, 10):
            resp = np.zeros(ng, bool); resp[np.argsort(-ab, kind="stable")[:int(round(ng * fr / 100))]] = True
            for k in LISTS[:4]:
                if len(ix[k]) < MIN_L: r[f"or{fr}_{k}"] = r[f"p{fr}_{k}"] = r[f"logor{fr}_{k}"] = np.nan; continue
                inl = np.zeros(ng, bool); inl[ix[k]] = True; t2 = [[int((inl & resp).sum()), int((~inl & resp).sum())], [int((inl & ~resp).sum()), int((~inl & ~resp).sum())]]
                o_, pv = stats.fisher_exact(t2); r[f"or{fr}_{k}"] = round(o_, 4); r[f"p{fr}_{k}"] = float(f"{pv:.3g}"); r[f"logor{fr}_{k}"] = round(np.log((t2[0][0] + .5) * (t2[1][1] + .5) / ((t2[0][1] + .5) * (t2[1][0] + .5))), 4)
        PER.append(r)
    # diagonal on the core lists, as in B3: TF i's list (a) under KD j, every KD unit-TF of the arm whose panel TF entered
    sel_t = [t for t in tfs if panel(t) in entered]; n = len(sel_t); ti = [tfs.index(t) for t in sel_t]; Mx = np.full((n, n), np.nan)
    for i, t in enumerate(sel_t):
        ix = uni_idx(LST[panel(t)]["a_core"]); rs = rand_sets(ix)
        for j in range(n): Mx[i, j] = A_[ti[j]][ix].mean() - A_[ti[j]][rs].mean(1).mean()
    np.savez_compressed(f"{O}/diagonal_{a}.npz", M=Mx, tfs=np.array(sel_t))
    same = np.array([[panel(x) == panel(y) for y in sel_t] for x in sel_t]); off = np.where(same, np.nan, Mx)
    def diag_test(sel, label):  # b3b4.py diag_test(), unchanged
        s = np.array(sel); d_ = Mx[s, s] - np.nanmean(off[np.ix_(s, s)], axis=1); w = stats.wilcoxon(d_, alternative="greater"); pos = int((d_ > 0).sum()); pr = np.random.default_rng(SEED); ob = d_.mean(); cnt = 0
        for _ in range(NB):
            pk = [pr.choice(s[~np.isnan(off[i, s])]) for i in s]; cnt += (np.mean([Mx[i, j] for i, j in zip(s, pk)]) - np.nanmean(off[np.ix_(s, s)], axis=1).mean()) >= ob
        DIAG.append(dict(arm=a, set=label, n_tfs=len(s), mean_diag=round(np.diag(Mx)[s].mean(), 4), mean_off=round(np.nanmean(off[np.ix_(s, s)]), 4), mean_diff=round(ob, 4), ci_lo=round(np.percentile(d_[rng.integers(0, len(d_), (NB, len(d_)))].mean(1), 2.5), 4), ci_hi=round(np.percentile(d_[rng.integers(0, len(d_), (NB, len(d_)))].mean(1), 97.5), 4), n_diag_above=pos, wilcoxon_p=float(f"{w.pvalue:.3g}"), perm_p=round((cnt + 1) / (NB + 1), 5)))
    ALL = list(range(n)); diag_test(ALL, "all"); ph = {t: bool((D["q"][D["q"].tf == t].phase1_kd_q < .05).any()) and panel(t) not in NOTEXP for t in sel_t}; diag_test([i for i in ALL if ph[sel_t[i]]], "phase1_pass_and_expressed")
    print(a, "kd tfs", len(tfs), "entered", len(entered), "universe", len(g), flush=True)
P = pd.DataFrame(PER); P.to_csv(f"{O}/per_tf.tsv", sep="\t", index=False, na_rep="NA")
rows = []; rng = np.random.default_rng(SEED)
CON = [("primary", "a_core", None), ("primary", "a_core", "d_atac_only"), ("primary", "a_core", "e_binding_only"),
       ("secondary", "b_promoter_only", None), ("secondary", "b_promoter_only", "d_atac_only"), ("secondary", "b_promoter_only", "e_binding_only"),
       ("secondary", "c_distal_only", None), ("secondary", "c_distal_only", "d_atac_only"), ("secondary", "c_distal_only", "e_binding_only"),
       ("secondary", "b_promoter_only", "c_distal_only")]
for a, G in P.groupby("arm", sort=False):
    for lev, x, y in CON:
        e = G[f"effect_{x}"] - (G[f"effect_{y}"] if y else 0); ok = e.notna(); lab = x if y is None else f"{x}_minus_{y}"
        rows.append(dict(arm=a, level=lev, contrast=lab, weighting="none", **bstat(e[ok].values, rng=rng)))
        wk = ok & G.weight.notna(); rows.append(dict(arm=a, level="secondary", contrast=lab, weighting="ceiling", **bstat(e[wk].values, G.weight[wk].values, rng)))
    for fr, lev in ((5, "coprimary_OR"), (10, "secondary_OR")):
        for k in LISTS[:4]:
            o_ = G[f"or{fr}_{k}"].dropna(); bm = o_.values[rng.integers(0, len(o_), (NB, len(o_)))]; med = np.median(bm, axis=1)
            rows.append(dict(arm=a, level=lev if k in ("a_core", "d_atac_only") else "secondary_OR", contrast=f"OR_top{fr}pct_{k}", weighting="none", n=len(o_), median_OR=round(o_.median(), 4), median_OR_ci_lo=round(np.percentile(med, 2.5), 4), median_OR_ci_hi=round(np.percentile(med, 97.5), 4), frac_OR_gt1=round((o_ > 1).mean(), 4)))
        for k in ("a_core", "b_promoter_only", "c_distal_only"):
            e = (G[f"logor{fr}_{k}"] - G[f"logor{fr}_d_atac_only"]).dropna()
            rows.append(dict(arm=a, level=lev if k == "a_core" else "secondary_OR", contrast=f"logOR_top{fr}pct_{k}_minus_d_atac_only", weighting="none", **bstat(e.values, rng=rng)))
    for k in LISTS:
        s = G[f"signed_{k}"].dropna(); rows.append(dict(arm=a, level="descriptive_signed", contrast=k, weighting="none", n=len(s), mean=round(s.mean(), 3)))
S = pd.DataFrame(rows); S.to_csv(f"{O}/pooled.tsv", sep="\t", index=False, na_rep="NA"); Dg = pd.DataFrame(DIAG); Dg.to_csv(f"{O}/diagonal.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 40)
print(S.to_string(index=False)); print(Dg.to_string(index=False))
