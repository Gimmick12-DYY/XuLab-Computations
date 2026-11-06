# B5: second Perturb-seq comparison, declared before v4 and run against R2b v5 (design ruling), "second comparison, after the first was seen".
# B3 machinery copied unchanged from m4_validation/b3b4.py (arms, universe, 20-bin expression-matched random sets with B3's seed 20261103 per arm,
# effect = (list mean |response| - random mean) / random SD, ATAC-only and binding-only nulls at list size, bootstrap CI 10,000, sign test).
# Eligible: >= 10 R2b v5 list genes in the arm's universe. Primary: all eligible lists; sensitivity: expressed only (B3 filter).
# Direct top-1,000 effects are taken from b3b4/per_tf_effects.tsv (not recomputed). B3 pooled results are written beside the B5 results.
# Reads frozen tables and kd_response/out_v2 only; writes m4_validation/b5_r2b/ and manifest_b5.sha256 (outputs + this script).
import gzip, json, os, sys
import numpy as np, pandas as pd
from scipy import stats
X = "/work/users/d/y/dyy12/XuLab"; M4 = f"{X}/regulatome/m4_validation"; R2 = f"{X}/regulatome/r2_targets"; R1 = f"{X}/regulatome/r1_tftf/v2_final_v3"; OUT = f"{M4}/b5_r2b"; os.makedirs(OUT, exist_ok=True); assert not os.path.exists(f"{OUT}/manifest_b5.sha256"), "B5 already run; outputs are frozen"
SEED = 20261103; NR = 1000; NB = 10000; ARMS = {"arm1": "arm1_high_knockdown", "arm2": "arm2_all_cell"}; ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; panel = lambda t: ALIAS.get(t, t)
Tier = pd.read_csv(f"{R2}/r2_summary.tsv", sep="\t").set_index("tf").tier; Expr = pd.read_csv(f"{M4}/expression_flag/hpa_hek293_flag.tsv", sep="\t"); NOTEXP = set(Expr[~Expr.expressed_nTPM_ge_1].panel_tf)
Prop = pd.read_csv(f"{R2}/propagation_ciscoord_v3/propagation_table_v3.tsv.gz", sep="\t", keep_default_na=False); PHYS = {"complex", "complex_weak", "DNA-guided", "physical-only (DNA)", "physical-only (structure)"}; assert not set(Prop.edge_class) - PHYS - {"module", "cis-coordinated", "cis-coordinated-loop"}
def perf(t): return pd.read_csv(f"{R2}/per_tf/{t}.tsv.gz", sep="\t", usecols=["gene", "S_atac", "S_rawcpm", "rank_k1", "n_elements"]).set_index("gene")
PT = {t: perf(t) for t in Tier.index}
def load(a):
    d = np.load(f"{M4}/kd_response/out_v2/{ARMS[a]}/kd_response.npz", allow_pickle=True); q = pd.read_csv(f"{M4}/kd_response/out_v2/{ARMS[a]}/guide_qc.tsv", sep="\t"); u = d["units"].tolist(); ui = {x: k for k, x in enumerate(u)}; uni = d["ntc_detect"] >= .1; g = d["genes"][uni]; Z = d["z"][:, uni].astype(float); kp = q[q.kept == 1]; prim = kp[kp.kind.isin(["all", "construct"])]; tfs = sorted(prim.tf.unique())
    P_ = np.vstack([np.nanmean(Z[[ui[x] for x in prim.unit[prim.tf == t]]], axis=0) for t in tfs]); P_ = np.nan_to_num(P_); gm = P_.mean(0); Rr = P_ - gm; tm = Rr.mean(1, keepdims=True); Rr = Rr - tm; rep = {}; cons = {}
    for t in tfs:
        h = sorted(kp.unit[(kp.tf == t) & (kp.kind == "half")]); c = sorted(kp.unit[(kp.tf == t) & (kp.kind == "construct")]); pair = h if len(h) == 2 else (c if len(c) == 2 else None); rep[t] = np.corrcoef(*[np.nan_to_num(Z[ui[x]]) - gm for x in pair])[0, 1] if pair else np.nan
        for x in c: v = np.nan_to_num(Z[ui[x]]) - gm; cons[x] = v - v.mean()
    return dict(g=g, gi={x: k for k, x in enumerate(g)}, R=Rr, tfs=tfs, rep=pd.Series(rep), cons=cons, mean=d["ntc_mean"][uni].astype(float), q=q, nused=prim.groupby("tf").n_used.sum())
def bstat(e, w=None, rng=None):
    e = np.asarray(e, float); n = len(e)
    if n == 0: return dict(n=0)
    w = np.ones(n) if w is None else np.asarray(w, float); mean = np.average(e, weights=w); o = dict(n=n, mean=round(mean, 3))
    if n < 2: return o
    ix = rng.integers(0, n, (NB, n)); ww = w[ix]; bs = np.where(ww.sum(1) > 0, (e[ix] * ww).sum(1) / np.maximum(ww.sum(1), 1e-12), np.nan); pos = int((e > 0).sum()); nz = int((e != 0).sum()); o.update(ci_lo=round(np.nanpercentile(bs, 2.5), 3), ci_hi=round(np.nanpercentile(bs, 97.5), 3), n_pos=pos, sign_p=float(f"{stats.binomtest(pos, nz, .5).pvalue:.3g}")); return o

R2B = f"{X}/regulatome/r2b_residual_v5"; LAB = "second comparison, after the first was seen"
def r2b(t):
    f = f"{R2B}/per_tf/{t}.tsv"
    if not os.path.exists(f): return []
    d = pd.read_csv(f, sep="\t"); d = d[d["pass"]].sort_values("X", ascending=False).head(1000); return list(d.gene)
B3 = pd.read_csv(f"{M4}/b3b4/per_tf_effects.tsv", sep="\t"); B3d = B3[B3.tier.str.startswith("direct-")].set_index(["arm", "tf"]).effect
PT_, ceil_rows = [], []
for a in ARMS:
    D = load(a); rng = np.random.default_rng(SEED + (a == "arm2")); g, gi, Rr, tfs = D["g"], D["gi"], D["R"], D["tfs"]; A_ = np.abs(Rr); ng = len(g); pts = sorted({panel(t) for t in tfs}); Rp = np.vstack([Rr[[i for i, t in enumerate(tfs) if panel(t) == p]].mean(0) for p in pts]); Ap = np.abs(Rp); pi = {p: k for k, p in enumerate(pts)}
    bins = pd.qcut(D["mean"], 20, labels=False, duplicates="drop"); BG = [np.flatnonzero(bins == b) for b in range(bins.max() + 1)]
    def rand_sets(ix):
        cnt = np.bincount(bins[ix], minlength=len(BG)); return np.hstack([BG[b][rng.integers(0, len(BG[b]), (NR, c))] for b, c in enumerate(cnt) if c])
    def score(ab, ix, rs=None):
        rs = rand_sets(ix) if rs is None else rs; m = ab[rs].mean(1); return ab[ix].mean() - m.mean(), (ab[ix].mean() - m.mean()) / m.std()
    uni_idx = lambda genes: np.array(sorted({gi[x] for x in genes if x in gi}), int)
    rp = D["rep"].copy(); ce = (2 * rp / (1 + rp)); wt = ce.where(rp > 0, 0.0); wt_pt = pd.Series({p: np.nanmean([wt[t] for t in tfs if panel(t) == p]) if any(not np.isnan(wt[t]) for t in tfs if panel(t) == p) else np.nan for p in pts})
    for t in tfs: ceil_rows.append(dict(arm=a, tf=t, panel_tf=panel(t), n_cells=int(D["nused"].get(t, 0)), replicate_r=round(rp[t], 4) if not np.isnan(rp[t]) else np.nan, ceiling=round(ce[t], 4) if not np.isnan(ce[t]) else np.nan, weight=round(wt[t], 4) if not np.isnan(wt[t]) else np.nan))
    for p in pts:
        if p not in Tier.index: continue
        P0 = PT[p]; atac = P0.S_atac.reindex(g).fillna(-np.inf).values; bind = P0.S_rawcpm.reindex(g).fillna(-np.inf).values; ix = uni_idx(r2b(p))
        if len(ix) < 10: continue
        n = len(ix); na = np.argsort(-atac)[:n]; nb = np.argsort(-bind)[:n]; s1, e1 = score(Ap[pi[p]], ix); s2, e2 = score(Ap[pi[p]], na); s3, e3 = score(Ap[pi[p]], nb)
        cp = [ce[t] for t in tfs if panel(t) == p and not np.isnan(ce[t])]
        PT_.append(dict(label=LAB, arm=a, tf=p, tier=Tier[p], n_genes=n, score=round(s1, 4), effect=round(e1, 3), score_atac=round(s2, 4), effect_atac=round(e2, 3), score_binding=round(s3, 4), effect_binding=round(e3, 3), effect_direct_b3=B3d.get((a, p), np.nan), ceiling=round(float(np.mean(cp)), 4) if cp else np.nan, weight=wt_pt[p], not_expressed=p in NOTEXP))
    print(a, "tfs", len(tfs), "panel", len(pts), "universe", ng, "eligible", sum(r["arm"] == a for r in PT_), flush=True)
T = pd.DataFrame(PT_); T.to_csv(f"{OUT}/per_tf_effects.tsv", sep="\t", index=False); C = pd.DataFrame(ceil_rows); C.insert(0, "label", LAB); C.to_csv(f"{OUT}/reliability_ceilings.tsv", sep="\t", index=False)
rows = []; rng = np.random.default_rng(SEED + 7)
for tset, sub in (("all_eligible (primary)", T), ("expressed_only (sensitivity)", T[~T.not_expressed])):
    for a, A_ in sub.groupby("arm"):
        for grp, G in (("all R2b lists", A_), ("non-specific TFs", A_[A_.tier != "specific"]), ("specific TFs", A_[A_.tier == "specific"])):
            for con, col in (("vs 0", G.effect), ("ATAC-only null", G.effect_atac), ("binding-only null", G.effect_binding), ("R2b minus ATAC-only", G.effect - G.effect_atac), ("R2b minus binding-only", G.effect - G.effect_binding)):
                rows.append(dict(label=LAB, tf_set=tset, arm=a, group=grp, contrast=con, **bstat(col.values, rng=rng)))
            H = G[G.effect_direct_b3.notna()]; rows.append(dict(label=LAB, tf_set=tset, arm=a, group=grp, contrast="R2b minus direct top-1,000 (B3), paired", **bstat((H.effect - H.effect_direct_b3).values, rng=rng)))
            W = G[G.weight.notna()]; rows.append(dict(label=LAB, tf_set=tset, arm=a, group=grp, contrast="vs 0, ceiling-weighted", **bstat(W.effect.values, W.weight.values, rng)))
Sx = pd.DataFrame(rows); Sx.to_csv(f"{OUT}/pooled.tsv", sep="\t", index=False)
F = pd.read_csv(f"{M4}/b3b4/pooled_tiers.tsv", sep="\t"); F = F[F.tier.str.startswith("direct")]; F.insert(0, "label", "first comparison (B3/B4), shown beside"); F.to_csv(f"{OUT}/first_comparison_b3_direct.tsv", sep="\t", index=False)
E = T.groupby("arm").tf.apply(lambda x: ",".join(sorted(x))).reset_index(); E.insert(0, "label", LAB); E.to_csv(f"{OUT}/eligible_tfs.tsv", sep="\t", index=False)
import hashlib
fs = [f"{OUT}/{f}" for f in ("per_tf_effects.tsv", "reliability_ceilings.tsv", "pooled.tsv", "first_comparison_b3_direct.tsv", "eligible_tfs.tsv", "README.md")] + [os.path.abspath(__file__)]
open(f"{OUT}/manifest_b5.sha256", "w").write("".join(f"{hashlib.sha256(open(f, 'rb').read()).hexdigest()}  {os.path.relpath(f, X)}\n" for f in fs))
pd.set_option("display.width", 240); pd.set_option("display.max_rows", 400); print(Sx.to_string(index=False)); print(F.to_string(index=False)); print(E.to_string(index=False))
print("manifest b5", hashlib.sha256(open(f"{OUT}/manifest_b5.sha256", "rb").read()).hexdigest())
