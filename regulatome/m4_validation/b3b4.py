# B3/B4: pooled knockdown validation of the frozen tiers, as pre-specified in m4_validation/README.md. Reads frozen tables and kd_response/out_v2 only; writes to m4_validation/b3b4/.
import gzip, json, os, sys
import numpy as np, pandas as pd
from scipy import stats
X = "/work/users/d/y/dyy12/XuLab"; M4 = f"{X}/regulatome/m4_validation"; R2 = f"{X}/regulatome/r2_targets"; R1 = f"{X}/regulatome/r1_tftf/v2_final_v3"; OUT = f"{M4}/b3b4"; os.makedirs(OUT, exist_ok=False)
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
PT_, TOP, TPAIR = [], [], []; DIAG = []; V3 = []; V4 = []; ceil_rows = []; AB = []; CON = []
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
    # tiers: per panel TF, list and same-size nulls within the universe
    for p in pts:
        if p not in Tier.index: continue
        P0 = PT[p]; atac = P0.S_atac.reindex(g).fillna(-np.inf).values; bind = P0.S_rawcpm.reindex(g).fillna(-np.inf).values; dl = uni_idx(P0.index[P0.rank_k1 <= 1000]); L = {f"direct-{Tier[p]}": dl}; Pr = Prop[Prop.target == p]; L["inferred-physical"] = uni_idx(Pr.gene[Pr.edge_class.isin(PHYS)]); L["inferred-module"] = uni_idx(Pr.gene[Pr.edge_class == "module"]); L["cis-coordinated"] = uni_idx(Pr.gene[Pr.edge_class == "cis-coordinated"]); L["cis-coordinated-loop"] = uni_idx(Pr.gene[Pr.edge_class == "cis-coordinated-loop"])
        for tier, ix in L.items():
            if len(ix) < 10: continue
            n = len(ix); na = np.argsort(-atac)[:n]; nb = np.argsort(-bind)[:n]; s1, e1 = score(Ap[pi[p]], ix); s2, e2 = score(Ap[pi[p]], na); s3, e3 = score(Ap[pi[p]], nb); PT_.append(dict(arm=a, tier=tier, tf=p, n_genes=n, score=round(s1, 4), effect=round(e1, 3), score_atac=round(s2, 4), effect_atac=round(e2, 3), score_binding=round(s3, 4), effect_binding=round(e3, 3), weight=wt_pt[p], not_expressed=p in NOTEXP, phase1_pass=bool((D["q"][D["q"].tf.map(panel) == p].phase1_kd_q < .05).any()), direct_primary=Tier[p] == "specific"))
        for ar in ("promoter_only", "imputed_residual", "blend"):
            ix = uni_idx(pd.read_csv(f"{M4}/arms/{ar}/{p}.tsv", sep="\t").gene); s1, e1 = score(Ap[pi[p]], ix); AB.append(dict(arm=a, ladder=ar, tf=p, n_genes=len(ix), effect=round(e1, 3), not_expressed=p in NOTEXP, tier=Tier[p]))
        s1, e1 = score(Ap[pi[p]], dl); AB.append(dict(arm=a, ladder="raw_direct", tf=p, n_genes=len(dl), effect=round(e1, 3), not_expressed=p in NOTEXP, tier=Tier[p]))
    # both constructs shown separately
    for x, v in D["cons"].items():
        t = x.split("|")[0]; ix = uni_idx(PT[panel(t)].index[PT[panel(t)].rank_k1 <= 1000]); s1, e1 = score(np.abs(v), ix); CON.append(dict(arm=a, unit=x, tf=t, n_genes=len(ix), effect=round(e1, 3)))
    # diagonal: TF i's direct list under KD j, every KD target of the arm
    n = len(tfs); Mx = np.full((n, n), np.nan)
    for i, t in enumerate(tfs):
        ix = uni_idx(PT[panel(t)].index[PT[panel(t)].rank_k1 <= 1000]); rs = rand_sets(ix)
        for j in range(n): Mx[i, j] = A_[j][ix].mean() - A_[j][rs].mean(1).mean()
    np.savez_compressed(f"{OUT}/diagonal_{a}.npz", M=Mx, tfs=np.array(tfs))
    same = np.array([[panel(x) == panel(y) for y in tfs] for x in tfs]); off = np.where(same, np.nan, Mx)
    def diag_test(sel, label):
        s = np.array(sel); d_ = Mx[s, s] - np.nanmean(off[np.ix_(s, s)], axis=1); w = stats.wilcoxon(d_, alternative="greater"); pos = int((d_ > 0).sum()); pr = np.random.default_rng(SEED); ob = d_.mean(); cnt = 0
        for _ in range(NB):
            pk = [pr.choice(s[~np.isnan(off[i, s])]) for i in s]; cnt += (np.mean([Mx[i, j] for i, j in zip(s, pk)]) - np.nanmean(off[np.ix_(s, s)], axis=1).mean()) >= ob
        B = bstat(d_, rng=rng); DIAG.append(dict(arm=a, set=label, n_tfs=len(s), mean_diag=round(np.diag(Mx)[s].mean(), 4), mean_off=round(np.nanmean(off[np.ix_(s, s)]), 4), mean_diff=round(ob, 4), ci_lo=round(np.percentile(d_[rng.integers(0, len(d_), (NB, len(d_)))].mean(1), 2.5), 4), ci_hi=round(np.percentile(d_[rng.integers(0, len(d_), (NB, len(d_)))].mean(1), 97.5), 4), n_diag_above=pos, wilcoxon_p=float(f"{w.pvalue:.3g}"), perm_p=round((cnt + 1) / (NB + 1), 5)))
    ALL = list(range(n)); diag_test(ALL, "all"); ph = {t: bool((D["q"][D["q"].tf == t].phase1_kd_q < .05).any()) and panel(t) not in NOTEXP for t in tfs}; diag_test([i for i in ALL if ph[tfs[i]]], "phase1_pass_and_expressed")
    # V3: signature correlation by frozen class, against no-edge pairs
    Pf = pd.read_csv(f"{R1}/r1_pairs_frozen_v3.tsv", sep="\t", keep_default_na=False, na_values=[""]); Cc = np.corrcoef(Rp); pr = np.random.default_rng(SEED); rows = [(r.a, r.b, r._3, Cc[pi[r.a], pi[r.b]]) for r in Pf.itertuples() if r.a in pi and r.b in pi and r.a not in NOTEXP and r.b not in NOTEXP]; Tv = pd.DataFrame(rows, columns=["a", "b", "cls", "corr"]); Tv["arm"] = a; Tv.to_csv(f"{OUT}/v3_pairs_{a}.tsv", sep="\t", index=False, float_format="%.4f"); ne = Tv["corr"][Tv.cls == "."].values
    for c, G in Tv[Tv.cls != "."].groupby("cls"):
        v = G["corr"].values; pool = np.concatenate([v, ne]); ob = v.mean() - ne.mean(); ct = 0
        for _ in range(NB): pm = pr.permutation(pool); ct += (pm[:len(v)].mean() - pm[len(v):].mean()) >= ob
        V3.append(dict(arm=a, cls=c, n_pairs=len(v), mean_corr=round(v.mean(), 4), mean_no_edge=round(ne.mean(), 4), n_no_edge=len(ne), diff=round(ob, 4), perm_p=round((ct + 1) / (NB + 1), 5)))
    # V4: hub promoters vs matched promoters, fraction of the core's own hub-TF knockdowns that move the gene (|r| top decile of that KD)
    Hb = pd.read_csv(f"{X}/cobinding/results/ourrun_k50_anchor_hubs/anchor_hubs_named.tsv", sep="\t"); Hb = Hb[(Hb.n_tfs >= 5) & (Hb.dist_tss.abs() <= 1000) & Hb.gene_symbol.notna()]; mv = Ap >= np.quantile(Ap, .9, axis=1, keepdims=True); Mt = perf("ADNP"); cov = pd.DataFrame(dict(atac=Mt.S_atac.reindex(g).values, links=Mt.n_elements.reindex(g).values, expr=D["mean"]), index=g).rank(pct=True); hubs = {}
    for r in Hb.itertuples():
        if r.gene_symbol in gi:
            ks = [pi[panel(x)] for x in r.tfs.split(",") if panel(x) in pi]
            if len(ks) >= 3: hubs.setdefault(r.gene_symbol, set()).update(ks)
    ctrl = cov.dropna().drop(index=[h for h in hubs if h in cov.index], errors="ignore"); pr = np.random.default_rng(SEED); Hv = []; Cv = []
    for h, ks in hubs.items():
        if h not in cov.index or cov.loc[h].isna().any(): continue
        k = sorted(ks); dd = ((ctrl.values - cov.loc[h].values) ** 2).sum(1); nn = ctrl.index[np.argsort(dd)[:10]]; Hv.append(mv[k, gi[h]].mean()); Cv.append([mv[k, gi[x]].mean() for x in nn])
    Hv = np.array(Hv); Cv = np.array(Cv); ob = Hv.mean() - Cv.mean(); Fl = np.column_stack([Hv, Cv]); ct = 0; nh = len(Hv)
    for _ in range(NB): pk = pr.integers(0, 11, nh); pv = Fl[np.arange(nh), pk].mean(); ct += abs(pv - (Fl.sum(1) - Fl[np.arange(nh), pk]).mean() / 10) >= abs(ob)
    V4.append(dict(arm=a, n_hub_genes=nh, frac_kds_moving_hub=round(Hv.mean(), 4), frac_kds_moving_matched=round(Cv.mean(), 4), diff=round(ob, 4), perm_p_two_sided=round((ct + 1) / (NB + 1), 5)))
    print(a, "tfs", len(tfs), "panel", len(pts), "universe", ng, "done", flush=True)
T = pd.DataFrame(PT_); T.to_csv(f"{OUT}/per_tf_effects.tsv", sep="\t", index=False); pd.DataFrame(ceil_rows).to_csv(f"{OUT}/reliability_ceilings.tsv", sep="\t", index=False); pd.DataFrame(CON).to_csv(f"{OUT}/construct_effects.tsv", sep="\t", index=False); Ab = pd.DataFrame(AB); Ab.to_csv(f"{OUT}/ablation_per_tf.tsv", sep="\t", index=False)
rows = []; rng = np.random.default_rng(SEED + 7); EXPR = T[~T.not_expressed]
def add(sub, label):
    for (a, tier), G in sub.groupby(["arm", "tier"]):
        if G.tier.iloc[0].startswith("cis-coordinated") and len(G) < 3:
            rows.append(dict(arm=a, tier=tier, set=label, contrast="list", n=len(G), mean=round(G.effect.mean(), 3), note=f"n = {len(G)} ({','.join(G.tf)}), descriptive")); continue
        for con, col in (("list", G.effect), ("atac_null", G.effect_atac), ("binding_null", G.effect_binding), ("list_minus_atac", G.effect - G.effect_atac), ("list_minus_binding", G.effect - G.effect_binding)):
            rows.append(dict(arm=a, tier=tier, set=label, contrast=con, **bstat(col.values, rng=rng)))
        W = G[G.weight.notna()]; rows.append(dict(arm=a, tier=tier, set=label + "_ceiling_weighted", contrast="list", **bstat(W.effect.values, W.weight.values, rng)))
add(EXPR, "expressed"); add(EXPR[EXPR.phase1_pass], "phase1_pass_and_expressed"); add(EXPR[~EXPR.direct_primary & EXPR.tier.str.startswith("inferred")], "recipients_not_specific")
S = pd.DataFrame(rows); S.to_csv(f"{OUT}/pooled_tiers.tsv", sep="\t", index=False); Dg = pd.DataFrame(DIAG); Dg.to_csv(f"{OUT}/diagonal_tests.tsv", sep="\t", index=False); V3t = pd.DataFrame(V3); V3t.to_csv(f"{OUT}/v3_class_similarity.tsv", sep="\t", index=False); V4t = pd.DataFrame(V4); V4t.to_csv(f"{OUT}/v4_hubs.tsv", sep="\t", index=False)
Ar = []
for (a, l), G in Ab[~Ab.not_expressed].groupby(["arm", "ladder"]): Ar.append(dict(arm=a, ladder=l, **bstat(G.effect.values, rng=rng)))
Ar = pd.DataFrame(Ar); Ar.to_csv(f"{OUT}/ablation_pooled.tsv", sep="\t", index=False)
pd.set_option("display.width", 240); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 60)
print(S[S.set == "expressed"].to_string(index=False)); print(S[S.set != "expressed"].to_string(index=False)); print(Dg.to_string(index=False)); print(V3t.to_string(index=False)); print(V4t.to_string(index=False)); print(Ar.to_string(index=False)); print(pd.DataFrame(CON).to_string(index=False))
