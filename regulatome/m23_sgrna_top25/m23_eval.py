# m23_sgrna_top25 evaluation (SPEC.md + Amendment 1): per-TF reliability etc. across matrix versions, m16/m19 comparisons on all-cell vs top25_own, m14 positive control on three versions. Diagnostic, benchmark-developed.
import json, os
import numpy as np, pandas as pd
from scipy.stats import rankdata
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m23_sgrna_top25"; MX = f"{O}/out/matrices"; OUT = f"{O}/out"; LAB = "diagnostic, benchmark-developed"; rng = np.random.default_rng(20261112)
ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
STRONG = ["CTCF", "DNMT1", "E4F1", "MEF2A", "NRF1", "PA2G4", "RBBP7", "RBPJ", "SAFB", "SETDB1"]
VERS = ["all_cell_min25", "top25_own", "top25_own_ntc25", "random25_d0", "random25_d1", "random25_d2"]
UNIV = np.load(f"{RG}/m9_abc_diagnosis/work/responses_arm2.npz", allow_pickle=True)["genes"].astype(str); ui = {g: k for k, g in enumerate(UNIV)}; NG = len(UNIV)
def load(v):
    d = np.load(f"{MX}/{v}/kd_response.npz", allow_pickle=True); q = pd.read_csv(f"{MX}/{v}/guide_qc.tsv", sep="\t"); units = list(d["units"].astype(str)); u = {x: k for k, x in enumerate(units)}
    keep = d["ntc_detect"] >= .1; g = d["genes"].astype(str); gi = pd.Series(np.arange(len(g)), index=g); gi = gi[~gi.index.duplicated(keep="last")]; cols = gi.reindex(UNIV).values
    Z = d["z"].astype(float); ok = keep
    def vec(names):
        r = np.nanmean(Z[[u[x] for x in names]], 0); r = np.where(ok, r, np.nan); out = np.full(NG, np.nan); m = ~np.isnan(cols); out[m] = r[cols[m].astype(int)]; return out
    K = q[q.kept == 1]; dual = set(K[K.kind == "construct"].tf); TF = {}
    for t, H in K.groupby("tf"):
        p = ALIAS.get(t, t); prim = list(H.unit[H.kind.isin(["all", "construct"])])
        if not prim: continue
        if t in dual: h1, h2 = [f"{t}|construct1"], [f"{t}|construct2"]
        else: h1, h2 = [f"{t}|half1"], [f"{t}|half2"]
        h1 = [x for x in h1 if x in u]; h2 = [x for x in h2 if x in u]
        TF.setdefault(p, dict(all=[], h1=[], h2=[], rows=[]))
        TF[p]["all"].append(vec(prim)); TF[p]["rows"].append(H[H.kind.isin(["all", "construct"])])
        if h1 and h2: TF[p]["h1"].append(vec(h1)); TF[p]["h2"].append(vec(h2))
    out = {}
    for p, D in TF.items():
        a = np.mean(D["all"], 0); h1 = np.mean(D["h1"], 0) if D["h1"] else None; h2 = np.mean(D["h2"], 0) if D["h2"] else None
        for o in OWN.get(p, [p]):
            if o in ui: a[ui[o]] = np.nan; (h1 is not None) and h1.__setitem__(ui[o], np.nan); (h2 is not None) and h2.__setitem__(ui[o], np.nan)
        R = pd.concat(D["rows"]); out[p] = dict(z=a, h1=h1, h2=h2, cells=int(R.n_used.sum()), ontarget_logfc=float(R.ontarget_logfc.mean()), rep_r=float(R.replicate_r.mean()))
    return out
V = {v: load(v) for v in VERS if os.path.exists(f"{MX}/{v}/kd_response.npz")}; print({v: len(d) for v, d in V.items()}, flush=True)
def dc(M): M = M - np.nanmean(M, 0, keepdims=True); return M - np.nanmean(M, 1, keepdims=True)
# ---- Part 2: per TF
P2 = []
for v, D in V.items():
    for t, r in D.items():
        rr = r["rep_r"]; sb = 2 * rr / (1 + rr) if np.isfinite(rr) and rr > -1 else np.nan
        nrep = int(((np.abs(r["h1"]) >= 2) & (np.abs(r["h2"]) >= 2) & (np.sign(r["h1"]) == np.sign(r["h2"]))).sum()) if r["h1"] is not None else np.nan
        P2.append(dict(label=LAB, version=v, tf=t, cells=r["cells"], ontarget_logfc=r["ontarget_logfc"], split_half_r=rr, reliability_SB=sb, reproducible_responders=nrep, strong_phenotype=t in STRONG))
P2 = pd.DataFrame(P2); P2.to_csv(f"{OUT}/part2_per_tf.tsv", sep="\t", index=False, float_format="%.4g")
W = P2.pivot_table(index="tf", columns="version", values=["cells", "ontarget_logfc", "reliability_SB", "reproducible_responders"])
rand = [c for c in VERS if c.startswith("random25") and c in V]
def boot_med(d): d = d.dropna().values; b = np.median(d[rng.integers(0, len(d), (2000, len(d)))], 1) if len(d) else [np.nan]; return float(np.median(d)) if len(d) else np.nan, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(d)
S2 = []
for m in ("ontarget_logfc", "reliability_SB", "reproducible_responders", "cells"):
    a, t25 = W[m].get("all_cell_min25"), W[m].get("top25_own"); r25 = W[m][rand].mean(1) if rand else None
    for nm, x, y in (("top25_own minus all cells", t25, a), ("top25_own minus random25 (mean of draws)", t25, r25), ("random25 minus all cells", r25, a)):
        if x is None or y is None: continue
        md, lo, hi, n = boot_med(x - y); S2.append(dict(label=LAB, metric=m, comparison=nm, n_tfs=n, median_diff=md, lo=lo, hi=hi))
S2 = pd.DataFrame(S2); S2.to_csv(f"{OUT}/part2_paired_summary.tsv", sep="\t", index=False, float_format="%.4g"); print(S2.drop(columns="label").round(4).to_string(index=False), flush=True)
# ---- Part 3: m16 simple model and m19 lists on all-cell vs top25_own (same lists)
Wsc = pd.read_csv(f"{RG}/m11_peak_regulome/out/wide_score_q0.05.tsv.gz", sep="\t", index_col=0).reindex(columns=UNIV).fillna(0)
IX = np.load(f"{RG}/m18_ml_v2/features/index.npz", allow_pickle=True); TT = list(IX["tfs"].astype(str))
def feats(vv):
    d = np.load(f"{RG}/m18_ml_v2/features/pair_{vv}.npz", allow_pickle=True); n = list(d["names"]); A = d["X"].reshape(len(TT), NG, len(n)); return {k: A[:, :, n.index(k)].astype(float) for k in ("B_prom_peak", "B_distal_abc_peak")}
FR, FS = feats("real"), feats("swap"); B1 = {t: (FR["B_prom_peak"][i] > 0) | (FR["B_distal_abc_peak"][i] > 0) for i, t in enumerate(TT)}; B2 = {t: FR["B_prom_peak"][i] > 0 for i, t in enumerate(TT)}
B1s = {t: (FS["B_prom_peak"][i] > 0) | (FS["B_distal_abc_peak"][i] > 0) for i, t in enumerate(TT)}; B2s = {t: FS["B_prom_peak"][i] > 0 for i, t in enumerate(TT)}
def top1000(t):
    if t not in Wsc.index: return None
    p = Wsc.loc[t].values; pos = np.flatnonzero(p > 0); o = pos[np.argsort(-p[pos], kind="stable")][:1000]; m = np.zeros(NG, bool); m[o] = True; return m
MIS = dict(zip([l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()], json.load(open(f"{RG}/m7_step2/features/raw_hicfixed/feature_qc.json"))["mis_paired_map"]["mis0"]))
gA = np.load(f"{RG}/m18_ml_v2/features/gene_A.npz", allow_pickle=True); an = list(gA["names"]); atac = gA["X"][:, an.index("A_prom_atac")]; cgi = gA["X"][:, an.index("A_cgi")]
strat = pd.qcut(pd.Series(atac).rank(method="first"), 10, labels=False).values * 2 + (cgi > 0); SI = {s: np.flatnonzero(strat == s) for s in np.unique(strat)}
def tab(b, r, ok): return np.array([(b & r & ok).sum(), (b & ~r & ok).sum(), (~b & r & ok).sum(), (~b & ~r & ok).sum()], float)
def orr(t): return (t[0] + .5) * (t[3] + .5) / ((t[1] + .5) * (t[2] + .5))
def boot_or(t): m = rng.multinomial(int(t.sum()), t / t.sum(), 2000).astype(float); o = (m[:, 0] + .5) * (m[:, 3] + .5) / ((m[:, 1] + .5) * (m[:, 2] + .5)); return np.percentile(o, [2.5, 97.5])
def matched(b, r, ok):
    cnt = {s: int((b & ok)[SI[s]].sum()) for s in SI}; res = []
    for _ in range(200):
        m = np.zeros(NG, bool)
        for s, c in cnt.items():
            if c: pool = SI[s][ok[SI[s]]]; m[rng.choice(pool, min(c, len(pool)), replace=False)] = True
        res.append(orr(tab(m, r, ok)))
    return float(np.median(res))
def mh(tabs):
    num = den = P_ = Q_ = S_ = 0.0
    for a_, b_, c_, d_ in tabs:
        N = a_ + b_ + c_ + d_
        if N == 0: continue
        rr_, ss = a_ * d_ / N, b_ * c_ / N; num += rr_; den += ss; p_, q_ = (a_ + d_) / N, (b_ + c_) / N; P_ += p_ * rr_; Q_ += p_ * ss + q_ * rr_; S_ += q_ * ss
    o = num / den; se = np.sqrt(P_ / (2 * num ** 2) + Q_ / (2 * num * den) + S_ / (2 * den ** 2)); return o, o * np.exp(-1.96 * se), o * np.exp(1.96 * se)
ROWS = []; POOL = []
for v in ("all_cell_min25", "top25_own"):
    if v not in V: continue
    for comp in ("m16_top1000_vs_top5pct", "m19_B1xP4", "m19_B2xP1"):
        T_ = {}
        for t, r in V[v].items():
            z = r["z"]; ok = np.isfinite(z)
            if comp.endswith("P4") or comp == "m19_B1xP4":
                if r["h1"] is None: continue
                resp_ = (np.abs(r["h1"]) >= 2) & (np.abs(r["h2"]) >= 2) & (np.sign(r["h1"]) == np.sign(r["h2"])) & ok
            else:
                a = np.abs(np.nan_to_num(z)); idx = np.flatnonzero(ok); resp_ = np.zeros(NG, bool); resp_[idx[np.argsort(-a[idx], kind="stable")[:int(round(len(idx) * .05))]]] = True
            if comp.startswith("m16"): b = top1000(t); bs = top1000(MIS.get(t, ""))
            elif comp == "m19_B1xP4": b, bs = B1.get(t), B1s.get(t)
            else: b, bs = B2.get(t), B2s.get(t)
            row = dict(label=LAB, version=v, comparison=comp, tf=t, strong_phenotype=t in STRONG, n_responders=int(resp_.sum()))
            if b is None or not b.any() or resp_.sum() < 20: row["status"] = "too few / no list"; ROWS.append(row); continue
            tb = tab(b, resp_, ok); T_[t] = tb; ci = boot_or(tb); row.update(status="ok", OR=orr(tb), OR_lo=ci[0], OR_hi=ci[1], OR_swapped=orr(tab(bs, resp_, ok)) if (bs is not None and bs.any()) else np.nan, OR_matched=matched(b, resp_, ok)); ROWS.append(row)
        for grp, names in (("strong10", STRONG), ("other", [t for t in V[v] if t not in STRONG])):
            tl = [T_[t] for t in names if t in T_]; o, lo, hi = mh(tl) if tl else (np.nan, np.nan, np.nan); POOL.append(dict(label=LAB, version=v, comparison=comp, group=grp, n_tfs=len(tl), MH_OR=o, lo=lo, hi=hi))
R3 = pd.DataFrame(ROWS); R3.to_csv(f"{OUT}/part3_per_tf.tsv", sep="\t", index=False, float_format="%.4g"); P3 = pd.DataFrame(POOL); P3.to_csv(f"{OUT}/part3_pooled.tsv", sep="\t", index=False, float_format="%.4g"); print(P3.drop(columns="label").round(3).to_string(index=False), flush=True)
# ---- observed_kd_top25 export for m22
t25 = V.get("top25_own", {}); EX = []
for t, r in t25.items():
    for k, g in enumerate(UNIV):
        if np.isfinite(r["z"][k]): EX.append((t, g, r["z"][k], r["h1"][k] if r["h1"] is not None else np.nan, r["h2"][k] if r["h2"] is not None else np.nan))
E = pd.DataFrame(EX, columns=["tf", "gene", "z_top25", "z_top25_half1", "z_top25_half2"]); E["reproducible_top25"] = (E.z_top25_half1.abs() >= 2) & (E.z_top25_half2.abs() >= 2) & (np.sign(E.z_top25_half1) == np.sign(E.z_top25_half2)); E.to_csv(f"{OUT}/observed_kd_top25.tsv.gz", sep="\t", index=False, float_format="%.4g")
# ---- Amendment 1: m14 positive control on all-cell, top25_own, random25_d0
C = pd.read_csv(f"{X}/complex_screen/candidates_frozen_v1/candidates_all3081.tsv", sep="\t"); known = {tuple(sorted((a.upper(), b.upper()))) for a, b, k in zip(C.tf_a, C.tf_b, C.known) if k}
NAMED = {tuple(sorted(p)) for p in [("RBBP4", "RBBP7"), ("MBD3", "RBBP4"), ("MBD3", "RBBP7"), ("EZH2", "MTF2"), ("PCGF6", "EZH2"), ("PCGF6", "MTF2"), ("MEF2A", "MEF2D"), ("NFYA", "DR1"), ("CTCF", "STAG2"), ("HP1", "SETDB1")]}
FL = pd.read_csv(f"{RG}/m6_improve/repro_v1/per_tf_floor_test.tsv", sep="\t").set_index("tf"); FT = pd.read_csv(f"{RG}/m7_step2/splits/tf_folds.tsv", sep="\t").set_index("TF").ceiling
def sp(a, b): m = np.isfinite(a) & np.isfinite(b); return np.nan if m.sum() < 50 else float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1])
PC = []; PER = []
for v in ("all_cell_min25", "top25_own", "random25_d0"):
    if v not in V: continue
    tfs = sorted(t for t, r in V[v].items() if r["h1"] is not None); H1 = dc(np.vstack([V[v][t]["h1"] for t in tfs])); H2 = dc(np.vstack([V[v][t]["h2"] for t in tfs]))
    rows = []
    for i in range(len(tfs)):
        for j in range(i + 1, len(tfs)):
            rows.append(dict(a=tfs[i], b=tfs[j], K=np.nanmean([sp(H1[i], H2[j]), sp(H2[i], H1[j])])))
    P = pd.DataFrame(rows); P["key"] = [tuple(sorted((a, b))) for a, b in zip(P.a, P.b)]; P["ctl"] = P.key.isin(NAMED | known)
    lc = np.log10(FL.cells.reindex(tfs).values.astype(float)); ct = dict(zip(tfs, pd.qcut(pd.Series(lc), 3, labels=False).values)); rl = FT.reindex(tfs); rt = {t: (int(np.digitize(rl[t], [.1, .3])) if np.isfinite(rl[t]) else -1) for t in tfs}
    P["stratum"] = [str(tuple(sorted((rt[a], rt[b])))) + str(tuple(sorted((int(ct[a]), int(ct[b]))))) for a, b in zip(P.a, P.b)]
    ctl = P[P.ctl & P.K.notna()]; rest = P[~P.ctl & P.K.notna()]; pools = {s: g.K.values for s, g in rest.groupby("stratum")}; draws = np.zeros(10000)
    for s, g in ctl.groupby("stratum"): draws += rng.choice(pools.get(s, rest.K.values), (10000, len(g))).sum(1)
    draws /= len(ctl); obs = ctl.K.mean(); bt = [ctl.K.values[rng.integers(0, len(ctl), len(ctl))].mean() for _ in range(2000)]
    PC.append(dict(label=LAB, version=v, n_tfs=len(tfs), n_control_pairs=len(ctl), mean_K_control=obs, mean_K_matched=draws.mean(), diff=obs - draws.mean(), lo=np.percentile(bt, 2.5) - draws.mean(), hi=np.percentile(bt, 97.5) - draws.mean(), p=(1 + (draws >= obs).sum()) / 10001))
    for k in NAMED:
        m = P[P.key == k]; PER.append(dict(version=v, pair="-".join(k), K=float(m.K.iloc[0]) if len(m) else np.nan))
PCd = pd.DataFrame(PC); PCd.to_csv(f"{OUT}/positive_control_versions.tsv", sep="\t", index=False, float_format="%.4g"); pd.DataFrame(PER).pivot(index="pair", columns="version", values="K").to_csv(f"{OUT}/positive_control_named_pairs.tsv", sep="\t", float_format="%.4f")
print(PCd.drop(columns="label").round(4).to_string(index=False)); print(pd.DataFrame(PER).pivot(index="pair", columns="version", values="K").round(3).to_string())
t25p = PCd[PCd.version == "top25_own"]; r25p = PCd[PCd.version == "random25_d0"]
print("PAIR-LEVEL RERUN WARRANTED:", bool(len(t25p) and t25p.p.iloc[0] < .05 and len(r25p) and t25p["diff"].iloc[0] > r25p["diff"].iloc[0]))
print("done")
