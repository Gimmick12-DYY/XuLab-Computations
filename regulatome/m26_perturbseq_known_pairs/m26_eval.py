# m26 evaluation (SPEC.md): Part 1 cohesin/CTCF knockdown facts and pair similarity under two definitions; Part 2 known-pair control the authors' way; Part 3 named-pair diagnosis. Tables only. Diagnostic.
import numpy as np, pandas as pd
from scipy.stats import rankdata
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m26_perturbseq_known_pairs"; OUT = f"{O}/out"; rng = np.random.default_rng(20261113); LAB = "diagnostic"
ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; COH = ["RAD21", "SMC3", "STAG1", "NIPBL", "WAPL", "PDS5A", "PDS5B", "MAU2", "CTCFL"]; FOCAL = ["CTCF", "STAG2"] + COH
STRONG = ["CTCF", "DNMT1", "E4F1", "MEF2A", "NRF1", "PA2G4", "RBBP7", "RBPJ", "SAFB", "SETDB1"]
# ---- definition (i): kd_response units (arm 2), halves double-centred, cross-half Spearman
d = np.load(f"{O}/out/matrices/arm2_all_cell/kd_response.npz", allow_pickle=True); q = pd.read_csv(f"{O}/out/matrices/arm2_all_cell/guide_qc.tsv", sep="\t"); u = list(d["units"].astype(str)); ui = {x: k for k, x in enumerate(u)}
keep = d["ntc_detect"] >= .1; G = d["genes"].astype(str); Z = d["z"][:, keep].astype(float); Gk = G[keep]; K = q[q.kept == 1]; dual = set(K[K.kind == "construct"].tf)
def rows(kind):
    out = {}
    for t, H in K.groupby("tf"):
        if t in dual: names = [f"{t}|construct1"] if kind == "h1" else ([f"{t}|construct2"] if kind == "h2" else list(H.unit[H.kind == "construct"]))
        else: names = [f"{t}|half1"] if kind == "h1" else ([f"{t}|half2"] if kind == "h2" else [f"{t}|all"])
        names = [x for x in names if x in ui]
        if names: out.setdefault(ALIAS.get(t, t), []).append(np.nanmean(Z[[ui[x] for x in names]], 0))
    return {p: np.mean(v, 0) for p, v in out.items()}
A_, H1, H2 = rows("all"), rows("h1"), rows("h2"); PT = sorted(set(H1) & set(H2)); gk = {g: k for k, g in enumerate(Gk)}
def dc(M): M = M - np.nanmean(M, 0, keepdims=True); return M - np.nanmean(M, 1, keepdims=True)
M1, M2 = dc(np.vstack([H1[p] for p in PT])), dc(np.vstack([H2[p] for p in PT])); OWNG = {p: [gk[o] for o in {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}.get(p, [p]) if o in gk] for p in PT}
def sp(a, b): m = np.isfinite(a) & np.isfinite(b); return np.nan if m.sum() < 50 or np.std(a[m]) == 0 or np.std(b[m]) == 0 else float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1])
def Ki(a, b):
    i, j = PT.index(a), PT.index(b); m = np.ones(M1.shape[1], bool); m[OWNG[a] + OWNG[b]] = False; return np.nanmean([sp(M1[i, m], M2[j, m]), sp(M2[i, m], M1[j, m])])
# ---- definition (ii): authors' z (all NTC mean/sd), mean per perturbation, 2,000 HVG (seurat_v3-like on NTC raw counts), Spearman
pc = np.load(f"{O}/work/profiles_cells.npz", allow_pickle=True); perts = [ALIAS.get(p, p) for p in pc["perts"].astype(str)]; gsym = pc["genes"].astype(str); mu, sdv = pc["ntc_mean"], pc["ntc_sd"]
rm, rv = pc["raw_mean"], pc["raw_var"]; ok = rm > 0; lm, lv = np.log10(rm[ok]), np.log10(np.maximum(rv[ok], 1e-12)); b_ = pd.qcut(lm, 50, labels=False, duplicates="drop"); tr = pd.Series(lv).groupby(b_).transform("mean").values
score = np.full(len(rm), -np.inf); score[np.flatnonzero(ok)] = lv - tr; HVG = np.argsort(-score)[:2000]
ZM = (pc["mean_all"][:, HVG] - mu[HVG]) / np.where(sdv[HVG] > 0, sdv[HVG], np.nan); PZ = {}
for k, p in enumerate(perts): PZ.setdefault(p, []).append(ZM[k])
PZ = {p: np.nanmean(v, 0) for p, v in PZ.items()}; NC = {}
for k, p in enumerate(perts): NC[p] = NC.get(p, 0) + int(pc["n"][k])
def Kii(a, b): return sp(PZ[a], PZ[b]) if a in PZ and b in PZ else np.nan
# ---- Part 1 table
own = dict(zip(perts, pc["own_cp10k"])); ntc_cp = dict(zip(gsym, pc["ntc_cp10k"])); det = dict(zip(G, d["ntc_detect"]))
SP = pd.read_csv(f"{RG}/m16_strong_phenotype/out/strong_phenotype_per_tf.tsv", sep="\t").set_index("panel_tf").status
P1 = []
for g in FOCAL:
    H = q[(q.tf == g) & q.kind.isin(["all", "construct"])]; rr = H.replicate_r.mean(); kdp = 100 * (1 - own.get(g, np.nan) / ntc_cp.get(g, np.nan)) if ntc_cp.get(g, 0) else np.nan
    nrep = int(((np.abs(H1[g]) >= 2) & (np.abs(H2[g]) >= 2) & (np.sign(H1[g]) == np.sign(H2[g]))).sum()) if g in H1 and g in H2 else np.nan
    P1.append(dict(label=LAB, gene=g, cells=int(H.n_cells_all.sum()) if len(H) else 0, min_cells_per_guide_pair=int(H.n_cells_all.min()) if len(H) else 0, knockdown_pct_vs_all_NTC=kdp, ontarget_logfc_gem_matched=float(H.ontarget_logfc.mean()) if len(H) else np.nan,
                   target_ntc_mean_cp10k=ntc_cp.get(g, np.nan), target_ntc_detect_frac=det.get(g, np.nan), crit_ge25_cells=bool(H.n_cells_all.sum() >= 25), crit_ge30pct_kd=bool(kdp >= 30) if np.isfinite(kdp) else False,
                   classifier=("m16 rebuild: " + SP.get(g, "")) if g in SP.index else "not run (not a panel TF)", split_half_r=rr, reliability_SB=(2 * rr / (1 + rr)) if np.isfinite(rr) else np.nan, reproducible_responders=nrep, in_halves_matrix=g in PT))
P1 = pd.DataFrame(P1); P1.to_csv(f"{OUT}/part1_cohesin_ctcf_knockdown.tsv", sep="\t", index=False, float_format="%.4g"); print(P1.drop(columns="label").round(3).to_string(index=False), flush=True)
# pair similarity among focal genes, both definitions, vs matched random pairs (cell-number tertile)
allp = sorted(set(PT) & set(PZ)); ct = pd.qcut(pd.Series([NC.get(p, 0) for p in allp], index=allp).rank(method="first"), 3, labels=False).to_dict()
foc = [g for g in FOCAL if g in allp]; FP = [(a, b) for i, a in enumerate(foc) for b in foc[i + 1:]]
PR = []
for a, b in FP: PR.append(dict(label=LAB, pair=f"{a}-{b}", K_i=Ki(a, b), K_ii=Kii(a, b), involves_CTCF="CTCF" in (a, b)))
PR = pd.DataFrame(PR); rand = [(a, b) for i, a in enumerate(allp) for b in allp[i + 1:] if not (a in foc and b in foc)]
RS = pd.DataFrame([dict(a=a, b=b, s=tuple(sorted((ct[a], ct[b])))) for a, b in rand]); RS = RS.sample(min(3000, len(RS)), random_state=1); RS["K_i"] = [Ki(a, b) for a, b in zip(RS.a, RS.b)]; RS["K_ii"] = [Kii(a, b) for a, b in zip(RS.a, RS.b)]
SUMP = []
for col in ("K_i", "K_ii"):
    for nm, sub in (("all focal pairs", PR), ("CTCF with cohesin/STAG2", PR[PR.involves_CTCF]), ("cohesin/STAG2 among themselves", PR[~PR.involves_CTCF])):
        obs = sub[col].mean(); st_ = [tuple(sorted((ct[x.split("-")[0]], ct[x.split("-")[1]]))) for x in sub.pair]; draws = []
        for _ in range(5000):
            v = [rng.choice(RS[RS.s == s][col].dropna().values) if (RS.s == s).any() else np.nan for s in st_]; draws.append(np.nanmean(v))
        draws = np.array(draws); SUMP.append(dict(label=LAB, definition=col, set=nm, n_pairs=len(sub), mean_similarity=obs, matched_random_mean=float(np.nanmean(draws)), p=float((1 + (draws >= obs).sum()) / (len(draws) + 1))))
PR.to_csv(f"{OUT}/part1_focal_pairs.tsv", sep="\t", index=False, float_format="%.4g"); SUMP = pd.DataFrame(SUMP); SUMP.to_csv(f"{OUT}/part1_focal_pairs_vs_random.tsv", sep="\t", index=False, float_format="%.4g")
print(PR.drop(columns="label").round(3).to_string(index=False)); print(SUMP.drop(columns="label").round(4).to_string(index=False), flush=True)
# ---- Part 2: m14 positive control with definition (ii), panel TFs only
C = pd.read_csv(f"{X}/complex_screen/candidates_frozen_v1/candidates_all3081.tsv", sep="\t"); known = {tuple(sorted((a.upper(), b.upper()))) for a, b, k in zip(C.tf_a, C.tf_b, C.known) if k}
NAMED = {tuple(sorted(p)) for p in [("RBBP4", "RBBP7"), ("MBD3", "RBBP4"), ("MBD3", "RBBP7"), ("EZH2", "MTF2"), ("PCGF6", "EZH2"), ("PCGF6", "MTF2"), ("MEF2A", "MEF2D"), ("NFYA", "DR1"), ("CTCF", "STAG2"), ("HP1", "SETDB1")]}
FL = pd.read_csv(f"{RG}/m6_improve/repro_v1/per_tf_floor_test.tsv", sep="\t").set_index("tf"); FT = pd.read_csv(f"{RG}/m7_step2/splits/tf_folds.tsv", sep="\t").set_index("TF").ceiling
tfs79 = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]; panel = [t for t in tfs79 if t in PZ]
R2 = []
for nm, tl in (("strong-phenotype TFs (m16 rebuild)", [t for t in STRONG if t in PZ]), ("all panel TFs with cells", panel)):
    pairs = [(a, b) for i, a in enumerate(tl) for b in tl[i + 1:]]; P = pd.DataFrame(dict(a=[x for x, _ in pairs], b=[y for _, y in pairs])); P["K"] = [Kii(a, b) for a, b in pairs]; P["key"] = [tuple(sorted(p)) for p in pairs]; P["ctl"] = P.key.isin(NAMED | known)
    lc = np.log10(FL.cells.reindex(tl).astype(float)); ctt = dict(zip(tl, pd.qcut(lc.rank(method="first"), 3, labels=False).values)); rl = FT.reindex(tl); rt = {t: (int(np.digitize(rl[t], [.1, .3])) if np.isfinite(rl[t]) else -1) for t in tl}
    P["stratum"] = [str(tuple(sorted((rt[a], rt[b])))) + str(tuple(sorted((ctt[a], ctt[b])))) for a, b in zip(P.a, P.b)]
    ctl = P[P.ctl & P.K.notna()]; rest = P[~P.ctl & P.K.notna()]
    if not len(ctl): R2.append(dict(label=LAB, tf_set=nm, n_tfs=len(tl), n_control_pairs=0)); continue
    pools = {s: g.K.values for s, g in rest.groupby("stratum")}; draws = np.zeros(10000)
    for s, g in ctl.groupby("stratum"): draws += rng.choice(pools.get(s, rest.K.values), (10000, len(g))).sum(1)
    draws /= len(ctl); obs = ctl.K.mean(); bt = [ctl.K.values[rng.integers(0, len(ctl), len(ctl))].mean() for _ in range(2000)]
    R2.append(dict(label=LAB, tf_set=nm, n_tfs=len(tl), n_control_pairs=len(ctl), mean_K_control=obs, mean_K_matched=float(draws.mean()), diff=obs - draws.mean(), lo=np.percentile(bt, 2.5) - draws.mean(), hi=np.percentile(bt, 97.5) - draws.mean(), p=float((1 + (draws >= obs).sum()) / 10001), control_pairs=";".join(f"{a}-{b}:{k:.3f}" for a, b, k in zip(ctl.a, ctl.b, ctl.K))))
R2 = pd.DataFrame(R2); R2.to_csv(f"{OUT}/part2_positive_control_authors_def.tsv", sep="\t", index=False, float_format="%.4g"); print(R2.drop(columns=["label", "control_pairs"]).round(4).to_string(index=False), flush=True)
R1 = pd.read_csv(f"{RG}/r1_tftf/v2_final/r1_pairs_frozen.tsv", sep="\t", keep_default_na=False, na_values=[""]); R1["key"] = [tuple(sorted((str(a).upper(), str(b).upper()))) for a, b in zip(R1.a, R1.b)]; strg = dict(zip(R1.key, pd.to_numeric(R1.string_physical, errors="coerce")))
pairs = [(a, b) for i, a in enumerate(panel) for b in panel[i + 1:]]; P = pd.DataFrame(dict(pair=[f"{a}-{b}" for a, b in pairs], K=[Kii(a, b) for a, b in pairs], string_physical=[strg.get(tuple(sorted((a, b))), np.nan) for a, b in pairs]))
P = P[P.K.notna()]; P["bin"] = pd.cut(P.K, 4); SB = P.groupby("bin", observed=True).agg(n_pairs=("K", "size"), mean_string_physical=("string_physical", "mean"), n_with_string=("string_physical", lambda v: int(v.notna().sum())), frac_string_ge_0_7=("string_physical", lambda v: float((v >= .7).mean()))).reset_index()
SB["bin"] = SB["bin"].astype(str); SB.insert(0, "label", LAB); SB.to_csv(f"{OUT}/part2_string_vs_correlation_bins.tsv", sep="\t", index=False, float_format="%.4g"); print(SB.drop(columns="label").to_string(index=False), flush=True)
# ---- Part 3: named pairs
qp = pd.read_csv(f"{RG}/m4_validation/kd_response/out_v2/arm2_all_cell/guide_qc.tsv", sep="\t"); qp = qp[qp.kind.isin(["all", "construct"])]; qp["p"] = qp.tf.map(lambda t: ALIAS.get(t, t))
stt = pd.read_csv(f"{RG}/phase1/panel_tf_xatlas_status.tsv", sep="\t").groupby("panel_tf").knockdown_pct.min(); RC = pd.read_csv(f"{RG}/m19_binary_matrices/out/responder_counts.tsv", sep="\t", index_col=0)["P4_reproducible_halves|all"]
def member(t): return dict(cells=int(qp[qp.p == t].n_cells_all.sum()), knockdown_pct=float(stt.get(t, np.nan)), reliability=float(FT.get(t, np.nan)), strong=SP.get(t, ""), reproducible_responders=float(RC.get(t, np.nan)))
P3 = []
for a, b in sorted(NAMED):
    ma, mb = member(a), member(b); det = lambda m: (m["strong"] == "strong phenotype") or (np.isfinite(m["reproducible_responders"]) and m["reproducible_responders"] >= 20)
    P3.append(dict(label=LAB, pair=f"{a}-{b}", **{f"{a and 'A'}_{k}": v for k, v in ma.items()}, **{f"B_{k}": v for k, v in mb.items()}, both_detectable=bool(det(ma) and det(mb)), K_i=Ki(a, b) if a in PT and b in PT else np.nan, K_ii=Kii(a, b)))
P3 = pd.DataFrame(P3); P3.to_csv(f"{OUT}/part3_named_pairs.tsv", sep="\t", index=False, float_format="%.4g"); print(P3.drop(columns="label").round(3).to_string(index=False))
print("detectable = m16 rebuilt strong phenotype, or >= 20 reproducible responders (|z| >= 2 in both halves, same sign)"); print("done")
