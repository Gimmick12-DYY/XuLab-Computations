# m14_kd_profile_similarity (SPEC.md): binding similarity vs knockdown-profile similarity over TF pairs. Diagnostic, benchmark-developed.
import hashlib, json, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, mannwhitneyu
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
import glob as _g, matplotlib.font_manager as _fm
for _f in _g.glob("/work/users/d/y/dyy12/fonts/arial/*.ttf"): _fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none", "font.size": 6, "axes.linewidth": .4})
X_ = "/work/users/d/y/dyy12/XuLab"; RG_ = f"{X_}/regulatome"; O_ = f"{RG_}/m14_kd_profile_similarity"; OUT_ = f"{O_}/out"; CS = f"{X_}/complex_screen"; LAB_ = "diagnostic, benchmark-developed"; rng_ = np.random.default_rng(20261112); NP = 10000
STRONG = ["CTCF", "DNMT1", "E4F1", "MEF2A", "NRF1", "PA2G4", "RBBP7", "RBPJ", "SAFB", "SETDB1"]
sys.argv = ["m15.py", "dc"]; src = open(f"{RG_}/m15_gene_to_tf/m15.py").read(); exec(src[:src.index("# ---- binding percentiles")])
D1, D2, DA = Z1.copy(), Z2.copy(), ZA.copy(); TFk = list(TF); nG = len(UG)
sys.argv = ["m15.py", "raw"]; exec(src[:src.index("# ---- binding percentiles")]); R1, R2 = Z1.copy(), Z2.copy(); assert list(TF) == TFk
TF = TFk; nT = len(TF); ug = {g: k for k, g in enumerate(UG)}; OWNG = {t: [ug[o] for o in OWN.get(t, [t]) if o in ug] for t in TF}
var = np.nanvar(DA, axis=0); top2k = np.argsort(-np.nan_to_num(var))[:2000]
def rk(v): out = np.full(len(v), np.nan); m = np.isfinite(v); out[m] = rankdata(v[m]); return out
def sp(a, b): m = np.isfinite(a) & np.isfinite(b); return np.nan if m.sum() < 50 else float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1])
def Kmat(cols):
    K = np.full((nT, nT), np.nan)
    for i in range(nT):
        for j in range(i + 1, nT):
            m = np.zeros(nG, bool); m[cols] = True; m[OWNG[TF[i]] + OWNG[TF[j]]] = False
            K[i, j] = K[j, i] = np.mean([sp(D1[i, m], D2[j, m]), sp(D2[i, m], D1[j, m])])
    return K
KM = {"all_genes": Kmat(np.arange(nG)), "top2000_var": Kmat(top2k)}; print("K done", flush=True)
# binding similarity
PCT = {k: f"{CS}/af3_corr060_20261104/align_corr_20261109/inputs/pct_{k}.tsv" for k in ("all", "noquies", "tss", "enhancer_biv", "repressive")}
SHA = {k: hashlib.sha256(open(v, "rb").read()).hexdigest() for k, v in PCT.items()}; json.dump(dict(files=PCT, sha256=SHA), open(f"{OUT_}/binding_matrix_sources.json", "w"), indent=1)
def ld(f): m = pd.read_csv(f, sep="\t", index_col=0); m.index = m.index.str.strip().str.upper(); m.columns = m.columns.str.strip().str.upper(); return m.reindex(index=TF, columns=TF).values.astype(float)
BM = {k: ld(v) for k, v in PCT.items()}; BM = {"mean5": np.nanmean(np.stack(list(BM.values())), 0), **{f"pct_{k}": v for k, v in BM.items()}}
print("B missing TFs:", [t for i, t in enumerate(TF) if np.isnan(BM["mean5"][i]).all()], flush=True)
FL = pd.read_csv(f"{RG_}/m6_improve/repro_v1/per_tf_floor_test.tsv", sep="\t").set_index("tf"); FT = pd.read_csv(f"{RG_}/m7_step2/splits/tf_folds.tsv", sep="\t").set_index("TF").ceiling
cells = FL.cells.reindex(TF).values.astype(float); lc = np.log10(cells); rel = FT.reindex(TF).values.astype(float); tert = FL.tertile.reindex(TF).values
SETS = {"primary_rel_ge_0.2": [t for t in TF if FT.get(t, -1) >= .2], "strong10": [t for t in STRONG if t in TF], "all76": TF}
json.dump({k: v for k, v in SETS.items()}, open(f"{OUT_}/tf_sets.json", "w"), indent=1); print({k: len(v) for k, v in SETS.items()}, SETS["primary_rel_ge_0.2"], flush=True)
iu = np.triu_indices(nT, 1); PAIRS = pd.DataFrame(dict(a=np.array(TF)[iu[0]], b=np.array(TF)[iu[1]], i=iu[0], j=iu[1]))
for k, M in KM.items(): PAIRS[f"K_{k}"] = M[iu]
for k, M in BM.items(): PAIRS[f"B_{k}"] = M[iu]
PAIRS["key"] = [tuple(sorted((a, b))) for a, b in zip(PAIRS.a, PAIRS.b)]
# annotations
C = pd.read_csv(f"{CS}/candidates_frozen_v1/candidates_all3081.tsv", sep="\t"); C["key"] = [tuple(sorted((a.upper(), b.upper()))) for a, b in zip(C.tf_a, C.tf_b)]; CI = C.set_index("key")
PAIRS["known"] = PAIRS.key.map(CI.known).fillna(False).astype(bool); PAIRS["known_reason"] = PAIRS.key.map(CI.known_reason).fillna("")
PAIRS["candidate"] = PAIRS.key.map(CI.candidate).fillna(False).astype(bool); PAIRS["best_pct"] = PAIRS.key.map(CI.best_pct)
PAIRS["capselex"] = PAIRS.known_reason.str.contains("CAP-SELEX", case=False)
T5 = pd.read_csv(f"{CS}/af3_corr060_20261104/rescore_5seed/tiers_5seed.tsv", sep="\t"); T5["key"] = [tuple(sorted(p.upper().split("-"))) for p in T5.panel_pair]; PAIRS["af3_5seed"] = PAIRS.key.map(T5.set_index("key").tier_5seed).fillna("none / not run")
DN = pd.read_csv(f"{CS}/af3_dna_included_20261111/results.tsv", sep="\t"); DN["key"] = [tuple(sorted(p.upper().split("_"))) for p in DN.pair]; PAIRS["dna_displaced"] = PAIRS.key.isin(set(DN.key[DN.outcome == "DISPLACED"]))
MP = pd.read_csv(f"{X_}/cobinding/results/ourrun_k50_clique_hubs_blacklist_removed_loop_edges/tf_pair_shared_regions.tsv", sep="\t"); MP["key"] = [tuple(sorted((a.upper(), b.upper()))) for a, b in zip(MP.tf1, MP.tf2)]
PAIRS["shared_module"] = PAIRS.key.isin(set(MP.key[MP.p < .05]))
PAIRS["rel_tert_pair"] = [tuple(sorted((int(np.digitize(rel[i], [.1, .3])) if np.isfinite(rel[i]) else -1, int(np.digitize(rel[j], [.1, .3])) if np.isfinite(rel[j]) else -1))) for i, j in zip(PAIRS.i, PAIRS.j)]
ct = pd.qcut(pd.Series(lc), 3, labels=False).values; PAIRS["cell_tert_pair"] = [tuple(sorted((int(ct[i]), int(ct[j])))) for i, j in zip(PAIRS.i, PAIRS.j)]
PAIRS["stratum"] = [str(a) + str(b) for a, b in zip(PAIRS.rel_tert_pair, PAIRS.cell_tert_pair)]
# 3. positive control
NAMED = [("RBBP4", "RBBP7"), ("MBD3", "RBBP4"), ("MBD3", "RBBP7"), ("EZH2", "MTF2"), ("PCGF6", "EZH2"), ("PCGF6", "MTF2"), ("MEF2A", "MEF2D"), ("NFYA", "DR1"), ("CTCF", "STAG2"), ("HP1", "SETDB1")]
named = {tuple(sorted(p)) for p in NAMED}; PAIRS["named_control"] = PAIRS.key.isin(named)
PC = []
for nm, mask in (("named + known", PAIRS.named_control | PAIRS.known), ("named + known, without NEUROD4-rule pairs", (PAIRS.named_control | PAIRS.known) & ~((PAIRS.a == "NEUROD4") | (PAIRS.b == "NEUROD4")) | PAIRS.named_control), ("named only", PAIRS.named_control)):
    for kk in KM:
        col = f"K_{kk}"; ctl = PAIRS[mask & PAIRS[col].notna()]; rest = PAIRS[~mask & PAIRS[col].notna()]; pools = {s: g[col].values for s, g in rest.groupby("stratum")}
        draws = np.zeros(NP)
        for s, g in ctl.groupby("stratum"):
            pool = pools.get(s, rest[col].values); draws += rng_.choice(pool, (NP, len(g))).sum(1)
        draws /= len(ctl); obs = ctl[col].mean(); bt = [ctl[col].values[rng_.integers(0, len(ctl), len(ctl))].mean() for _ in range(2000)]
        PC.append(dict(label=LAB_, control_set=nm, K=kk, n_pairs=len(ctl), mean_K_control=obs, mean_K_matched_random=draws.mean(), diff=obs - draws.mean(), ci_lo=np.percentile(bt, 2.5) - draws.mean(), ci_hi=np.percentile(bt, 97.5) - draws.mean(), p_perm=(1 + (draws >= obs).sum()) / (NP + 1), pairs=";".join(f"{a}-{b}:{v:.3f}" for a, b, v in zip(ctl.a, ctl.b, ctl[col]))))
PCd = pd.DataFrame(PC); PCd.to_csv(f"{OUT_}/positive_control.tsv", sep="\t", index=False, float_format="%.4g"); print(PCd.drop(columns=["label", "pairs"]).to_string(index=False), flush=True)
VERDICT = "PASS" if PCd[(PCd.control_set == "named + known") & (PCd.K == "all_genes")].p_perm.iloc[0] < .05 else "FAIL (verdicts below are UNINFORMATIVE)"; print("positive control:", VERDICT, flush=True)
# 4. Mantel
def mantel(Bm, Km, idx, cov=None):
    Bs, Ks = Bm[np.ix_(idx, idx)], Km[np.ix_(idx, idx)]; u = np.triu_indices(len(idx), 1); b, k = Bs[u], Ks[u]; ok = np.isfinite(b) & np.isfinite(k)
    if cov is not None:
        Cv = np.column_stack([c[np.ix_(idx, idx)][u] for c in cov] + [np.ones(len(b))]); ok &= np.isfinite(Cv).all(1)
    b0, k0 = rankdata(b[ok]), rankdata(k[ok])
    def res(v, Cm): beta = np.linalg.lstsq(Cm, v, rcond=None)[0]; return v - Cm @ beta
    if cov is not None: Cok = Cv[ok]; b0 = res(b0, Cok); k0r = res(k0, Cok)
    else: k0r = k0
    obs = np.corrcoef(b0, k0r)[0, 1]; null = np.zeros(NP); n = len(idx)
    for p in range(NP):
        pr = rng_.permutation(n); kp = Ks[np.ix_(pr, pr)][u][ok]; kp = rankdata(kp)
        if cov is not None: kp = res(kp, Cok)
        null[p] = np.corrcoef(b0, kp)[0, 1]
    return obs, (1 + (np.abs(null) >= abs(obs)).sum()) / (NP + 1), int(ok.sum())
LCp = np.add.outer(lc, lc); LCd = np.abs(np.subtract.outer(lc, lc)); ST = (tert[:, None] == tert[None, :]).astype(float); RLd = np.abs(np.subtract.outer(rel, rel)); RLp = np.multiply.outer(rel, rel)
DEP = -LCd; MAN = []
for sn, tl in SETS.items():
    idx = [TF.index(t) for t in tl]
    for kk, Km in KM.items():
        for bk, Bm in BM.items():
            r, p, n = mantel(Bm, Km, idx); rp, pp, _ = mantel(Bm, Km, idx, [LCp, LCd, ST, RLd, RLp])
            MAN.append(dict(label=LAB_, tf_set=sn, n_tfs=len(idx), K=kk, B=bk, n_pairs=n, mantel_rho=r, p=p, partial_mantel_rho=rp, partial_p=pp))
        for dn, Dm in (("depth: -|dlog cells|", DEP), ("depth: log cells product", LCp)):
            r, p, n = mantel(Dm, Km, idx); MAN.append(dict(label=LAB_, tf_set=sn, n_tfs=len(idx), K=kk, B=f"control {dn}", n_pairs=n, mantel_rho=r, p=p))
    print(sn, "mantel done", flush=True)
MANd = pd.DataFrame(MAN); MANd.to_csv(f"{OUT_}/mantel.tsv", sep="\t", index=False, float_format="%.4g"); print(MANd.drop(columns="label").round(4).to_string(index=False), flush=True)
# 5. secondary
SEC = []
def cmp(name, a_mask, b_mask, col="K_all_genes", strat=False):
    A_, B_ = PAIRS[a_mask & PAIRS[col].notna()], PAIRS[b_mask & PAIRS[col].notna()]
    if not len(A_) or not len(B_): SEC.append(dict(label=LAB_, comparison=name, n_a=len(A_), n_b=len(B_))); return
    if strat:
        ds, ws = [], []
        for s, g in A_.groupby("cell_tert_pair"):
            h = B_[B_.cell_tert_pair == s]
            if len(h): ds.append(g[col].mean() - h[col].mean()); ws.append(len(g))
        d = float(np.average(ds, weights=ws)) if ds else np.nan
    else: d = A_[col].mean() - B_[col].mean()
    SEC.append(dict(label=LAB_, comparison=name, n_a=len(A_), n_b=len(B_), mean_K_a=A_[col].mean(), mean_K_b=B_[col].mean(), diff=d, mannwhitney_p=mannwhitneyu(A_[col], B_[col]).pvalue))
cmp("shared coordinated-binding module regions (p < 0.05) vs other pairs", PAIRS.shared_module, ~PAIRS.shared_module)
cmp("frozen candidates vs non-candidates (best_pct < 0.60), cell-tertile stratified", PAIRS.candidate, PAIRS.best_pct < .6, strat=True)
for t_ in ("strong", "weak"): cmp(f"AF3 5-seed {t_} vs none / not run", PAIRS.af3_5seed == t_, PAIRS.af3_5seed == "none / not run")
cmp("DNA-test DISPLACED vs other AF3-strong", PAIRS.dna_displaced, (PAIRS.af3_5seed == "strong") & ~PAIRS.dna_displaced)
cmp("CAP-SELEX called vs rest", PAIRS.capselex, ~PAIRS.capselex)
SECd = pd.DataFrame(SEC); SECd.to_csv(f"{OUT_}/secondary.tsv", sep="\t", index=False, float_format="%.4g"); print(SECd.drop(columns="label").round(4).to_string(index=False), flush=True)
# 6. direction check
rep = (np.abs(R1) >= 2) & (np.abs(R2) >= 2) & (np.sign(R1) == np.sign(R2)); sgn = np.sign(R1)
P_ = PAIRS[PAIRS.K_all_genes.notna() & PAIRS.B_mean5.notna()].copy(); P_["rank_sum"] = P_.B_mean5.rank(ascending=False) + P_.K_all_genes.rank(ascending=False); TOP = P_.nsmallest(20, "rank_sum").copy()
TOP["n_shared_responders"] = [int((rep[i] & rep[j]).sum()) for i, j in zip(TOP.i, TOP.j)]; TOP["frac_same_direction"] = [float((sgn[i][rep[i] & rep[j]] == sgn[j][rep[i] & rep[j]]).mean()) if (rep[i] & rep[j]).any() else np.nan for i, j in zip(TOP.i, TOP.j)]
TOP[["a", "b", "B_mean5", "K_all_genes", "n_shared_responders", "frac_same_direction", "known", "known_reason", "af3_5seed", "shared_module"]].to_csv(f"{OUT_}/top20_pairs.tsv", sep="\t", index=False, float_format="%.4g")
PAIRS.drop(columns=["key"]).to_csv(f"{OUT_}/pairs.tsv", sep="\t", index=False, float_format="%.4g"); print(TOP[["a", "b", "B_mean5", "K_all_genes", "n_shared_responders", "frac_same_direction", "known", "af3_5seed"]].round(3).to_string(index=False))
# 7. figures
fig, ax = plt.subplots(figsize=(110 / 25.4, 90 / 25.4)); ok = PAIRS.K_all_genes.notna() & PAIRS.B_mean5.notna(); Q = PAIRS[ok]
ax.scatter(Q.B_mean5, Q.K_all_genes, s=3, c="#BDBDBD", lw=0, label="TF pairs"); kq = Q[Q.known | Q.named_control]; ax.scatter(kq.B_mean5, kq.K_all_genes, s=10, facecolors="none", edgecolors="#D55E00", lw=.6, label="known / named complex pairs")
aq = Q[Q.af3_5seed == "strong"]; ax.scatter(aq.B_mean5, aq.K_all_genes, s=10, marker="^", c="#0072B2", lw=0, label="AF3 5-seed strong")
for r in TOP.head(8).itertuples(): ax.annotate(f"{r.a}-{r.b}", (r.B_mean5, r.K_all_genes), fontsize=5, xytext=(3, 2), textcoords="offset points")
mrow = MANd[(MANd.tf_set == "all76") & (MANd.K == "all_genes") & (MANd.B == "mean5")].iloc[0]
ax.set_xlabel("binding similarity B (mean of five ChromHMM percentile matrices)"); ax.set_ylabel("knockdown-profile similarity K (cross-half Spearman)"); ax.axhline(0, color="k", lw=.3)
ax.set_title(f"76 TFs, {len(Q):,} pairs; Mantel rho = {mrow.mantel_rho:.3f} (p = {mrow.p:.3g}); positive control: {VERDICT.split(' (')[0]}", fontsize=6); ax.legend(frameon=False, fontsize=5, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); [fig.savefig(f"{OUT_}/scatter_B_vs_K.{e}", dpi=300) for e in ("png", "svg")]; plt.close(fig)
idx = [TF.index(t) for t in SETS["primary_rel_ge_0.2"]]; Kp = KM["all_genes"][np.ix_(idx, idx)].copy(); np.fill_diagonal(Kp, np.nanmax(Kp)); o = leaves_list(linkage(np.nan_to_num(1 - Kp)[np.triu_indices(len(idx), 1)], "average")); names = [SETS["primary_rel_ge_0.2"][k] for k in o]
fig, axs = plt.subplots(1, 2, figsize=(170 / 25.4, 80 / 25.4))
for ax, (M, ttl, cm) in zip(axs, ((BM["mean5"], "binding similarity B (percentile)", "Purples"), (KM["all_genes"], "knockdown similarity K (Spearman)", "RdBu_r"))):
    S_ = M[np.ix_(idx, idx)][np.ix_(o, o)].copy(); np.fill_diagonal(S_, np.nan); lim = np.nanmax(np.abs(S_))
    im = ax.imshow(S_, cmap=cm, vmin=(0 if cm == "Purples" else -lim), vmax=(1 if cm == "Purples" else lim)); ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=5); ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=5)
    ax.set_title(ttl, fontsize=6); fig.colorbar(im, ax=ax, fraction=.045, pad=.02)
fig.suptitle(f"primary set: {len(idx)} TFs with knockdown reliability >= 0.2; same order (clustered on K)", fontsize=6); fig.tight_layout(); [fig.savefig(f"{OUT_}/heatmaps_B_K_primary.{e}", dpi=300) for e in ("png", "svg")]
print("done")
