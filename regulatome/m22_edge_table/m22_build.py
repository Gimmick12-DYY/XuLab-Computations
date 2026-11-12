# m22_edge_table (SPEC.md): the HEK293T Regulome edge table from our HITAG data (edges = linked q005 peaks); Perturb-seq appears only as external columns.
# usage: m22_build.py [--background atac|igg]   (igg path is prepared but only used after the user confirms the IgG sample is HEK293T)
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import nbinom, rankdata
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m22_edge_table"; OUT = f"{O}/out"; os.makedirs(OUT, exist_ok=True); V = "/vast/som/xujie_lab/TAGATG-293T/bigwig"; QMIN = -np.log10(0.05)
BG = sys.argv[sys.argv.index("--background") + 1] if "--background" in sys.argv else "atac"; assert BG in ("atac", "igg")
tfs79 = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]; ti = {t: i for i, t in enumerate(tfs79)}
UNIV = np.load(f"{RG}/m9_abc_diagnosis/work/responses_arm2.npz", allow_pickle=True)["genes"].astype(str); gi = {g: k for k, g in enumerate(UNIV)}; NG = len(UNIV)
EL = pd.read_csv(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/EnhancerList.txt", sep="\t"); acol = [c for c in EL.columns if c.startswith("ATAC.") and c.endswith(".RPM")][0]
EL = EL[["chr", "start", "end", acol]].rename(columns={acol: "atac"}).reset_index(drop=True); NE = len(EL); EL["mid"] = (EL.start + EL.end) // 2; ek = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(EL.chr, EL.start, EL.end))}
L = pd.read_csv(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Predictions/EnhancerPredictionsFull_threshold0.02_self_promoter.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene", "TargetGeneTSS", "ABC.Score", "isSelfPromoter"])
L = L[L.TargetGene.isin(gi)].copy(); L["e"] = (L.chr + ":" + L.start.astype(str) + "-" + L.end.astype(str)).map(ek).astype(int); L["g"] = L.TargetGene.map(gi); L["self"] = L.isSelfPromoter.astype(str).str.lower().eq("true"); L["abc"] = L["ABC.Score"].astype(float); L["dist"] = np.abs(EL.mid.values[L.e.values] - L.TargetGeneTSS.values)
Le, Lg, La, Ls, Ld = L.e.values, L.g.values, L.abc.values, L.self.values, L.dist.values
# element-level peaks (strongest q005 summit per element per TF) and reads
CHI = {c: (g.start.values, g.end.values, g.index.values) for c, g in EL.groupby("chr")}
def elem_of(c, p):
    r = np.full(len(p), -1)
    for ch in np.unique(c):
        if ch not in CHI: continue
        st, en, ix = CHI[ch]; o = np.argsort(st); st, en, ix = st[o], en[o], ix[o]; m = c == ch; k = np.searchsorted(st, p[m], side="right") - 1; ok = (k >= 0) & (p[m] < en[np.maximum(k, 0)]); rr = np.full(m.sum(), -1); rr[ok] = ix[k[ok]]; r[m] = rr
    return r
PE = np.zeros((NE, 79), np.float32)
for t in tfs79:
    f = f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"
    if not os.path.exists(f) or os.path.getsize(f) == 0: continue
    P = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 6, 8, 9], names=["c", "s", "sig", "q", "off"]); P = P[P.q >= QMIN]; e = elem_of(P.c.values.astype(str), (P.s + P.off).values); m = e >= 0; np.maximum.at(PE[:, ti[t]], e[m], P.sig.values[m])
C = np.load(f"{RG}/m7_step2/features/canon_counts/raw_counts.npy").astype(float); TOT = C.sum(0); RE = C / np.maximum(TOT, 1) * 1e6
WP = np.zeros((79, NG)); WPc = np.zeros((79, NG))
for t in tfs79:
    c_ = np.load(f"{RG}/m10_promoter_regulome/work/counts/{t}.npz", allow_pickle=True); ix = pd.Index(c_["genes"].astype(str)).get_indexer(UNIV); WPc[ti[t]] = np.where(ix >= 0, c_["w500"][ix], 0); WP[ti[t]] = WPc[ti[t]] / float(c_["total"]) * 1e6
print("inputs loaded", flush=True)
# background: per TF, element reads ~ NB with log mu = a + b log(ATAC RPM + 0.1) + c log(width); fitted by Poisson IRLS, NB2 dispersion by moments
def bg_design():
    if BG == "atac": return np.log(EL.atac.values + .1)
    raise SystemExit("IgG background not enabled: the user has not confirmed that N2.IgG is HEK293T")
xb = bg_design(); xw = np.log(EL.end.values - EL.start.values); Xd = np.column_stack([np.ones(NE), xb, xw]); ENR = np.zeros((NE, 79), np.float32); PV = np.ones((NE, 79), np.float32); FIT = []
for t in tfs79:
    y = C[:, ti[t]]; beta = np.zeros(3); beta[0] = np.log(max(y.mean(), 1e-6))
    for _ in range(50):
        eta = Xd @ beta; mu = np.exp(np.clip(eta, -30, 30)); z = eta + (y - mu) / np.maximum(mu, 1e-9); Wt = mu; A = Xd.T @ (Xd * Wt[:, None]); bnew = np.linalg.solve(A + 1e-8 * np.eye(3), Xd.T @ (Wt * z))
        if np.max(np.abs(bnew - beta)) < 1e-7: beta = bnew; break
        beta = bnew
    mu = np.exp(Xd @ beta); alpha = max(np.sum(((y - mu) ** 2 - y) / np.maximum(mu, 1e-9) ** 2) / max(len(y) - 3, 1), 1e-6); n_ = 1 / alpha; p_ = n_ / (n_ + mu)
    ENR[:, ti[t]] = (y + .5) / (mu + .5); PV[:, ti[t]] = nbinom.sf(y - 1, n_, p_); FIT.append(dict(tf=t, b0=beta[0], b_atac=beta[1], b_width=beta[2], nb_alpha=alpha))
pd.DataFrame(FIT).to_csv(f"{OUT}/background_fits.tsv", sep="\t", index=False, float_format="%.5g"); print("background fitted", flush=True)
# modules (Figure 3, as built) per TF -> element ids
def reg_elems(regs):
    out = []
    for r in regs:
        if not isinstance(r, str) or ":" not in r: continue
        c, se = r.split(":"); s, e = map(int, se.split("-"))
        if c not in CHI: continue
        st, en, ix = CHI[c]; o = np.argsort(st); st, en, ix = st[o], en[o], ix[o]; k = max(np.searchsorted(st, s, side="right") - 1, 0)
        while k < len(st) and st[k] < e:
            if en[k] > s: out.append(ix[k])
            k += 1
    return out
MODE = {}
for t in tfs79:
    f = f"{X}/cobinding/results/ourrun_k50/{t}/modules.tsv"
    if not os.path.exists(f): continue
    for mid, regs in zip(pd.read_csv(f, sep="\t").module_id, pd.read_csv(f, sep="\t").regions.astype(str)):
        for e in reg_elems(regs.split(";")): MODE.setdefault((t, e), []).append(mid)
CAND = pd.read_csv(f"{X}/complex_screen/candidates_frozen_v1/candidates_frozen_v1.tsv", sep="\t"); PART = {}
for a, b in zip(CAND.tf_a, CAND.tf_b):
    if a in ti and b in ti: PART.setdefault(a, set()).add(b); PART.setdefault(b, set()).add(a)
# own motif sites at the promoter (m10 corrected scans) and in elements
MOT = sorted(f[7:-4] for f in os.listdir(f"{RG}/m10_promoter_regulome/work/g") if f.startswith("psites_") and f.endswith(".tsv"))
PSITE = {t: set(pd.read_csv(f"{RG}/m10_promoter_regulome/work/g/psites_{t}.tsv", sep="\t", header=None)[0].astype(str)) for t in MOT}
ESITE = {t: set(pd.read_csv(f"{RG}/m10_promoter_regulome/work/g/sites_{t}.tsv", sep="\t", header=None)[0].astype(int)) for t in MOT}
# external Perturb-seq columns
Rz = np.load(f"{RG}/m9_abc_diagnosis/work/responses_arm2.npz", allow_pickle=True); KT = list(Rz["tfs"].astype(str)); Z = Rz["r1"].astype(float)
sys.argv = ["m15.py", "raw"]; src = open(f"{RG}/m15_gene_to_tf/m15.py").read(); _keep = (O, OUT); exec(src[:src.index("# ---- binding percentiles")]); O, OUT = _keep
H1 = pd.DataFrame(Z1, index=TF, columns=UG).reindex(columns=UNIV); H2 = pd.DataFrame(Z2, index=TF, columns=UG).reindex(columns=UNIV)
FT = pd.read_csv(f"{RG}/m7_step2/splits/tf_folds.tsv", sep="\t").set_index("TF").ceiling; SPH = pd.read_csv(f"{RG}/m16_strong_phenotype/out/strong_phenotype_per_tf.tsv", sep="\t").set_index("panel_tf").status
TIER = pd.read_csv(f"{RG}/m21_tf_tiers/out/tf_evidence_table.tsv", sep="\t").set_index("tf").tier
# per (TF, gene) aggregation over linked elements
score = np.zeros((79, NG)); nd = np.zeros((79, NG), int); prom_sig = np.zeros((79, NG)); best_abc = np.zeros((79, NG)); best_dist = np.full((79, NG), np.nan); best_el = np.full((79, NG), -1); enr_g = np.zeros((79, NG)); pmin = np.ones((79, NG)); cpm_g = np.zeros((79, NG))
for k in range(79):
    p = PE[Le, k]; bound = p > 0; sc = p * La; score[k] = np.bincount(Lg, weights=sc, minlength=NG); nd[k] = np.bincount(Lg, weights=(bound & ~Ls).astype(float), minlength=NG).astype(int)
    np.maximum.at(prom_sig[k], Lg[Ls], p[Ls]); cpm_g[k] = WP[k] + np.bincount(Lg, weights=RE[Le, k] * (~Ls), minlength=NG)
    o = np.lexsort((-sc, Lg)); first = np.r_[True, Lg[o][1:] != Lg[o][:-1]]; oo = o[first & bound[o]]; best_abc[k, Lg[oo]] = La[oo]; best_dist[k, Lg[oo]] = Ld[oo]; best_el[k, Lg[oo]] = Le[oo]
    e_ = np.where(bound, ENR[Le, k], 0); np.maximum.at(enr_g[k], Lg, e_); pv_ = np.where(bound, PV[Le, k], 1); np.minimum.at(pmin[k], Lg, pv_)
bound_g = score > 0; shared_by = bound_g.sum(0); pct = np.vstack([rankdata(score[k]) / NG for k in range(79)]); share = score / np.maximum(score.sum(0, keepdims=True), 1e-12)
ROWS = []
for k, t in enumerate(tfs79):
    for g in np.flatnonzero(bound_g[k]):
        gene = UNIV[g]; e = best_el[k, g]; others = [tfs79[j] for j in np.flatnonzero(bound_g[:, g]) if j != k]
        loo = (cpm_g[:, g].sum() - cpm_g[k, g]) / 78
        r = dict(tf=t, gene=gene, promoter_peak=bool(prom_sig[k, g] > 0), promoter_signalValue=float(prom_sig[k, g]), n_distal_elements_with_peak=int(nd[k, g]), best_element=f"{EL.chr[e]}:{EL.start[e]}-{EL.end[e]}" if e >= 0 else "",
                 best_abc=float(best_abc[k, g]), best_dist_to_tss=float(best_dist[k, g]), score_signal_x_abc=float(score[k, g]), pct_within_tf=float(pct[k, g]), promoter_reads_cpm=float(WP[k, g]),
                 shared_by=int(shared_by[g]), co_bound_tfs=",".join(others), panel_relative_ratio=float((cpm_g[k, g] + .01) / (loo + .01)),
                 frozen_partners_cobound_same_element=",".join(sorted(b for b in PART.get(t, ()) if e >= 0 and PE[e, ti[b]] > 0)), coordinated_binding_module=";".join(MODE.get((t, e), [])) if e >= 0 else "",
                 own_motif_promoter=(gene in PSITE[t]) if t in PSITE else None, own_motif_best_element=(int(e) in ESITE[t]) if (t in ESITE and e >= 0) else None,
                 enrichment_over_background=float(enr_g[k, g]), background_p_min=float(pmin[k, g]), background=BG,
                 predicted_impact=float(score[k, g]), predicted_impact_share_at_gene=float(share[k, g]), tier=TIER.get(t, ""))
        if t in KT:
            i = KT.index(t); r.update({"observed_kd_z [external: X-Atlas/Orion HEK293T]": float(Z[i, g]), "observed_kd_z_half1 [external]": float(H1.loc[t, gene]) if t in H1.index else np.nan, "observed_kd_z_half2 [external]": float(H2.loc[t, gene]) if t in H2.index else np.nan,
                                   "kd_reliability [external]": float(FT.get(t, np.nan)), "kd_strong_phenotype [external]": SPH.get(t, "") == "strong phenotype"})
        ROWS.append(r)
E = pd.DataFrame(ROWS); h1c, h2c = "observed_kd_z_half1 [external]", "observed_kd_z_half2 [external]"
E["observed_kd_reproducible [external]"] = (E[h1c].abs() >= 2) & (E[h2c].abs() >= 2) & (np.sign(E[h1c]) == np.sign(E[h2c]))
E.to_csv(f"{OUT}/regulome_edges.tsv.gz", sep="\t", index=False, float_format="%.5g"); print("edges", len(E), flush=True)
np.savez_compressed(f"{O}/work_matrices.npz", score=score, share=share, enr=enr_g, tfs=np.array(tfs79), genes=UNIV)
