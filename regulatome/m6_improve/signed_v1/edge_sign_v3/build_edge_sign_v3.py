# edge_sign_v3 (rule in README.md, declared before any output): 95th-percentile enrichment AND CPM >= 75th percentile. Enrichment-based co-binding flag; everything else as edge_sign_v1/build_edge_sign.py. Builds the per-edge sign layer on v8 and then the
# DESCRIPTIVE check against the knockdown z. Writes edge_sign_v1.tsv, tf_departure_summary.tsv, check_per_tf_DESCRIPTIVE.tsv,
# check_pooled_DESCRIPTIVE.tsv. usage: build_edge_sign.py
import os, subprocess
import numpy as np, pandas as pd, scipy.sparse as sp
X = "/work/users/d/y/dyy12/XuLab"; R = f"{X}/regulatome"; M4 = f"{R}/m4_validation"; S1 = f"{R}/m6_improve/signed_v1"; O = f"{S1}/edge_sign_v3"; G1 = f"{R}/gate1/cellmats"
assert not os.path.exists(f"{O}/edge_sign_v1.tsv")
v = subprocess.run("sha256sum -c --quiet manifest_v8.sha256", shell=True, cwd=f"{R}/r6_v8", capture_output=True, text=True); assert v.returncode == 0, v.stdout + v.stderr
SEED = 20261109; ARMS = {"arm1": "arm1_high_knockdown", "arm2": "arm2_all_cell"}; ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; panel = lambda t: ALIAS.get(t, t)
E_ = pd.read_csv(f"{R}/m0_universe/elements_v2.bed", sep="\t", header=None, names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"]); Ea = E_[E_.cls.isin(["promoter", "distal_active", "bivalent"])].reset_index(drop=True)
A = pd.read_csv(f"{R}/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene"])
coord = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(Ea.chrom, Ea.start, Ea.end))}; A["e"] = (A.chr + ":" + A.start.astype(str) + "-" + A.end.astype(str)).map(coord).astype(int)
# ChromHMM state per element (largest bp overlap)
CH = pd.read_csv(f"{X}/tf_complex/work/chromhmm_elements_counts/elements.tsv", sep="\t"); st = np.array(["none"] * len(Ea), dtype=object)
for ch, G in Ea.groupby("chrom"):
    H = CH[CH.chrom == ch].sort_values("start"); hs, he, hst = H.start.values, H.end.values, H.state.values
    for i, s0, e0 in zip(G.index, G.start.values, G.end.values):
        a, b = np.searchsorted(he, s0, side="right"), np.searchsorted(hs, e0)
        if b > a: ov = np.minimum(he[a:b], e0) - np.maximum(hs[a:b], s0); st[i] = hst[a:b][np.argmax(ov)]
ACT = {"TssA", "TssFlnk", "TssFlnkU", "TssFlnkD", "EnhA1", "EnhA2", "EnhG1", "EnhG2", "EnhWk", "Tx", "TxWk"}; PCG = {"ReprPC", "ReprPCWk", "TssBiv", "EnhBiv"}; HET = {"Het", "ZNF/Rpts"}
cstate = np.array(["active" if s in ACT else "pcg" if s in PCG else "het" if s in HET else "quiescent" for s in st], dtype=object)
tfs = [l.strip() for l in open(f"{R}/m1_binding/v2/tfs.txt") if l.strip()]
CPM = {}
for t in tfs:
    y = np.asarray(sp.load_npz(f"{G1}/{t}.raw.npz").sum(0)).ravel(); CPM[panel(t)] = y / np.load(f"{G1}/{t}.cells.npz")["reads"].astype(float).sum() * 1e6
REPM = ["EZH2", "MTF2", "MBD3", "RBBP4", "RBBP7", "HP1"]; ACTM = ["NRF1", "MAZ", "NFYA", "E2F3"]
TOT = np.sum([CPM[t] for t in CPM], axis=0); NT = len(CPM); assert NT == 79
E90 = {m: np.quantile(np.log2((CPM[m] + .5) / (TOT / NT + .5)), .95) for m in REPM + ACTM}; MED = {m: np.quantile(CPM[m], .75) for m in REPM + ACTM}
def flag(m, t, e):
    pool = (TOT[e] - CPM[t][e]) / (NT - 1); Em = np.log2((CPM[m][e] + .5) / (pool + .5)); return (Em >= E90[m]) & (CPM[m][e] >= MED[m])
CAN = pd.read_csv(f"{S1}/tf_sign_annotation.tsv", sep="\t"); CAN["panel"] = CAN.tf.map(panel)
base = {}
for r in CAN.itertuples():
    base[r.panel] = (1 if r.predicted_sign == "activator" else -1 if r.predicted_sign == "repressor" else 0) if r.confidence in ("high", "medium") else 0
base["SREBF2"] = 1; BSRC = {t: ("rc79" if t == "SREBF2" else "canonical") for t in base}
L8 = {f[:-4]: pd.read_csv(f"{R}/r6_v8/lists/{f}", sep="\t").gene.tolist() for f in os.listdir(f"{R}/r6_v8/lists")}
rows, dep = [], []
for t in sorted(L8):
    Asub = A[A.TargetGene.isin(set(L8[t]))]; e = Asub.e.values
    rep_cb = np.zeros(len(e), bool); act_cb = np.zeros(len(e), bool)
    for m in REPM:
        if m != t: rep_cb |= flag(m, t, e)
    for m in ACTM:
        if m != t: act_cb |= flag(m, t, e)
    s = cstate[e]; rep_ctx = np.isin(s, ["pcg", "het"]) | rep_cb; b0 = base.get(t, 0)
    es = np.where(rep_ctx & ~act_cb, -1, np.where(((s == "active") | act_cb) & ~rep_ctx, 1, b0))
    w = CPM[t][e]; gs = pd.Series(es * w).groupby(Asub.TargetGene.values).sum(); gsign = np.sign(gs)
    D_ = pd.DataFrame(dict(TF=t, gene=Asub.TargetGene.values, element=[f"{c}:{a}-{b}" for c, a, b in zip(Asub.chr, Asub.start, Asub.end)], state=s, chromhmm=st[e], rep_cobound=rep_cb, act_cobound=act_cb, tf_cpm=w, edge_sign=es))
    D_["gene_sign"] = D_.gene.map(gsign).astype(int); D_["baseline"] = b0; D_["baseline_source"] = BSRC.get(t, "canonical"); rows.append(D_)
    g = gsign.reindex(L8[t]).dropna()
    dep.append(dict(TF=t, baseline=b0, n_edges=len(es), frac_edges_activating=(es == 1).mean(), frac_edges_repressing=(es == -1).mean(), frac_edges_unsigned=(es == 0).mean(), frac_edges_depart=((es != b0)).mean(),
                    n_genes=len(g), frac_genes_activating=(g == 1).mean(), frac_genes_repressing=(g == -1).mean(), frac_genes_unsigned=(g == 0).mean(), frac_genes_depart=(g != b0).mean()))
EG = pd.concat(rows); EG.to_csv(f"{O}/edge_sign_v1.tsv", sep="\t", index=False, float_format="%.4g"); DP = pd.DataFrame(dep); DP.to_csv(f"{O}/tf_departure_summary.tsv", sep="\t", index=False, float_format="%.4f")
print("edges", len(EG), "state mix", pd.Series(cstate).value_counts().to_dict(), flush=True)
fr = dict(n_edges=len(EG), rep_cobound=EG.rep_cobound.mean(), act_cobound=EG.act_cobound.mean(), both=(EG.rep_cobound & EG.act_cobound).mean(), edge_activating=(EG.edge_sign == 1).mean(), edge_repressing=(EG.edge_sign == -1).mean(), edge_unsigned=(EG.edge_sign == 0).mean())
pd.Series(fr).to_csv(f"{O}/flag_rates.tsv", sep="\t", header=["value"]); print(fr, flush=True)
if not all(.02 <= fr[k] <= .5 for k in ("rep_cobound", "act_cobound")):
    open(f"{O}/GATE_FAILED.txt", "w").write(f"sanity gate failed (flag rate outside [0.02, 0.50]): {fr}\n"); raise SystemExit("GATE FAILED; knockdown z not read")
# ---- DESCRIPTIVE check vs knockdown z
GS = EG.drop_duplicates(["TF", "gene"]).set_index(["TF", "gene"]).gene_sign; Sp = pd.read_csv(f"{R}/m6_improve/split_v1.tsv", sep="\t").set_index("tf"); out = []
for a in ARMS:
    d = np.load(f"{M4}/kd_response/out_v2/{ARMS[a]}/kd_response.npz", allow_pickle=True); q = pd.read_csv(f"{M4}/kd_response/out_v2/{ARMS[a]}/guide_qc.tsv", sep="\t"); u = d["units"].tolist(); ui = {x: k for k, x in enumerate(u)}
    uni = d["ntc_detect"] >= .1; g = d["genes"][uni]; Z = d["z"][:, uni].astype(float); kp = q[q.kept == 1]; prim = kp[kp.kind.isin(["all", "construct"])]; tf_ = sorted(prim.tf.unique())
    P_ = np.nan_to_num(np.vstack([np.nanmean(Z[[ui[x] for x in prim.unit[prim.tf == t]]], axis=0) for t in tf_])); pts = sorted({panel(t) for t in tf_})
    Zp = np.vstack([P_[[i for i, t in enumerate(tf_) if panel(t) == p]].mean(0) for p in pts]); Zdc = Zp - Zp.mean(0); Zdc = Zdc - Zdc.mean(1, keepdims=True); gi = {x: k for k, x in enumerate(g)}
    rng = np.random.default_rng(SEED + (a == "arm2"))
    for i, p in enumerate(pts):
        if p not in L8: continue
        s = GS.loc[p]; s = s[s.index.isin(gi.keys())]; sg = s[s != 0]
        if len(sg) < 20: continue
        z = Zdc[i][[gi[x] for x in sg.index]]; sv = sg.values.astype(float); conc = np.mean(sv * -z); na, nr = int((sv > 0).sum()), int((sv < 0).sum())
        amr = (z[sv > 0].mean() - z[sv < 0].mean()) if na and nr else np.nan
        rnd = np.array([np.mean(rng.permutation(sv) * -z) for _ in range(1000)])
        b0 = base.get(p, 0); zall = Zdc[i][[gi[x] for x in s.index]]; const = float(np.mean(b0 * -zall)) if b0 else np.nan
        out.append(dict(DESCRIPTIVE="DESCRIPTIVE", arm=a, tf=p, split=Sp.set.get(p, "na"), baseline={1: "activator", -1: "repressor", 0: "unsigned"}[b0], n_signed_genes=len(sg), n_act=na, n_rep=nr,
                        concordance=conc, act_minus_rep=amr, conc_act_genes=float(np.mean(-z[sv > 0])) if na else np.nan, conc_rep_genes=float(np.mean(z[sv < 0])) if nr else np.nan, random_mean=rnd.mean(), random_sd=rnd.std(), pct_vs_random=float((rnd < conc).mean()), const_baseline_concordance=const))
C_ = pd.DataFrame(out); C_.to_csv(f"{O}/check_per_tf_DESCRIPTIVE.tsv", sep="\t", index=False, float_format="%.4g")
PO = []; rng = np.random.default_rng(SEED); MARK = set(REPM + ACTM)
for excl, mk in ((False, False), (True, False), (False, True), (True, True)):
    CC = pd.concat([C_.assign(cls="pooled"), C_.assign(cls=C_.baseline)]); keep = ~((excl & (CC.tf == "NRF1")) | (mk & CC.tf.isin(MARK)))
    for (a, cls), G in CC[keep.values].groupby(["arm", "cls"]):
        dd = (G.concordance - G.random_mean).values; bs = dd[rng.integers(0, len(dd), (10000, len(dd)))].mean(1)
        dc_ = (G.concordance - G.const_baseline_concordance).dropna().values; bc = dc_[rng.integers(0, len(dc_), (10000, len(dc_)))].mean(1) if len(dc_) else np.array([np.nan])
        PO.append(dict(DESCRIPTIVE="DESCRIPTIVE", nrf1="excluded" if excl else "included", marker_tfs="excluded" if mk else "included", arm=a, baseline_class=cls, n_tfs=len(G), mean_concordance=G.concordance.mean(), mean_random=G.random_mean.mean(),
                       model_minus_random=dd.mean(), mmr_ci=f"[{np.percentile(bs, 2.5):.4f}, {np.percentile(bs, 97.5):.4f}]", n_tfs_above_random_95=int((G.pct_vs_random >= .95).sum()),
                       n_with_baseline=len(dc_), model_minus_constant=dc_.mean() if len(dc_) else np.nan, mmc_ci=f"[{np.nanpercentile(bc, 2.5):.4f}, {np.nanpercentile(bc, 97.5):.4f}]", median_act_minus_rep=G.act_minus_rep.median(), mean_conc_act_genes=G.conc_act_genes.mean(), mean_conc_rep_genes=G.conc_rep_genes.mean()))
pd.DataFrame(PO).to_csv(f"{O}/check_pooled_DESCRIPTIVE.tsv", sep="\t", index=False, float_format="%.4g")
