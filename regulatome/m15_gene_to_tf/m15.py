# m15_gene_to_tf (SPEC.md): rank-based gene-to-TF direction on confident knockdown pairs. Diagnostic / development, benchmark already read.
# usage: m15.py <dc|raw>  -> out/<reading>_stats.tsv, out/<reading>_pairs_K<k>.tsv, out/<reading>_composition.tsv, out/<reading>_signed.tsv, out/<reading>_reverse_lookup.tsv, out/<reading>_statistic_vs_K.png
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m15_gene_to_tf"; OUT = f"{O}/out"; os.makedirs(OUT, exist_ok=True)
LAB = "diagnostic / development, benchmark already read"; SEED = 20261111; rng = np.random.default_rng(SEED); NULL = 2000; NB = 1000; READING = sys.argv[1]
ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; panel = lambda t: ALIAS.get(t, t); OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
tfs79 = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]; MIS = dict(zip(tfs79, json.load(open(f"{RG}/m7_step2/features/raw_hicfixed/feature_qc.json"))["mis_paired_map"]["mis0"]))
FT = pd.read_csv(f"{RG}/m7_step2/splits/tf_folds.tsv", sep="\t").set_index("TF")
# ---- knockdown matrices (half1, half2, all), panel naming, genes >= 10% detection, restricted to the m9 universe
d = np.load(f"{RG}/m4_validation/kd_response/out_v2/arm2_all_cell/kd_response.npz", allow_pickle=True); q = pd.read_csv(f"{RG}/m4_validation/kd_response/out_v2/arm2_all_cell/guide_qc.tsv", sep="\t")
u = d["units"].tolist(); ui = {x: k for k, x in enumerate(u)}; keep = d["ntc_detect"] >= .1; g = pd.Series(d["genes"][keep].astype(str)); Zm = d["z"][:, keep].astype(float); gi = pd.Series(np.arange(len(g)), index=g.values); gi = gi[~gi.index.duplicated(keep="last")]
UG = np.array([x for x in np.load(f"{RG}/m9_abc_diagnosis/work/responses_arm2.npz", allow_pickle=True)["genes"].astype(str) if x in gi.index]); gix = gi[UG].values; nG = len(UG)
K_ = q[q.kept == 1]; dual = set(K_[K_.kind == "construct"].tf)
def unit_rows(kind):
    rows = {}
    for t, H in K_.groupby("tf"):
        if t in dual: names = [f"{t}|construct1"] if kind == "half1" else ([f"{t}|construct2"] if kind == "half2" else list(H.unit[H.kind == "construct"]))
        else: names = [f"{t}|{kind}"] if kind in ("half1", "half2") else [f"{t}|all"]
        names = [x for x in names if x in ui]
        if names: rows.setdefault(panel(t), []).append(np.nanmean(Zm[[ui[x] for x in names]], axis=0)[gix])
    return {p: np.mean(v, axis=0) for p, v in rows.items()}
M = {k: unit_rows(k) for k in ("half1", "half2", "all")}; TF = sorted(set(M["half1"]) & set(M["half2"]) & set(M["all"]) & set(tfs79)); nT = len(TF); ug = {x: k for k, x in enumerate(UG)}
def mat(k):
    A = np.vstack([M[k][t] for t in TF])
    for i, t in enumerate(TF):
        for o in OWN.get(t, [t]):
            if o in ug: A[i, ug[o]] = np.nan
    if READING == "dc": A = A - np.nanmean(A, 0, keepdims=True); A = A - np.nanmean(A, 1, keepdims=True)
    return A
Z1, Z2, ZA = mat("half1"), mat("half2"), mat("all")
# ---- binding percentiles (TF x gene) for the four binding readings
W11 = f"{RG}/m11_peak_regulome/out"; CT = {t: np.load(f"{RG}/m10_promoter_regulome/work/counts/{t}.npz", allow_pickle=True) for t in TF}; pg = pd.Index(CT[TF[0]]["genes"].astype(str)).get_indexer(UG)
def wide(f): W = pd.read_csv(f, sep="\t", index_col=0); return W.reindex(index=TF, columns=UG).fillna(0).values
BIND = {"a_peak_score_q005": wide(f"{W11}/wide_score_q0.05.tsv.gz"), "b_raw_promoter_reads": np.vstack([CT[t]["w500"][pg] / float(CT[t]["total"]) * 1e6 for t in TF]),
        "c_promoter_peak_score_q005": wide(f"{W11}/wide_promoter_q0.05.tsv.gz"), "d_imputed_promoter": np.vstack([CT[t]["imp500"][pg] for t in TF])}
PCT = {k: np.vstack([rankdata(r) / nG for r in v]) for k, v in BIND.items()}
WRANK = {k: np.vstack([rankdata(-v[:, j]) for j in range(nG)]).T for k, v in PCT.items()}  # 1 = highest-binding TF at that gene
# ---- matched-gene strata: promoter ATAC decile x NTC expression decile
GLf = pd.read_csv(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/GeneList.txt", sep="\t").drop_duplicates("name", keep="last").set_index("name"); acol = [c for c in GLf.columns if c.startswith("ATAC.") and c.endswith(".RPM")][0]
atac = GLf[acol].reindex(UG).fillna(0).values; expr = d["ntc_mean"][keep][gix]; st = pd.qcut(pd.Series(atac).rank(method="first"), 10, labels=False).values * 10 + pd.qcut(pd.Series(expr).rank(method="first"), 10, labels=False).values; SI = {s: np.flatnonzero(st == s) for s in np.unique(st)}
# own motif at promoter (corrected m10 promoter scan)
MOT = {}
for t in TF:
    f = f"{RG}/m10_promoter_regulome/work/g/psites_{t}.tsv"
    if os.path.exists(f) and os.path.getsize(f) > 0: MOT[t] = set(pd.read_csv(f, sep="\t", header=None)[0].astype(str))
NPK = {}
for ch in pd.read_csv(f"{W11}/long_q0.05.tsv.gz", sep="\t", usecols=["tf", "gene", "n_linked_with_peak", "promoter_peak"], chunksize=2_000_000):
    ch = ch[ch.tf.isin(TF) & ch.gene.isin(set(UG))]
    for r in ch.itertuples(index=False): NPK[(r.tf, r.gene)] = (int(r.n_linked_with_peak), bool(r.promoter_peak))
# ---- selection
def topmask(A, frac):
    m = np.zeros_like(A, bool)
    for i in range(A.shape[0]):
        a = np.abs(A[i]); ok = np.isfinite(a); k = int(round(ok.sum() * frac)); o = np.argsort(-np.where(ok, a, -1))[:k]; m[i, o] = True
    return m
T2, T1 = topmask(Z2, .05), topmask(Z1, .05)
def select(A, K):
    a = np.abs(A); flat = np.where(np.isfinite(a), a, -1).ravel(); o = np.argsort(-flat)[:K]; return np.unravel_index(o, A.shape)
STATS = []; COMP = []; SIGN = []; allpairs = {}
def statistic(ti, gj, key):
    P = PCT[key][ti, gj]; R = WRANK[key][ti, gj]; return float(P.mean()), float((R <= 5).mean()), float((R <= 10).mean())
def evaluate(sel_name, K, ti, gj):
    if not len(ti): return
    for key in BIND:
        obs = statistic(ti, gj, key)
        n1 = np.array([PCT[key][rng.integers(0, nT, len(ti)), gj].mean() for _ in range(NULL)])
        n2 = np.array([PCT[key][ti, np.array([rng.choice(SI[st[j]]) for j in gj])].mean() for _ in range(300)])
        sw = np.array([TF.index(MIS[TF[i]]) if MIS.get(TF[i]) in TF else -1 for i in ti]); okw = sw >= 0; n3 = float(PCT[key][sw[okw], gj[okw]].mean()) if okw.any() else np.nan
        ug_ = np.unique(gj); bt = []
        for _ in range(NB):
            s = rng.choice(ug_, len(ug_)); idx = np.concatenate([np.flatnonzero(gj == x) for x in s]); bt.append(PCT[key][ti[idx], gj[idx]].mean())
        r = dict(label=LAB, reading=READING, selection=sel_name, K=K, binding=key, n_pairs=len(ti), n_tfs=len(np.unique(ti)), n_genes=len(ug_), mean_pct=obs[0], lo=float(np.percentile(bt, 2.5)), hi=float(np.percentile(bt, 97.5)), frac_top5=obs[1], frac_top10=obs[2],
                 null_permTF_mean=float(n1.mean()), null_permTF_p2_5=float(np.percentile(n1, 2.5)), null_permTF_p97_5=float(np.percentile(n1, 97.5)), p_permTF=float((1 + (n1 >= obs[0]).sum()) / (NULL + 1)),
                 null_matchedgene_mean=float(n2.mean()), null_matchedgene_p97_5=float(np.percentile(n2, 97.5)), swapped_TF_mean=n3)
        for lo_name, keepm in (("without_NRF1", np.array([TF[i] != "NRF1" for i in ti])), ("without_top3_TFs", ~np.isin(ti, pd.Series(ti).value_counts().index[:3])), ("ceiling_ge_0.2", np.array([FT.ceiling.get(TF[i], np.nan) >= .2 for i in ti]))):
            if keepm.sum() >= 5:
                o2 = PCT[key][ti[keepm], gj[keepm]].mean(); n1b = np.array([PCT[key][rng.integers(0, nT, keepm.sum()), gj[keepm]].mean() for _ in range(500)]); r[f"{lo_name}_mean_pct"] = float(o2); r[f"{lo_name}_null_mean"] = float(n1b.mean()); r[f"{lo_name}_n"] = int(keepm.sum())
        STATS.append(r)
    vc = pd.Series([TF[i] for i in ti]).value_counts(); COMP.append(dict(label=LAB, reading=READING, selection=sel_name, K=K, n_pairs=len(ti), n_tfs=len(vc), n_genes=len(np.unique(gj)), pairs_per_tf=";".join(f"{a}:{b}" for a, b in vc.items())))
    zz = ZA[ti, gj]; sg = pd.DataFrame(dict(tf=[TF[i] for i in ti], down=zz < 0)).groupby("tf").down.agg(["mean", "size"]).reset_index()
    for x in sg.itertuples(): SIGN.append(dict(label=LAB, reading=READING, selection=sel_name, K=K, tf=x.tf, n_pairs=int(x[3]), frac_down_on_knockdown=float(x[2])))
for K in (100, 500, 2000, 10000):
    a1, b1 = select(Z1, K); c1 = T2[a1, b1]; a2, b2 = select(Z2, K); c2 = T1[a2, b2]
    STATS.append(dict(label=LAB, reading=READING, selection="survival", K=K, binding="", n_pairs=K, survival_half1_to_half2=float(c1.mean()), survival_half2_to_half1=float(c2.mean()), chance=0.05))
    s1 = (a1[c1], b1[c1]); s2 = (a2[c2], b2[c2]); un = set(zip(*s1)) | set(zip(*s2)); su = (np.array([x[0] for x in un], int), np.array([x[1] for x in un], int))
    for nm, (ti, gj) in (("half1_confirmed_half2", s1), ("half2_confirmed_half1", s2), ("union", su)): evaluate(nm, K, ti, gj)
    allpairs[K] = su
# per-gene version: same top-|z| TF in half1 and half2
b1g = np.nanargmax(np.abs(np.where(np.isfinite(Z1), Z1, 0)), axis=0); b2g = np.nanargmax(np.abs(np.where(np.isfinite(Z2), Z2, 0)), axis=0); same = np.flatnonzero(b1g == b2g)
STATS.append(dict(label=LAB, reading=READING, selection="per_gene_same_top_TF", K=0, binding="", n_pairs=len(same), survival_half1_to_half2=float(len(same) / nG), chance=float(1 / nT)))
evaluate("per_gene_same_top_TF", 0, b1g[same], same)
pd.DataFrame(STATS).to_csv(f"{OUT}/{READING}_stats.tsv", sep="\t", index=False, float_format="%.4f"); pd.DataFrame(COMP).to_csv(f"{OUT}/{READING}_composition.tsv", sep="\t", index=False); pd.DataFrame(SIGN).to_csv(f"{OUT}/{READING}_signed.tsv", sep="\t", index=False, float_format="%.3f")
for K, (ti, gj) in allpairs.items():
    rows = []
    for i, j in zip(ti, gj):
        t, gname = TF[i], UG[j]; npk = NPK.get((t, gname), (0, False))
        rows.append(dict(gene=gname, tf=t, z_half1=Z1[i, j], z_half2=Z2[i, j], z_all=ZA[i, j], **{f"pct_{k}": PCT[k][i, j] for k in BIND}, **{f"rank_among_TFs_{k}": int(WRANK[k][i, j]) for k in BIND}, n_linked_elements_with_peak_q005=npk[0], promoter_peak_q005=npk[1], own_motif_at_promoter=(gname in MOT[t]) if t in MOT else None))
    pd.DataFrame(rows).sort_values("z_all", key=np.abs, ascending=False).to_csv(f"{OUT}/{READING}_pairs_K{K}.tsv", sep="\t", index=False, float_format="%.4g")
top_genes = np.argsort(-np.nanmax(np.abs(np.where(np.isfinite(ZA), ZA, 0)), axis=0))[:15]; RL = []
for j in top_genes:
    o = np.argsort(-np.abs(np.where(np.isfinite(ZA[:, j]), ZA[:, j], 0)))[:10]
    for rank, i in enumerate(o, 1): RL.append(dict(gene=UG[j], kd_rank=rank, tf=TF[i], z_all=ZA[i, j], z_half1=Z1[i, j], z_half2=Z2[i, j], **{f"pct_{k}": PCT[k][i, j] for k in BIND}))
pd.DataFrame(RL).to_csv(f"{OUT}/{READING}_reverse_lookup.tsv", sep="\t", index=False, float_format="%.4g")
S = pd.DataFrame(STATS); S = S[(S.selection == "union") & (S.binding != "")]; fig, axs = plt.subplots(1, 4, figsize=(15, 3.4))
for ax, key in zip(axs, BIND):
    Q = S[S.binding == key].sort_values("K"); x = np.arange(len(Q))
    ax.errorbar(x, Q.mean_pct, yerr=[Q.mean_pct - Q.lo, Q.hi - Q.mean_pct], marker="o", label="confident pairs (union)"); ax.fill_between(x, Q.null_permTF_p2_5, Q.null_permTF_p97_5, alpha=.25, label="TF labels permuted (95%)")
    ax.plot(x, Q.null_matchedgene_mean, "s--", label="matched random genes"); ax.plot(x, Q.swapped_TF_mean, "^:", label="swapped TF"); ax.set_xticks(x); ax.set_xticklabels(Q.K.astype(int)); ax.set_xlabel("K"); ax.set_title(key, fontsize=8)
axs[0].set_ylabel("mean binding percentile of the pair's TF"); axs[0].legend(fontsize=6); fig.suptitle(f"Gene-to-TF, confident knockdown pairs ({READING} z); {LAB}", fontsize=8); fig.tight_layout(); fig.savefig(f"{OUT}/{READING}_statistic_vs_K.png", dpi=150)
print("TFs", nT, "genes", nG); print(pd.DataFrame(STATS)[["selection", "K", "binding", "n_pairs", "mean_pct", "lo", "hi", "null_permTF_mean", "p_permTF", "null_matchedgene_mean", "swapped_TF_mean"]].to_string(index=False))
