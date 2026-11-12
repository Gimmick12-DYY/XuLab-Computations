# m22 Amendment 1: simple correlation between the ABC score (signal x ABC) and Perturb-seq, whole matrix then per TF. Tables only. Diagnostic.
import json, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m22_edge_table"; OUT = f"{O}/out"; rng = np.random.default_rng(20261112); LAB = "diagnostic"
W = np.load(f"{O}/work_matrices.npz", allow_pickle=True); tfs79 = list(W["tfs"]); UNIV = W["genes"].astype(str); NG = len(UNIV); S79 = W["score"]
Rz = np.load(f"{RG}/m9_abc_diagnosis/work/responses_arm2.npz", allow_pickle=True); assert (Rz["genes"].astype(str) == UNIV).all(); KT = list(Rz["tfs"].astype(str))
OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}; gi = {g: k for k, g in enumerate(UNIV)}
def own_mask(T):
    M = np.zeros((len(T), NG), bool)
    for i, t in enumerate(T):
        for o in OWN.get(t, [t]):
            if o in gi: M[i, gi[o]] = True
    return M
AT = pd.read_csv(f"{RG}/m11_peak_regulome/out/atac_x_abc_per_gene.tsv.gz", sep="\t").set_index("gene").atac_x_abc.reindex(UNIV).fillna(0).values
FL = pd.read_csv(f"{RG}/m6_improve/repro_v1/per_tf_floor_test.tsv", sep="\t").set_index("tf"); FT = pd.read_csv(f"{RG}/m7_step2/splits/tf_folds.tsv", sep="\t").set_index("TF").ceiling
SP = pd.read_csv(f"{RG}/m16_strong_phenotype/out/strong_phenotype_per_tf.tsv", sep="\t").set_index("panel_tf").status
MIS = dict(zip([l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()], json.load(open(f"{RG}/m7_step2/features/raw_hicfixed/feature_qc.json"))["mis_paired_map"]["mis0"]))
def versions():
    Z = Rz["r1"].astype(float); D = Rz["r2"].astype(float); yield "all_cells", KT, Z, D
    T = pd.read_csv(f"{RG}/m23_sgrna_top25/out/observed_kd_top25.tsv.gz", sep="\t"); P = T.pivot(index="tf", columns="gene", values="z_top25").reindex(columns=UNIV); tl = list(P.index)
    Zt = P.values.astype(float); Dt = Zt - np.nanmean(Zt, 0, keepdims=True); Dt = Dt - np.nanmean(Dt, 1, keepdims=True); yield "top25_sgRNA", tl, Zt, Dt
def pear(a, b): m = np.isfinite(a) & np.isfinite(b); return np.nan if m.sum() < 10 or np.std(a[m]) == 0 or np.std(b[m]) == 0 else float(np.corrcoef(a[m], b[m])[0, 1])
def spear(a, b): m = np.isfinite(a) & np.isfinite(b); return np.nan if m.sum() < 10 or np.std(a[m]) == 0 or np.std(b[m]) == 0 else float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1])
def rowstd(M): mu = np.nanmean(M, 1, keepdims=True); sd = np.nanstd(M, 1, keepdims=True); sd[sd == 0] = np.nan; return (M - mu) / sd
def rowpct(M):
    out = np.full(M.shape, np.nan)
    for i in range(len(M)):
        m = np.isfinite(M[i])
        if m.sum(): out[i, m] = rankdata(M[i, m]) / m.sum()
    return out
WHOLE = []; PER = []
for ver, T, Z, D in versions():
    keep = [i for i, t in enumerate(T) if t in tfs79]; T = [T[i] for i in keep]; Z = Z[keep]; D = D[keep]
    om = own_mask(T); Z[om] = np.nan; D[om] = np.nan; S = np.vstack([S79[tfs79.index(t)] for t in T]); S = np.where(np.isfinite(Z), S, np.nan); A = np.where(np.isfinite(Z), np.tile(AT, (len(T), 1)), np.nan)
    tert = np.array([str(FL.tertile.get(t, "na")) for t in T]); groups = {k: np.flatnonzero(tert == k) for k in np.unique(tert)}
    for target, Y in (("|z|", np.abs(Z)), ("signed z", Z)):
        for way, f in (("1 raw pooled", lambda M: M), ("2 row-standardised", rowstd), ("3 within-TF percentile", rowpct)):
            Sx, Yx, Ax = f(S), f(Y), f(A)
            obs_s, obs_p = spear(Sx.ravel(), Yx.ravel()), pear(Sx.ravel(), Yx.ravel()); at_s = spear(Ax.ravel(), Yx.ravel())
            null = []
            for _ in range(1000):
                pr = np.arange(len(T))
                for k, ix in groups.items(): pr[ix] = rng.permutation(ix)
                null.append(spear(f(S[pr]).ravel(), Yx.ravel()))
            bt = []
            for _ in range(2000):
                b = rng.integers(0, len(T), len(T)); bt.append(spear(Sx[b].ravel(), Yx[b].ravel()))
            WHOLE.append(dict(label=LAB, version=ver, target=target, way=way, n_tfs=len(T), n_entries=int(np.isfinite(Sx.ravel()) .__and__(np.isfinite(Yx.ravel())).sum()), spearman=obs_s, ci_lo=np.nanpercentile(bt, 2.5), ci_hi=np.nanpercentile(bt, 97.5), pearson=obs_p,
                              swapped_tf_spearman_mean=float(np.nanmean(null)), swapped_p=(1 + int(np.sum(np.array(null) >= obs_s))) / 1001, atac_only_spearman=at_s))
            print(ver, target, way, round(obs_s, 4), flush=True)
    for i, t in enumerate(T):
        s, y, z, d = S[i], np.abs(Z[i]), Z[i], D[i]; ok = np.isfinite(y); bnd = ok & (s > 0)
        bt = [spear(s[ok][j], y[ok][j]) for j in (rng.integers(0, ok.sum(), ok.sum()) for _ in range(500))]
        sw = MIS.get(t); ssw = S79[tfs79.index(sw)] if sw in tfs79 else None
        PER.append(dict(label=LAB, version=ver, tf=t, n_genes=int(ok.sum()), n_bound=int(bnd.sum()), spearman_abs_z=spear(s, y), sp_lo=np.nanpercentile(bt, 2.5), sp_hi=np.nanpercentile(bt, 97.5), pearson_abs_z=pear(s, y),
                        spearman_signed_z=spear(s, z), pearson_signed_z=pear(s, z), spearman_dc=spear(s, d), pearson_dc=pear(s, d), spearman_abs_z_bound_only=spear(np.where(bnd, s, np.nan), np.where(bnd, y, np.nan)),
                        pointbiserial_edge_abs_z=pear((s > 0).astype(float), y), atac_only_spearman_abs_z=spear(AT, y), swapped_tf=sw, swapped_spearman_abs_z=spear(ssw, y) if ssw is not None else np.nan,
                        reliability=FT.get(t, np.nan), strong_phenotype=SP.get(t, "") == "strong phenotype"))
Wd = pd.DataFrame(WHOLE); Wd.to_csv(f"{OUT}/simple_correlation_whole_matrix.tsv", sep="\t", index=False, float_format="%.4g"); print(Wd.drop(columns="label").round(4).to_string(index=False), flush=True)
P = pd.DataFrame(PER); P.to_csv(f"{OUT}/simple_correlation.tsv", sep="\t", index=False, float_format="%.4g")
for ver, g in P.groupby("version"):
    ok = g.sp_lo > 0; ab = ok & (g.spearman_abs_z > g.atac_only_spearman_abs_z) & (g.spearman_abs_z > g.swapped_spearman_abs_z.fillna(-9))
    print(ver, "TFs", len(g), "median Spearman |z|", round(g.spearman_abs_z.median(), 4), "range", round(g.spearman_abs_z.min(), 3), round(g.spearman_abs_z.max(), 3), "interval > 0:", int(ok.sum()), "and above both controls:", int(ab.sum()), ",".join(g.tf[ab]))
print(P[P.version == "all_cells"].set_index("tf").loc[[t for t in ["NRF1", "CTCF", "NFYA", "MAZ", "ZNF143", "E4F1"] if t in set(P.tf)], ["spearman_abs_z", "sp_lo", "sp_hi", "spearman_signed_z"]].round(4).to_string())
