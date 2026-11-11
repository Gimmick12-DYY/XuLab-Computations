# m11_peak_regulome Amendment 1: entry-wise comparison of the peak matrix P with the knockdown matrix Z (diagnostic, benchmark already read).
# usage: m11_compare.py -> compare_all_entries.tsv, compare_per_tf.tsv, compare_per_tf_summary.tsv, compare_per_gene_summary.tsv, compare_top_genes.tsv, aligned/ (P, Z, long table)
import json, os
import numpy as np, pandas as pd
from scipy.stats import rankdata
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m11_peak_regulome"; OUT = f"{O}/out"; AL = f"{OUT}/aligned"; os.makedirs(AL, exist_ok=True)
LAB = "diagnostic, benchmark already read"; SEED = 20261111; rng = np.random.default_rng(SEED); NP = 200; NB = 2000
ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; panel = lambda t: ALIAS.get(t, t); OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
tfs79 = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]; MIS = dict(zip(tfs79, json.load(open(f"{RG}/m7_step2/features/raw_hicfixed/feature_qc.json"))["mis_paired_map"]["mis0"]))
LEAD = ["NRF1", "CTCF", "ZNF143", "MAZ", "NFYA", "E4F1"]; TESTABLE = {"NRF1", "CTCF", "MAZ", "MSX2", "RBPJ", "SETDB1", "ZNF143", "ZNF282", "ZNF746"}
GL = pd.read_csv(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/GeneList.txt", sep="\t", usecols=["name", "Ensembl_ID"]).drop_duplicates("name", keep="last")
ens2sym = dict(zip(GL.Ensembl_ID.astype(str).str.split(".").str[0], GL.name))
def load_z(arm):
    d = np.load(f"{RG}/m4_validation/kd_response/out_v2/{arm}/kd_response.npz", allow_pickle=True); q = pd.read_csv(f"{RG}/m4_validation/kd_response/out_v2/{arm}/guide_qc.tsv", sep="\t")
    u = d["units"].tolist(); ui = {x: k for k, x in enumerate(u)}; keep = d["ntc_detect"] >= .1; g = pd.Series(d["genes"][keep].astype(str)); Z = d["z"][:, keep].astype(float)
    ens = g.str.match(r"^ENSG\d+"); g2 = g.where(~ens, g.str.split(".").str[0].map(ens2sym).fillna(g)); rec = int((ens & (g2 != g)).sum())
    prim = q[(q.kept == 1) & q.kind.isin(["all", "construct"])]; tf_ = sorted(prim.tf.unique()); P_ = np.vstack([np.nanmean(Z[[ui[x] for x in prim.unit[prim.tf == t]]], axis=0) for t in tf_])
    pts = sorted({panel(t) for t in tf_}); M = np.vstack([P_[[i for i, t in enumerate(tf_) if panel(t) == p]].mean(0) for p in pts])
    df = pd.DataFrame(M, index=pts, columns=g2.values); df = df.loc[:, ~df.columns.duplicated(keep="last")]
    return df, dict(arm=arm, kd_genes_ge10pct=len(g), ensembl_named=int(ens.sum()), ensembl_recovered_to_symbol=rec, kd_rows=",".join(tf_))
def sp_vec(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or np.std(a[m]) == 0 or np.std(b[m]) == 0: return np.nan
    return float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1])
def pe_vec(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or np.std(a[m]) == 0 or np.std(b[m]) == 0: return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])
def col_sp(A, B):  # Spearman per column (genes), along rows (TFs); NaN-aware by masking with B
    out = np.full(A.shape[1], np.nan)
    for j in range(A.shape[1]): out[j] = sp_vec(A[:, j], B[:, j])
    return out
def bmean(v):
    v = v[np.isfinite(v)]; b = v[rng.integers(0, len(v), (NB, len(v)))].mean(1); return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
ATAC = pd.read_csv(f"{OUT}/atac_x_abc_per_gene.tsv.gz", sep="\t").set_index("gene").atac_x_abc
ALL = []; PTF = []; PTFS = []; PGS = []; TOP = []; JOIN = []
for arm in ("arm2_all_cell", "arm1_high_knockdown"):
    Zdf, j = load_z(arm); JOIN.append(j)
    for ver in ("as_called_p0.01", "q0.05"):
        for part in ("score", "promoter", "distal"):
            if arm != "arm2_all_cell" and not (ver == "as_called_p0.01" and part == "score"): continue
            Pdf = pd.read_csv(f"{OUT}/wide_{part}_{ver}.tsv.gz", sep="\t", index_col=0)
            tfs = [t for t in Pdf.index if t in Zdf.index]; genes = [g for g in Pdf.columns if g in Zdf.columns]
            P = Pdf.loc[tfs, genes].values; Zr = Zdf.loc[tfs, genes].values.copy(); gi = {g: k for k, g in enumerate(genes)}
            for i, t in enumerate(tfs):
                for o in OWN.get(t, [t]):
                    if o in gi: Zr[i, gi[o]] = np.nan
            Zc = Zr - np.nanmean(Zr, 0, keepdims=True); Zc = Zc - np.nanmean(Zc, 1, keepdims=True)
            READ = {"z": Zr, "abs_z": np.abs(Zr), "doublecentred_z": Zc}; A = np.tile(ATAC.reindex(genes).fillna(0).values, (len(tfs), 1))
            Pall = Pdf.loc[:, genes]; pooled = (Pall.sum(0).values[None, :] - P); sw = np.vstack([Pall.loc[MIS[t]].values if MIS.get(t) in Pall.index else np.full(len(genes), np.nan) for t in tfs])
            if arm == "arm2_all_cell" and ver == "as_called_p0.01" and part == "score":
                Pp = pd.read_csv(f"{OUT}/wide_promoter_{ver}.tsv.gz", sep="\t", index_col=0).loc[tfs, genes].values; Pd = pd.read_csv(f"{OUT}/wide_distal_{ver}.tsv.gz", sep="\t", index_col=0).loc[tfs, genes].values
                pd.DataFrame(P, index=tfs, columns=genes).to_csv(f"{AL}/P_signal_x_abc_as_called.tsv.gz", sep="\t", float_format="%.5g"); pd.DataFrame(Zr, index=tfs, columns=genes).to_csv(f"{AL}/Z_arm2_raw_z.tsv.gz", sep="\t", float_format="%.4g")
                Pq = pd.read_csv(f"{OUT}/wide_score_q0.05.tsv.gz", sep="\t", index_col=0).loc[tfs, genes].values; pd.DataFrame(Pq, index=tfs, columns=genes).to_csv(f"{AL}/P_signal_x_abc_q0.05.tsv.gz", sep="\t", float_format="%.5g")
                pd.DataFrame(dict(tf=np.repeat(tfs, len(genes)), gene=np.tile(genes, len(tfs)), P_as_called=P.ravel(), P_promoter=Pp.ravel(), P_distal=Pd.ravel(), P_q0_05=Pq.ravel(), z=Zr.ravel())).to_csv(f"{AL}/long_P_Z.tsv.gz", sep="\t", index=False, float_format="%.5g")
            for rn, Y in READ.items():
                tag = dict(label=LAB, arm=arm, P_version=ver, P_part=part, reading=rn, n_tfs=len(tfs), n_genes=len(genes))
                # (1) all entries
                y = Y.ravel(); r = dict(tag, stat="all_entries", spearman=sp_vec(P.ravel(), y), pearson_log1p=pe_vec(np.log1p(P.ravel()), y), spearman_P_pos_only=sp_vec(np.where(P.ravel() > 0, P.ravel(), np.nan), y),
                                       ctrl_atac=sp_vec(A.ravel(), y), ctrl_pooled_other78=sp_vec(pooled.ravel(), y), ctrl_swapped=sp_vec(sw.ravel(), y))
                rp = rankdata(P.ravel()).reshape(P.shape); m = np.isfinite(y); ry = rankdata(y[m]); nul = []
                for _ in range(NP): x = rp[rng.permutation(len(tfs))].ravel()[m]; nul.append(np.corrcoef(rankdata(x), ry)[0, 1])
                r.update(ctrl_rowperm_mean=float(np.mean(nul)), ctrl_rowperm_p2_5=float(np.percentile(nul, 2.5)), ctrl_rowperm_p97_5=float(np.percentile(nul, 97.5))); ALL.append(r)
                # (2) per TF along genes
                rows = []
                for i, t in enumerate(tfs):
                    pz = np.abs(Y[i]); bnd = P[i] > 0
                    rows.append(dict(tag, tf=t, testable=t in TESTABLE, n_genes_P_pos=int(bnd.sum()), spearman=sp_vec(P[i], Y[i]), pearson_log1p=pe_vec(np.log1p(P[i]), Y[i]), spearman_P_pos_only=sp_vec(np.where(bnd, P[i], np.nan), Y[i]),
                                     ctrl_atac=sp_vec(A[i], Y[i]), ctrl_pooled_other78=sp_vec(pooled[i], Y[i]), ctrl_swapped=sp_vec(sw[i], Y[i]), swap_tf=MIS.get(t),
                                     bound_minus_unbound_mean_absz=float(np.nanmean(pz[bnd]) - np.nanmean(pz[~bnd])) if bnd.any() and (~bnd).any() else np.nan))
                R = pd.DataFrame(rows); PTF.append(R)
                for col in ("spearman", "spearman_P_pos_only", "ctrl_atac", "ctrl_pooled_other78", "ctrl_swapped", "bound_minus_unbound_mean_absz"):
                    v = R[col].values.astype(float); mu, lo, hi = bmean(v); PTFS.append(dict(tag, statistic=col, median=float(np.nanmedian(v)), q25=float(np.nanpercentile(v, 25)), q75=float(np.nanpercentile(v, 75)), frac_gt0=float(np.nanmean(v > 0)), mean=mu, mean_lo=lo, mean_hi=hi))
                # (3) per gene along TFs, genes with P > 0 in >= 5 TFs
                gsel = (P > 0).sum(0) >= 5
                if gsel.sum() >= 20:
                    obs = col_sp(P[:, gsel], Y[:, gsel]); sws = col_sp(sw[:, gsel], Y[:, gsel]); mu, lo, hi = bmean(obs); nm = []
                    for _ in range(min(NP, 100)): nm.append(np.nanmean(col_sp(P[rng.permutation(len(tfs))][:, gsel], Y[:, gsel])))
                    PGS.append(dict(tag, n_genes_ge5_bound=int(gsel.sum()), median=float(np.nanmedian(obs)), q25=float(np.nanpercentile(obs, 25)), q75=float(np.nanpercentile(obs, 75)), frac_gt0=float(np.nanmean(obs > 0)), mean=mu, mean_lo=lo, mean_hi=hi,
                                    ctrl_swapped_mean=float(np.nanmean(sws)), ctrl_rowperm_mean=float(np.mean(nm)), ctrl_rowperm_p2_5=float(np.percentile(nm, 2.5)), ctrl_rowperm_p97_5=float(np.percentile(nm, 97.5)),
                                    note="ATAC control undefined (identical across TFs); pooled-other control = column total minus P, i.e. -P up to a constant per gene"))
                    if arm == "arm2_all_cell" and ver == "as_called_p0.01" and part == "score" and rn == "z":
                        gg = np.array(genes)[gsel]; o = np.argsort(-np.nan_to_num(obs, nan=-9))[:30]
                        for k in o: TOP.append(dict(label=LAB, gene=gg[k], n_tfs_bound=int((P[:, gsel][:, k] > 0).sum()), spearman_P_vs_z_across_tfs=float(obs[k])))
            print(arm, ver, part, flush=True)
pd.DataFrame(ALL).to_csv(f"{O}/compare_all_entries.tsv", sep="\t", index=False, float_format="%.4f"); pd.concat(PTF).to_csv(f"{O}/compare_per_tf.tsv", sep="\t", index=False, float_format="%.4f")
pd.DataFrame(PTFS).to_csv(f"{O}/compare_per_tf_summary.tsv", sep="\t", index=False, float_format="%.4f"); pd.DataFrame(PGS).to_csv(f"{O}/compare_per_gene_summary.tsv", sep="\t", index=False, float_format="%.4f")
pd.DataFrame(TOP).to_csv(f"{O}/compare_top_genes.tsv", sep="\t", index=False, float_format="%.4f"); pd.DataFrame(JOIN).to_csv(f"{O}/compare_join.tsv", sep="\t", index=False); print("done")
