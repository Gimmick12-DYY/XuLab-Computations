# m22 Amendment 1, quick step: whole-matrix POINT ESTIMATES only (intervals and permutation p come from simple_correlation.py). Tables only. Diagnostic.
import json
import numpy as np, pandas as pd
from scipy.stats import rankdata
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m22_edge_table"; OUT = f"{O}/out"
W = np.load(f"{O}/work_matrices.npz", allow_pickle=True); tfs79 = list(W["tfs"]); UNIV = W["genes"].astype(str); NG = len(UNIV); S79 = W["score"]
Rz = np.load(f"{RG}/m9_abc_diagnosis/work/responses_arm2.npz", allow_pickle=True); KT = list(Rz["tfs"].astype(str)); gi = {g: k for k, g in enumerate(UNIV)}; OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
AT = pd.read_csv(f"{RG}/m11_peak_regulome/out/atac_x_abc_per_gene.tsv.gz", sep="\t").set_index("gene").atac_x_abc.reindex(UNIV).fillna(0).values
MIS = dict(zip([l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()], json.load(open(f"{RG}/m7_step2/features/raw_hicfixed/feature_qc.json"))["mis_paired_map"]["mis0"]))
def rowstd(M): mu = np.nanmean(M, 1, keepdims=True); sd = np.nanstd(M, 1, keepdims=True); sd[sd == 0] = np.nan; return (M - mu) / sd
def rowpct(M):
    out = np.full(M.shape, np.nan)
    for i in range(len(M)):
        m = np.isfinite(M[i])
        if m.sum(): out[i, m] = rankdata(M[i, m]) / m.sum()
    return out
def both(a, b):
    m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    return (np.nan, np.nan, int(m.sum())) if np.std(a) == 0 or np.std(b) == 0 else (float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]), float(np.corrcoef(a, b)[0, 1]), int(m.sum()))
T25 = pd.read_csv(f"{RG}/m23_sgrna_top25/out/observed_kd_top25.tsv.gz", sep="\t").pivot(index="tf", columns="gene", values="z_top25").reindex(columns=UNIV)
ROWS = []
for ver, T, Z in (("all_cells", KT, Rz["r1"].astype(float)), ("top25_sgRNA", list(T25.index), T25.values.astype(float))):
    keep = [i for i, t in enumerate(T) if t in tfs79]; T = [T[i] for i in keep]; Z = Z[keep].copy()
    for i, t in enumerate(T):
        for o in OWN.get(t, [t]):
            if o in gi: Z[i, gi[o]] = np.nan
    mats = {"real": np.vstack([S79[tfs79.index(t)] for t in T]), "swapped TF (m7 mis0, same depth tertile)": np.vstack([S79[tfs79.index(MIS[t])] if MIS.get(t) in tfs79 else np.full(NG, np.nan) for t in T]), "ATAC-only": np.tile(AT, (len(T), 1))}
    for target, Y in (("|z|", np.abs(Z)), ("signed z", Z)):
        for way, f in (("1 raw pooled", lambda M: M), ("2 row-standardised", rowstd), ("3 within-TF percentile", rowpct)):
            for nm, S in mats.items():
                S_ = np.where(np.isfinite(Z), S, np.nan); s, p, n = both(f(S_).ravel(), f(Y).ravel())
                ROWS.append(dict(label="diagnostic; point estimates (intervals in simple_correlation_whole_matrix.tsv)", version=ver, target=target, way=way, score_matrix=nm, n_tfs=len(T), n_entries=n, spearman=s, pearson=p))
R = pd.DataFrame(ROWS); R.to_csv(f"{OUT}/simple_correlation_whole_matrix_point.tsv", sep="\t", index=False, float_format="%.4f")
print(R.pivot_table(index=["version", "target", "way"], columns="score_matrix", values=["spearman", "pearson"]).round(4).to_string()); print(R.groupby("version")[["n_tfs", "n_entries"]].max().to_string())
