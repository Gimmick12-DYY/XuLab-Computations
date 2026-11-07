#!/usr/bin/env python
"""T4: aggregate obs/exp enrichment at reference-unique peaks.

For reference j with unique bins U_j and track i:
  L[i,j] = log2( (reads_i in U_j + 0.5) / reads_i in scope ) / ( pooled reads in U_j / pooled reads in scope ) )
pooled = all 79 TFs (raw) or mean of the 79 per-TF cell fractions (imputed).
Scopes: genome (universe) and atac (U_j and denominators restricted to ATAC union3, which removes
library-quality differences in the fraction of reads in open chromatin).
D = double-centred L (row and column means removed). Diagonal stat = mean diag(D) - mean offdiag(D),
exact permutation over all 5040 row-label shuffles.
Depth-matched null: all panel TFs thinned to ZNF777 depth (3 seeds, L averaged) and 20 pseudo-TFs, each a
thinning of the pooled panel to the same depth. Rows are centred across the references; for each
reference j, z_j = (own TF's centred value - mean over pseudo-TFs) / sd over pseudo-TFs, and the
same z for the other real TFs in column j (rank of own among the real TFs and the 20 pseudo-TFs).
Outputs OUT/spec/T4_*.tsv; summary rows.
"""
import numpy as np
import pandas as pd

from fig2_common import (NT, DM_DEPTH, DM_SEEDS, INP, OUT, TFS, append_summary, diag_stats, load_imp7, load_inputs,
                         load_raw7, raw7_dm, thin, write_mat)

NPSEUDO = 20
od = OUT / "spec"
od.mkdir(parents=True, exist_ok=True)
fr = load_inputs()
uni, atac = fr["universe"], fr["atac_union"]
U = fr["unique"]
raw7 = load_raw7().astype(np.float64)
pooled = np.load(INP / "pooled_raw.npy").astype(np.float64)
imp7 = load_imp7().astype(np.float64)
impfrac = np.load(INP / "imp79_frac.npy").astype(np.float64)


def Lmat(T, ref_track, scope):
    L = np.zeros((T.shape[0], NT))
    tot = T[:, scope].sum(1)
    ptot = ref_track[scope].sum()
    for j in range(NT):
        m = U[j] & scope
        L[:, j] = np.log2(((T[:, m].sum(1) + 0.5) / tot) / (ref_track[m].sum() / ptot))
    return L


def dcentre(L):
    return L - L.mean(1, keepdims=True) - L.mean(0, keepdims=True) + L.mean()


summ, zrows = [], []
pseudo = np.vstack([thin(pooled, DM_DEPTH, 9000 + k) for k in range(NPSEUDO)])
for scope_name, scope in (("genome", uni), ("atac", uni & atac)):
    variants = {
        "raw": Lmat(raw7, pooled, scope),
        "raw_dm": np.mean([Lmat(raw7_dm(s, raw7.astype(np.int64)), pooled, scope) for s in DM_SEEDS], 0),
        "imputed": Lmat(imp7, impfrac, scope),
    }
    for arm, L in variants.items():
        Dm = dcentre(L)
        write_mat(L, od / f"T4_L_{arm}_{scope_name}.tsv")
        write_mat(Dm, od / f"T4_D_{arm}_{scope_name}.tsv")
        summ.append(dict(test="T4_aggregate", arm=arm, variant=f"{scope_name}_doublecentred", **diag_stats(Dm),
                         mean_diag=float(np.diag(Dm).mean())))
        print(scope_name, arm, "L\n", pd.DataFrame(L, TFS, TFS).round(2).to_string(), "\nD\n",
              pd.DataFrame(Dm, TFS, TFS).round(2).to_string(), flush=True)
    # pseudo-TF null on the depth-matched rows
    Lp = Lmat(pseudo, pooled, scope)
    Lr = variants["raw_dm"]
    Rc = Lr - Lr.mean(1, keepdims=True)
    Pc = Lp - Lp.mean(1, keepdims=True)
    write_mat(Lp, od / f"T4_L_pseudo_{scope_name}.tsv", rows=[f"pseudo{k + 1}" for k in range(NPSEUDO)])
    mu, sd = Pc.mean(0), Pc.std(0, ddof=1)
    Z = (Rc - mu) / sd
    write_mat(Z, od / f"T4_z_vs_pseudo_{scope_name}.tsv")
    for j, tf in enumerate(TFS):
        allv = np.r_[Rc[:, j], Pc[:, j]]
        zrows.append(dict(scope=scope_name, reference=tf, own_z=Z[j, j],
                          max_other_real_z=float(np.max(np.delete(Z[:, j], j))),
                          own_rank_among_real=int((Rc[:, j] > Rc[j, j]).sum() + 1),
                          own_rank_among_all=int((allv > Rc[j, j]).sum() + 1),
                          pseudo_sd=sd[j], own_centred_log2=Rc[j, j]))
    zt = pd.DataFrame([r for r in zrows if r["scope"] == scope_name])
    d = diag_stats(Z)
    summ.append(dict(test="T4_aggregate", arm="raw_dm", variant=f"{scope_name}_z_vs_pseudo", **d,
                     mean_diag=float(np.diag(Z).mean()), n_own_z_gt2=int((zt.own_z > 2).sum())))
pd.DataFrame(zrows).to_csv(od / "T4_pseudo_z.tsv", sep="\t", index=False, float_format="%.4g")
append_summary(summ)
print(pd.DataFrame(zrows).to_string())
print(pd.DataFrame(summ).to_string())
