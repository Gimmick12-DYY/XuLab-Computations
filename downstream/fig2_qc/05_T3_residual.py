#!/usr/bin/env python
"""T3: pan-TF residual.

Raw: for TF i, Poisson GLM on 1 kb bins (fit on 500k random universe bins, applied to all):
  log E[y_i] = b0 + b1*log1p(P_-i) + b2*ATAC_union + b3*log1p(ATAC_fc)
with P_-i = pooled reads of the other 78 TFs, in reads per million of P_-i (depth-scaled).
Score = Pearson residual (y - mu)/sqrt(mu). raw_dm uses the thinned counts (3 seeds).
Imputed: same model with y = cells positive and P_-i = mean over the other 78 TFs of
(cells positive / n cells); fit within ATAC union3 (imputed is masked to it).
Then NTxNT AUROC/AUPRC vs all reference bins and vs reference-unique bins (T1 negatives), genome + atac.
Outputs OUT/spec/T3_{arm}_{refset}_{scope}_{auroc,auprc}.tsv, T3_glm_coef.tsv; summary rows.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from fig2_common import (NT, INP, OUT, TFS, append_summary, diag_stats, load_imp7, load_inputs, load_raw7, raw7_dm,
                         DM_SEEDS, score_matrix, write_mat)

od = OUT / "spec"
od.mkdir(parents=True, exist_ok=True)
fr = load_inputs()
uni, atac = fr["universe"], fr["atac_union"]
tf79 = list(fr["tf79"])
raw7 = load_raw7()
pooled = np.load(INP / "pooled_raw.npy").astype(np.float64)
imp7 = load_imp7().astype(np.float64)
impfrac = np.load(INP / "imp79_frac.npy").astype(np.float64) * len(tf79)
ncell7 = np.array([fr["ncells_imp79"][tf79.index(t)] for t in TFS])
afc = np.log1p(fr["atac_fc"].astype(np.float64))
au = atac.astype(np.float64)
rng = np.random.default_rng(3)
coefs = []


def resid(y, panel, fitmask, name):
    X = np.column_stack([np.ones_like(panel), np.log1p(panel), au, afc])
    idx = np.flatnonzero(fitmask)
    idx = rng.choice(idx, min(500_000, len(idx)), replace=False)
    m = sm.GLM(y[idx], X[idx], family=sm.families.Poisson()).fit()
    mu = np.exp(X @ m.params)
    coefs.append(dict(fit=name, **{f"b{k}": v for k, v in enumerate(m.params)}))
    return ((y - mu) / np.sqrt(mu)).astype(np.float32)


arms = {"raw": [], "raw_dm": [], "imputed": []}
full = np.zeros((NT, len(pooled)), np.float32)
for i, t in enumerate(TFS):
    P = pooled - raw7[i]
    full[i] = resid(raw7[i].astype(np.float64), P / P.sum() * 1e6, uni, f"raw_{t}")
arms["raw"].append(full)
for s in DM_SEEDS:
    th = raw7_dm(s, raw7)
    r = np.zeros_like(full)
    for i, t in enumerate(TFS):
        P = pooled - raw7[i]
        r[i] = resid(th[i].astype(np.float64), P / P.sum() * 1e6, uni, f"raw_dm{s}_{t}")
    arms["raw_dm"].append(r)
imp = np.zeros_like(full)
for i, t in enumerate(TFS):
    panel = (impfrac - imp7[i] / ncell7[i]) / (len(tf79) - 1) * 1e3   # per-mille of cells, mean of other 78
    imp[i] = resid(imp7[i], panel, uni & atac, f"imputed_{t}")
arms["imputed"].append(imp)
pd.DataFrame(coefs).to_csv(od / "T3_glm_coef.tsv", sep="\t", index=False, float_format="%.4g")

none = fr["refcount"] == 0
summ = []
for refset, pos, neg in (("all", fr["ref"], None), ("unique", fr["unique"], np.repeat(none[None], NT, 0))):
    for scope, keep in (("genome", uni), ("atac", uni & atac)):
        for arm, Ss in arms.items():
            mats = [score_matrix(S, pos, keep, neg=neg) for S in Ss]
            R = np.mean([m[0] for m in mats], 0)
            Pm = np.mean([m[1] for m in mats], 0)
            write_mat(R, od / f"T3_{arm}_{refset}_{scope}_auroc.tsv")
            write_mat(Pm, od / f"T3_{arm}_{refset}_{scope}_auprc.tsv")
            for met, M in (("auroc", R), ("auprc", Pm)):
                summ.append(dict(test=f"T3_residual_{refset}", arm=arm, variant=f"{scope}_{met}", **diag_stats(M),
                                 mean_diag=float(np.diag(M).mean())))
            print(refset, scope, arm, "\n", pd.DataFrame(R, TFS, TFS).round(3).to_string(), flush=True)
append_summary(summ)
print(pd.DataFrame(summ).to_string())
