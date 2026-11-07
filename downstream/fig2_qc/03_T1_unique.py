#!/usr/bin/env python
"""T1: reference-unique positives.

Positives for reference j = 1 kb bins overlapping a peak of j that overlaps no peak of the other 6
references (and the bin itself overlaps no other reference). Negatives = universe bins overlapping no
reference at all (bins of other references are dropped, not used as negatives).
Also scores the 'shared by >=3 references' bins (one column per track, vs the same negatives).
Scopes: genome and atac (ATAC union3 only). Arms: raw, raw_dm (ZNF777 depth, 3 seeds), imputed.
Outputs OUT/spec/T1_{arm}_{scope}_{auroc,auprc}.tsv, T1_shared3_auroc.tsv; summary rows.
"""
import numpy as np
import pandas as pd

from fig2_common import (NT, OUT, TFS, append_summary, auc_pair, diag_stats, load_imp7, load_inputs, raw_arms,
                         score_matrix, write_mat)

od = OUT / "spec"
od.mkdir(parents=True, exist_ok=True)
fr = load_inputs()
uni, atac = fr["universe"], fr["atac_union"]
none = fr["refcount"] == 0
neg = np.repeat(none[None, :], NT, 0)
arms = dict(raw_arms())
arms["imputed"] = [load_imp7()]
summ, sh_rows = [], []
for scope, keep in (("genome", uni), ("atac", uni & atac)):
    print(scope, "unique positives per ref:", dict(zip(TFS, (fr["unique"] & keep).sum(1))), flush=True)
    for arm, Ss in arms.items():
        mats = [score_matrix(S, fr["unique"], keep, neg=neg) for S in Ss]
        R = np.mean([m[0] for m in mats], 0)
        P = np.mean([m[1] for m in mats], 0)
        write_mat(R, od / f"T1_{arm}_{scope}_auroc.tsv")
        write_mat(P, od / f"T1_{arm}_{scope}_auprc.tsv")
        for met, M in (("auroc", R), ("auprc", P)):
            summ.append(dict(test="T1_unique", arm=arm, variant=f"{scope}_{met}", **diag_stats(M),
                             mean_diag=float(np.diag(M).mean())))
        sel = keep & (fr["shared3"] | none)
        y = fr["shared3"][sel]
        for i, tf in enumerate(TFS):
            a = np.mean([auc_pair(y, S[i][sel]) for S in Ss], 0)
            sh_rows.append(dict(scope=scope, arm=arm, track=tf, auroc_shared3=a[0], auprc_shared3=a[1],
                                n_pos=int(y.sum())))
        print(scope, arm, "\n", pd.DataFrame(R, TFS, TFS).round(3).to_string(), flush=True)
pd.DataFrame(sh_rows).to_csv(od / "T1_shared3.tsv", sep="\t", index=False, float_format="%.4g")
append_summary(summ)
print(pd.DataFrame(summ).to_string())
print(pd.DataFrame(sh_rows).to_string())
