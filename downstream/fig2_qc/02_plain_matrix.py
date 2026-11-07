#!/usr/bin/env python
"""Diagnostic 2b: plain matrix (sc track i vs bulk reference j), 1 kb bins.

Scores: raw = reads per bin (raw_dm = all panel TFs thinned to ZNF777 depth, mean of 3 seeds); imputed = cells positive per bin. Truth: bin overlaps reference j.
Scopes: genome (universe = chr1-22,X minus blacklist) and atac (universe & ATAC union3).
Also scores the OmniATAC fold-enrichment track against each reference as a baseline row.
Outputs OUT/diag/plain_{arm}_{scope}_{auroc,auprc}.tsv, plain_atac_baseline.tsv; summary rows.
"""
import numpy as np
import pandas as pd

from fig2_common import (NT, OUT, TFS, append_summary, diag_stats, load_imp7, load_inputs, raw_arms, score_matrix,
                         write_mat)

od = OUT / "diag"
od.mkdir(parents=True, exist_ok=True)
fr = load_inputs()
uni, atac = fr["universe"], fr["atac_union"]
arms = dict(raw_arms())
arms["imputed"] = [load_imp7()]
summ = []
base = {}
for scope, keep in (("genome", uni), ("atac", uni & atac)):
    b, bp = score_matrix(fr["atac_fc"][None, :], fr["ref"], keep)
    base[f"auroc_{scope}"], base[f"auprc_{scope}"] = b[0], bp[0]
    for arm, Ss in arms.items():
        mats = [score_matrix(S, fr["ref"], keep) for S in Ss]
        R = np.mean([m[0] for m in mats], 0)
        P = np.mean([m[1] for m in mats], 0)
        write_mat(R, od / f"plain_{arm}_{scope}_auroc.tsv")
        write_mat(P, od / f"plain_{arm}_{scope}_auprc.tsv")
        for met, M in (("auroc", R), ("auprc", P)):
            d = diag_stats(M)
            # column-centred view: AUPRC depends on prevalence, so also test on M / column mean
            summ.append(dict(test="plain", arm=arm, variant=f"{scope}_{met}", **d,
                             mean_diag=float(np.diag(M).mean())))
        print(scope, arm, "AUROC\n", pd.DataFrame(R, TFS, TFS).round(3).to_string(), flush=True)
        print(scope, arm, "AUPRC\n", pd.DataFrame(P, TFS, TFS).round(3).to_string(), flush=True)
pd.DataFrame(base, index=TFS).to_csv(od / "plain_atac_baseline.tsv", sep="\t", float_format="%.4g")
append_summary(summ)
print(pd.DataFrame(summ).to_string())
