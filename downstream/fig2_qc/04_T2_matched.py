#!/usr/bin/env python
"""T2: accessibility/sequence-matched negatives.

Positives for reference j = reference-unique bins (as T1). For each positive, K=5 negatives are drawn
(with replacement only when a stratum is short) from universe bins overlapping no reference, in the
same stratum of: OmniATAC fold-enrichment 2-percentile (50 bins) x GC decile x CpG o/e decile x promoter (TSS+-1kb).
Quantiles are computed within the scope (genome: universe; atac: universe & ATAC union3).
The negative set is fixed per reference, so all tracks are scored on the same bins.
Outputs OUT/spec/T2_{arm}_{scope}_{auroc,auprc}.tsv, T2_match_qc.tsv; summary rows.
"""
import numpy as np
import pandas as pd

from fig2_common import (NT, OUT, TFS, append_summary, auc_pair, diag_stats, load_imp7, load_inputs, raw_arms,
                         write_mat)

K = 5
od = OUT / "spec"
od.mkdir(parents=True, exist_ok=True)
fr = load_inputs()
uni, atac = fr["universe"], fr["atac_union"]
ok = np.isfinite(fr["gc"]) & np.isfinite(fr["cpg_oe"])
arms = dict(raw_arms())
arms["imputed"] = [load_imp7()]


def dec(x, keep, n=10):
    q = np.unique(np.quantile(x[keep], np.linspace(0, 1, n + 1)[1:-1]))
    return np.searchsorted(q, x, side="right")


summ, qc = [], []
for scope, keep in (("genome", uni & ok), ("atac", uni & ok & atac)):
    key = (dec(fr["atac_fc"], keep, 50) * 1000 + dec(fr["gc"], keep) * 100 + dec(fr["cpg_oe"], keep) * 10
           + fr["promoter"].astype(int))
    negpool = np.flatnonzero(keep & (fr["refcount"] == 0))
    order = np.argsort(key[negpool], kind="stable")
    negpool = negpool[order]
    nk = key[negpool]
    ukeys, ustart, ucount = np.unique(nk, return_index=True, return_counts=True)
    rng = np.random.default_rng(7)
    sets = []
    for j, tf in enumerate(TFS):
        pos = np.flatnonzero(keep & fr["unique"][j])
        pk = key[pos]
        loc = np.searchsorted(ukeys, pk)
        has = (loc < len(ukeys)) & (ukeys[np.minimum(loc, len(ukeys) - 1)] == pk)
        pos, loc = pos[has], loc[has]
        st, cn = ustart[loc], ucount[loc]
        negs = np.concatenate([negpool[s + rng.choice(c, K, replace=c < K)] for s, c in zip(st, cn)])
        sets.append((pos, negs))
        qc.append(dict(scope=scope, reference=tf, n_pos=len(pos), n_pos_dropped=int((~has).sum()), n_neg=len(negs),
                       atac_fc_pos=float(np.median(fr["atac_fc"][pos])), atac_fc_neg=float(np.median(fr["atac_fc"][negs])),
                       gc_pos=float(np.mean(fr["gc"][pos])), gc_neg=float(np.mean(fr["gc"][negs])),
                       cpg_pos=float(np.mean(fr["cpg_oe"][pos])), cpg_neg=float(np.mean(fr["cpg_oe"][negs])),
                       prom_pos=float(fr["promoter"][pos].mean()), prom_neg=float(fr["promoter"][negs].mean())))
    for arm, Ss in arms.items():
        R = np.zeros((NT, NT)); P = np.zeros((NT, NT))
        for j, (pos, negs) in enumerate(sets):
            idx = np.concatenate([pos, negs])
            y = np.r_[np.ones(len(pos), bool), np.zeros(len(negs), bool)]
            for i in range(NT):
                a = np.mean([auc_pair(y, S[i][idx]) for S in Ss], 0)
                R[i, j], P[i, j] = a
        write_mat(R, od / f"T2_{arm}_{scope}_auroc.tsv")
        write_mat(P, od / f"T2_{arm}_{scope}_auprc.tsv")
        for met, M in (("auroc", R), ("auprc", P)):
            summ.append(dict(test="T2_matched", arm=arm, variant=f"{scope}_{met}", **diag_stats(M),
                             mean_diag=float(np.diag(M).mean())))
        print(scope, arm, "\n", pd.DataFrame(R, TFS, TFS).round(3).to_string(), flush=True)
pd.DataFrame(qc).to_csv(od / "T2_match_qc.tsv", sep="\t", index=False, float_format="%.4g")
append_summary(summ)
print(pd.DataFrame(qc).to_string())
print(pd.DataFrame(summ).to_string())
