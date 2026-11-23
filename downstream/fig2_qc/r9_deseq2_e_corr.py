#!/usr/bin/env python
"""Round 9 step e: 6x6 Pearson r between our shrunken log2FC (DESeq2) and the ChIP log2FC (SPEC.md 5); statistics;
writes OUT/round9_deseq2/summary.tsv, m_*.tsv and paper_figures/fig2/source_data/corr_deseq2_6x6{,_diffunion}.tsv."""
import itertools
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/downstream/fig2_qc")
from fig2_common import OUT, ROOT, TFS  # noqa: E402

O9 = OUT / "round9_deseq2"; SD = ROOT / "paper_figures/fig2/source_data"
pk = pd.read_csv(O9 / "peaks.bed", sep="\t", header=None)
cn = pk[0].str.replace("chr", "").replace("X", "23").astype(int).values
H = {"all": np.ones(len(pk), bool), "odd": (cn % 2 == 1) & (cn != 23), "even_X": (cn % 2 == 0) | (cn == 23)}
FC = pd.read_csv(O9 / "chip_log2fc.tsv.gz", sep="\t")[TFS].values
OUR = np.column_stack([pd.read_csv(O9 / f"deseq2_{t}.tsv.gz", sep="\t").log2FC_shrunk.values for t in TFS])
PADJ = np.column_stack([pd.read_csv(O9 / f"deseq2_{t}.tsv.gz", sep="\t").padj.values for t in TFS])
ourdiff = (np.nan_to_num(PADJ, nan=1) < 0.05) & (OUR > 1); chipdiff = FC > 1


def perm(M):
    k = len(M); o = ~np.eye(k, dtype=bool); st = lambda A: A.diagonal().mean() - A[o].mean()
    obs = st(M); null = np.array([st(M[list(p)]) for p in itertools.permutations(range(k))])
    return float(obs), float(np.mean(null >= obs - 1e-12))


rows = []
for setname in ("all_peaks", "diff_union"):
    for half, hm in H.items():
        M = np.zeros((6, 6)); N = np.zeros((6, 6), int)
        for i in range(6):
            for j in range(6):
                m = hm & np.isfinite(OUR[:, i]) & ((ourdiff[:, i] | chipdiff[:, j]) if setname == "diff_union" else True)
                M[i, j] = np.corrcoef(OUR[m, i], FC[m, j])[0, 1]; N[i, j] = m.sum()
        obs, p = perm(M); _, p5 = perm(M[1:, 1:])
        rows.append(dict(set=setname, half=half, diag_minus_off=obs, perm_p=p, noCTCF_perm_p=p5,
                         own_best_rows=int(sum(np.argmax(M[i]) == i for i in range(6))), own_best_cols=int(sum(np.argmax(M[:, j]) == j for j in range(6))),
                         own_is_row_max=",".join(t for i, t in enumerate(TFS) if np.argmax(M[i]) == i),
                         **{f"own_{t}": M[i, i] for i, t in enumerate(TFS)}, n_peaks_min=int(N.min()), n_peaks_max=int(N.max())))
        D = pd.DataFrame(M, [f"sc {t}" for t in TFS], [f"Bulk {t}" for t in TFS])
        D.to_csv(O9 / f"m_{setname}_{half}.tsv", sep="\t", float_format="%.5f")
        if half == "all":
            D.to_csv(SD / ("corr_deseq2_6x6.tsv" if setname == "all_peaks" else "corr_deseq2_6x6_diffunion.tsv"), sep="\t", float_format="%.5f")
T = pd.DataFrame(rows); T.to_csv(O9 / "summary.tsv", sep="\t", index=False, float_format="%.4f")
nd = pd.DataFrame({"TF": TFS, "our_differential": ourdiff.sum(0), "chip_differential_log2FC_gt1": chipdiff.sum(0)})
nd.to_csv(O9 / "n_differential.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); print(nd.to_string()); print(T.round(3).to_string())
