#!/usr/bin/env python
"""Round 9b step e (SPEC.md 6, 8): 6x6 Pearson r (our shrunken LFC vs ChIP log2FC) for both arms with statistics, and
the round 10 presence blocks re-scored with these LFCs. Writes OUT/round9b_deseq2_multigroup/{summary,blocks}.tsv and
paper_figures/fig2/source_data/corr_deseq2mg_6x6{,_diffunion,_chip1kb}.tsv."""
import itertools
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/downstream/fig2_qc")
from fig2_common import OUT, ROOT, TFS  # noqa: E402

O = OUT / "round9b_deseq2_multigroup"; SD = ROOT / "paper_figures/fig2/source_data"; R10 = OUT / "round10_presence"
ARMS = {"consensus": (OUT / "round9_deseq2", O / "consensus"), "chip1kb": (O / "chip1kb", O / "chip1kb")}
B4 = ["CTCF", "ZBTB7A", "ZIC2", "ZNF282"]


def perm(M):
    k = len(M); o = ~np.eye(k, dtype=bool); st = lambda A: A.diagonal().mean() - A[o].mean()
    obs = st(M); null = np.array([st(M[list(p)]) for p in itertools.permutations(range(k))])
    return float(obs), float(np.mean(null >= obs - 1e-12))


# presence rows (round 10 main set) and their single-ChIP block
rows = pd.read_csv(R10 / "rows_main_kept.bed", sep="\t", header=None, usecols=[0, 1, 2])
def cover(b):
    out = subprocess.run(f"bedtools intersect -c -a {R10 / 'rows_main_kept.bed'} -b {b}", shell=True, check=True,
                         capture_output=True, text=True).stdout
    return np.array([int(l.split("\t")[-1]) > 0 for l in out.splitlines()])
chip4 = np.column_stack([cover(R10 / f"chip_{t}.bed") for t in B4])
single = np.where(chip4.sum(1) == 1, chip4.argmax(1), -1)

summ, blk = [], []
for arm, (src, res) in ARMS.items():
    pk = pd.read_csv(src / "peaks.bed", sep="\t", header=None)
    cn = pk[0].str.replace("chr", "").replace("X", "23").astype(int).values
    H = {"all": np.ones(len(pk), bool), "odd": (cn % 2 == 1) & (cn != 23), "even_X": (cn % 2 == 0) | (cn == 23)}
    FC = pd.read_csv(src / "chip_log2fc.tsv.gz", sep="\t")[TFS].values
    D = {t: pd.read_csv(res / f"deseq2mg_{t}.tsv.gz", sep="\t") for t in TFS}
    OUR = np.column_stack([D[t].log2FC_shrunk.values for t in TFS]); PADJ = np.column_stack([D[t].padj.values for t in TFS])
    ourdiff = (np.nan_to_num(PADJ, nan=1) < 0.05) & (OUR > 1); chipdiff = FC > 1
    for setname in ("all_peaks", "diff_union"):
        for half, hm in H.items():
            M = np.zeros((6, 6)); N = np.zeros((6, 6), int)
            for i in range(6):
                for j in range(6):
                    m = hm & np.isfinite(OUR[:, i]) & ((ourdiff[:, i] | chipdiff[:, j]) if setname == "diff_union" else True)
                    M[i, j] = np.corrcoef(OUR[m, i], FC[m, j])[0, 1] if m.sum() > 2 else np.nan; N[i, j] = m.sum()
            obs, p = perm(np.nan_to_num(M)); _, p5 = perm(np.nan_to_num(M[1:, 1:]))
            summ.append(dict(arm=arm, set=setname, half=half, diag_minus_off=obs, perm_p=p, noCTCF_perm_p=p5,
                             own_best_rows=int(sum(np.nanargmax(M[i]) == i for i in range(6))),
                             own_best_cols=int(sum(np.nanargmax(M[:, j]) == j for j in range(6))),
                             own_is_row_max=",".join(t for i, t in enumerate(TFS) if np.nanargmax(M[i]) == i),
                             **{f"own_{t}": M[i, i] for i, t in enumerate(TFS)}, n_min=int(N.min()), n_max=int(N.max())))
            T = pd.DataFrame(M, [f"sc {t}" for t in TFS], [f"Bulk {t}" for t in TFS])
            T.to_csv(res / f"m_{setname}_{half}.tsv", sep="\t", float_format="%.5f")
            if half == "all":
                name = {("consensus", "all_peaks"): "corr_deseq2mg_6x6.tsv", ("consensus", "diff_union"): "corr_deseq2mg_6x6_diffunion.tsv",
                        ("chip1kb", "all_peaks"): "corr_deseq2mg_6x6_chip1kb.tsv", ("chip1kb", "diff_union"): "corr_deseq2mg_6x6_chip1kb_diffunion.tsv"}[(arm, setname)]
                T.to_csv(SD / name, sep="\t", float_format="%.5f")
    # block check: each presence row takes the LFC of the first region it overlaps
    out = subprocess.run(f"bedtools intersect -wa -wb -loj -a {R10 / 'rows_main_kept.bed'} -b {src / 'peaks.bed'}", shell=True,
                         check=True, capture_output=True, text=True).stdout
    X = pd.read_csv(pd.io.common.StringIO(out), sep="\t", header=None)
    X["key"] = X[0] + ":" + X[1].astype(str) + "-" + X[2].astype(str); X = X.drop_duplicates("key")
    idmap = {v: i for i, v in enumerate(pk[3])}
    keys = rows[0] + ":" + rows[1].astype(str) + "-" + rows[2].astype(str)
    hit = X.set_index("key")[X.columns[-2]].reindex(keys).map(idmap)
    ok = hit.notna().values; hi = hit.fillna(0).astype(int).values
    for b, bt in enumerate(B4):
        m = ok & (single == b)
        blk.append(dict(arm=arm, block=f"{bt} only", n=int(m.sum()),
                        **{f"lfc_{t}": float(np.nanmean(OUR[hi[m], TFS.index(t)])) for t in B4}))
S = pd.DataFrame(summ); S.to_csv(O / "summary.tsv", sep="\t", index=False, float_format="%.4f")
Bk = pd.DataFrame(blk); Bk.to_csv(O / "blocks.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 250)
print(S.round(3).to_string()); print(Bk.round(3).to_string())
