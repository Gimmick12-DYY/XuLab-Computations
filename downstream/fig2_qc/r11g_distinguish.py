#!/usr/bin/env python
"""Round 11 follow-up (user 2026-11-23 22:53Z: the heatmap must show whether our 6 TFs can be told apart).
Rows = the round 11 ChIP-differential sites (same selection as r11_f_corr.py: per ChIP top 5,000 by LFC with its own
summit), one block per ChIP, ordered by ChIP LFC, aggregated into 50 consecutive groups per block. Values per group:
  raw      = the 6x6 value: log2((C+.5)/(F+.5)) - same for the 79-TF pool minus this TF; C = 5' ends +-100 bp, F = 1-2 kb
             flanks; our TF thinned to ZNF777 depth (seeds 11, 22, 33; averaged)
  imputed  = r25 recipe (Mdrop5; summit 1 kb bin vs flank bins 2-5 kb, minus the pool)
  ChIP     = log2 CPM (+1) of the block's own ChIP (rep 1 + rep 2) summed over the group's regions
Each of our columns is z-scored across all rows (removes each TF's overall offset). Summary: mean column z per (our TF,
ChIP block); own-block-is-max count; diagonal minus off-diagonal with an exact 720-permutation p over block labels.
Writes paper_figures/fig2/source_data/heatmap_chipdeseq2_distinguish.npz and O11/distinguish_summary.tsv."""
import itertools

import numpy as np
import pandas as pd
import scipy.sparse as sp

from r11_common import O11
from fig2_common import DM_DEPTH, DM_SEEDS, INP, ROOT, TFS, bin_index

SD = ROOT / "paper_figures/fig2/source_data"
NG, TOPN, CEN, FL0, FL1, E = 50, 5000, 100, 1000, 2000, 1e-4
Rg = pd.read_csv(O11 / "regions.tsv", sep="\t")
CNT = pd.read_csv(O11 / "counts.tsv.gz", sep="\t")
BULK = np.column_stack([CNT[f"{t}_rep1"].values + CNT[f"{t}_rep2"].values for t in TFS]).astype(float)
CPM = BULK / BULK.sum(0, keepdims=True) * 1e6
D = {t: pd.read_csv(O11 / f"deseq2_{t}.tsv.gz", sep="\t") for t in TFS}
sel = {}
for t in TFS:
    d = np.flatnonzero((np.nan_to_num(D[t].padj.values, nan=1) < 0.05) & (D[t].lfc.values > 1))
    ix = d[np.argsort(-D[t].lfc.values[d], kind="stable")][:TOPN]
    sel[t] = ix[Rg[f"summit_{t}"].values[ix] >= 0]          # ordered by LFC
site_tf = np.concatenate([[t] * len(sel[t]) for t in TFS]); site_reg = np.concatenate([sel[t] for t in TFS])
chrom = Rg.chrom.values[site_reg]; summ = np.array([Rg[f"summit_{t}"].values[r] for t, r in zip(site_tf, site_reg)], np.int64)
fr = np.load(INP / "frame.npz", allow_pickle=True); tf79 = [str(t) for t in fr["tf79"]]; ncell = fr["ncells_imp79"]
C79 = np.zeros((79, len(summ)), np.int64); F79 = np.zeros_like(C79); TOT = np.zeros(79)
for k, t in enumerate(tf79):
    z = np.load(INP / "five" / f"{t}.npz"); TOT[k] = sum(len(z[c]) for c in z.files)
    for c in np.unique(chrom):
        if c in z.files:
            q = z[c]; m = np.flatnonzero(chrom == c); s = summ[m]
            cnt = lambda a, b: np.searchsorted(q, s + b, side="right") - np.searchsorted(q, s + a, side="left")
            C79[k, m] = cnt(-CEN, CEN); F79[k, m] = cnt(-FL1, -FL0 - 1) + cnt(FL0 + 1, FL1)
pi = [tf79.index(t) for t in TFS]; Cpool, Fpool = C79.sum(0), F79.sum(0)
bi, off, N = bin_index()
b0 = np.array([off[c][0] for c in chrom]) + summ // 1000
nb = np.array([off[c][1] for c in chrom]); lo0 = np.array([off[c][0] for c in chrom])
fb = np.clip(b0[:, None] + np.r_[-5:-1, 2:6][None, :], lo0[:, None], lo0[:, None] + nb[:, None] - 1)
IMP = sp.load_npz(INP / "imp79_mdrop5.npz").tocsr()
Ci = np.zeros((79, len(summ))); Fi = np.zeros((79, len(summ)))
for k in range(79):
    v = IMP[k].toarray().ravel() / ncell[k]; Ci[k] = v[b0]; Fi[k] = v[fb].mean(1)
# groups: 50 consecutive (by LFC) per block
groups, gblk, st = [], [], 0
for b, t in enumerate(TFS):
    ix = np.arange(st, st + len(sel[t])); st += len(sel[t])
    for g in np.array_split(ix, min(NG, len(ix))):
        groups.append(g); gblk.append(b)
gblk = np.array(gblk)
chip = np.array([np.log2(CPM[site_reg[g], gblk[a]].sum() + 1) for a, g in enumerate(groups)])
raw = np.zeros((len(groups), 6))
for seed in DM_SEEDS:
    pf = np.minimum(1, DM_DEPTH / TOT[pi])[:, None]
    Cs = np.random.default_rng(seed).binomial(C79[pi], pf); Fs = np.random.default_rng(seed + 1).binomial(F79[pi], pf)
    for i, k in enumerate(pi):
        for a, g in enumerate(groups):
            c, f = Cs[i, g].sum(), Fs[i, g].sum(); cp, fp = Cpool[g].sum() - C79[k, g].sum(), Fpool[g].sum() - F79[k, g].sum()
            raw[a, i] += (np.log2((c + .5) / (f + .5)) - np.log2((cp + .5) / (fp + .5))) / len(DM_SEEDS)
imp = np.zeros((len(groups), 6)); Cip, Fip = Ci.sum(0), Fi.sum(0)
for i, k in enumerate(pi):
    for a, g in enumerate(groups):
        c, f = Ci[k, g].sum(), Fi[k, g].sum(); cp, fp = Cip[g].sum() - c, Fip[g].sum() - f
        imp[a, i] = np.log2((c + E) / (f + E)) - np.log2((cp + E) / (fp + E))
zs = lambda X: (X - X.mean(0)) / X.std(0)
rawz, impz = zs(raw), zs(imp)


def perm(M):
    o = ~np.eye(6, dtype=bool); s = lambda A: A.diagonal().mean() - A[o].mean()
    obs = s(M); null = np.array([s(M[list(p)]) for p in itertools.permutations(range(6))])
    return float(obs), float(np.mean(null >= obs - 1e-12))


rows, summ_m = [], {}
for arm, Z in (("raw", rawz), ("imputed", impz)):
    M = np.array([[Z[gblk == b, i].mean() for b in range(6)] for i in range(6)])   # our TF x ChIP block
    summ_m[arm] = M
    obs, p = perm(M); _, p5 = perm(M[1:, 1:]) if False else (None, None)
    o5 = ~np.eye(5, dtype=bool); s5 = lambda A: A.diagonal().mean() - A[o5].mean()
    M5 = M[1:, 1:]; null5 = np.array([s5(M5[list(q)]) for q in itertools.permutations(range(5))]); p5 = float(np.mean(null5 >= s5(M5) - 1e-12))
    rank = {t: int(1 + (M[i] > M[i, i]).sum()) for i, t in enumerate(TFS)}   # rank of own block within our TF's row
    rows.append(dict(arm=arm, diag_minus_off=obs, perm_p=p, noCTCF_perm_p=p5, own_block_is_max=sum(r == 1 for r in rank.values()),
                     **{f"rank_{t}": r for t, r in rank.items()}))
    print(arm); print(pd.DataFrame(M, [f"{t} (ours)" for t in TFS], [f"{t} block" for t in TFS]).round(2).to_string())
S = pd.DataFrame(rows); S.to_csv(O11 / "distinguish_summary.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 250); print(S.to_string())
np.savez_compressed(SD / "heatmap_chipdeseq2_distinguish.npz", tfs=np.array(TFS), chip=chip, rawz=rawz, impz=impz, gblk=gblk,
                    block_n=np.array([len(sel[t]) for t in TFS]), M_raw=summ_m["raw"], M_imp=summ_m["imputed"],
                    stats=np.array([[r["diag_minus_off"], r["perm_p"], r["noCTCF_perm_p"], r["own_block_is_max"]] for r in rows]))
print("done")
