#!/usr/bin/env python
"""Round 12 (user 2026-11-23 21:34Z): the main summit-group 6x6 (r2_05_pearson.py, set preferred_top5000, ZNF777 depth,
3 seeds) with the single-cell value WITHOUT the leave-one-out 79-TF pool. Everything else identical (same summits,
180 groups, bulk log2 CPM, thinning, Pearson across groups, 720 permutations, without CTCF, chromosome halves).
Variants of sc_i per group:
  pooled  (current)  log2((C_i+.5)/(F_i+.5)) - log2((C_pool-i+.5)/(F_pool-i+.5))   [recomputed as a check]
  nopool_cf          log2((C_i+.5)/(F_i+.5))            C = 5' ends within +-100 bp, F = 1-2 kb flanks
  nopool_cpm         log2(C_i / thinned total reads x 1e6 + 1)   (centre only, no flank)
Writes OUT/round12_nopool/{summary,colmeans}.tsv, m_<variant>_<half>.tsv and
paper_figures/fig2/source_data/corr_nopool_{cf,cpm}_6x6.tsv, corr_pooled_recomputed_6x6.tsv."""
import itertools
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/downstream/fig2_qc")
from fig2_common import DM_DEPTH, DM_SEEDS, OUT, ROOT, TFS  # noqa: E402

O = OUT / "round12_nopool"; O.mkdir(parents=True, exist_ok=True); SD = ROOT / "paper_figures/fig2/source_data"
G, NT = 30, len(TFS)
S = pd.read_csv(OUT / "round2/summits.tsv", sep="\t")
z = np.load(OUT / "round2/summit_counts.npz", allow_pickle=True)
C79, F79, tf79, tot79 = z["C"].astype(np.int64), z["F"].astype(np.int64), list(z["tf79"]), z["tot79"]
pi = [tf79.index(t) for t in TFS]; Cpool, Fpool = C79.sum(0), F79.sum(0)
Bk = S[[f"bulk_{t}" for t in TFS]].values
cn = np.array([23 if c == "chrX" else int(c[3:]) for c in S.chrom])
HM = {"all": np.ones(len(S), bool), "odd": (cn % 2 == 1) & (cn != 23), "even_X": (cn % 2 == 0) | (cn == 23)}


def groups(half):
    out = []
    for t in TFS:
        ix = np.flatnonzero((S.ref.values == t) & S.preferred.values.astype(bool))
        ix = ix[np.argsort(-S.signal.values[ix], kind="stable")][:5000]
        out += [g[HM[half][g]] for g in np.array_split(ix, G)]
    return [g for g in out if len(g)]


def perm(M):
    k = len(M); o = ~np.eye(k, dtype=bool); st = lambda A: A.diagonal().mean() - A[o].mean()
    obs = st(M); null = np.array([st(M[list(p)]) for p in itertools.permutations(range(k))])
    return float(obs), float(np.mean(null >= obs - 1e-12))


pf = np.minimum(1, DM_DEPTH / tot79[pi])
reps = [(np.random.default_rng(s).binomial(C79[pi], pf[:, None]), np.random.default_rng(s + 1).binomial(F79[pi], pf[:, None])) for s in DM_SEEDS]
rows, cm = [], []
for half in ("all", "odd", "even_X"):
    gr = groups(half)
    gB = np.array([Bk[g].sum(0) for g in gr]); bulk = np.log2(gB / gB.sum(0, keepdims=True) * 1e6 + 1)
    for var in ("pooled", "nopool_cf", "nopool_cpm"):
        Ms, SC = [], []
        for Cs, Fs in reps:
            sc = np.zeros((len(gr), NT))
            for i, k in enumerate(pi):
                for a, g in enumerate(gr):
                    c, f = Cs[i, g].sum(), Fs[i, g].sum()
                    if var == "pooled":
                        cp, fp = Cpool[g].sum() - C79[k, g].sum(), Fpool[g].sum() - F79[k, g].sum()
                        sc[a, i] = np.log2((c + .5) / (f + .5)) - np.log2((cp + .5) / (fp + .5))
                    elif var == "nopool_cf":
                        sc[a, i] = np.log2((c + .5) / (f + .5))
                    else:
                        sc[a, i] = np.log2(c / (tot79[k] * pf[i]) * 1e6 + 1)
            Ms.append(np.corrcoef(np.column_stack([bulk, sc]).T)[NT:, :NT]); SC.append(sc)
        M = np.mean(Ms, 0)
        obs, p = perm(M); _, p5 = perm(M[1:, 1:])
        T = pd.DataFrame(M, [f"{t} sc" for t in TFS], [f"{t} ChIP" for t in TFS]); T.to_csv(O / f"m_{var}_{half}.tsv", sep="\t", float_format="%.5f")
        if half == "all":
            T.to_csv(SD / {"pooled": "corr_pooled_recomputed_6x6.tsv", "nopool_cf": "corr_nopool_cf_6x6.tsv", "nopool_cpm": "corr_nopool_cpm_6x6.tsv"}[var],
                     sep="\t", float_format="%.5f")
            cm.append(dict(variant=var, **{f"rowmean_{t}": M[i].mean() for i, t in enumerate(TFS)}, **{f"colmean_{t}": M[:, j].mean() for j, t in enumerate(TFS)},
                           grand_mean=M.mean(), diag_mean=M.diagonal().mean(), off_mean=M[~np.eye(NT, dtype=bool)].mean()))
        rows.append(dict(variant=var, half=half, n_groups=len(gr), diag_minus_off=obs, perm_p=p, noCTCF_perm_p=p5,
                         own_best_rows=int(sum(np.argmax(M[i]) == i for i in range(NT))), own_best_cols=int(sum(np.argmax(M[:, j]) == j for j in range(NT))),
                         own_is_row_max=",".join(t for i, t in enumerate(TFS) if np.argmax(M[i]) == i), **{f"own_{t}": M[i, i] for i, t in enumerate(TFS)}))
Sm = pd.DataFrame(rows); Sm.to_csv(O / "summary.tsv", sep="\t", index=False, float_format="%.4f")
Cm = pd.DataFrame(cm); Cm.to_csv(O / "colmeans.tsv", sep="\t", index=False, float_format="%.4f")
cur = pd.read_csv(SD / "corr_matrix_6x6.tsv", sep="\t", index_col=0).values
print("check: recomputed pooled vs current corr_matrix_6x6 max abs diff", float(np.abs(pd.read_csv(SD / "corr_pooled_recomputed_6x6.tsv", sep="\t", index_col=0).values - cur).max()))
pd.set_option("display.width", 250)
print(Sm.round(3).to_string()); print(Cm.round(3).T.to_string())
for var in ("pooled", "nopool_cf", "nopool_cpm"):
    print(var); print(pd.read_csv(O / f"m_{var}_all.tsv", sep="\t", index_col=0).round(3).to_string())
