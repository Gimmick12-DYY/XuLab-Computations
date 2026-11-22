#!/usr/bin/env python
"""corr_matrix_6x6 recipe with IMPUTED single-cell tracks under the accessibility mask (MASK = "Mdrop5").
Same 180 summit groups (preferred top-5,000 per reference, 30 groups each; round2/summits.tsv) and the same bulk
values (log2 CPM of ChIP reads within +-250 bp per group) as fig2_corr_matrix_6x6. Imputed at 1 kb:
  sc_i(group) = log2((C_i + e)/(F_i + e)) - log2((C_pool-i + e)/(F_pool-i + e))
  C = fraction of cells positive in the 1 kb bin holding each summit (summed over the group's summits);
  F = mean over the flank bins 2-5 kb away on both sides (8 bins), summed over summits; pool-i = the other 78 TFs
  (fractions summed); e = 1e-4 (fraction-of-cells scale).
Arms (inputs from r20a_imp_mdrop5.py): "all" = raw-positive cells + model calls inside Mdrop5 (inputs/imp79_mdrop5.npz);
"added" = model calls inside Mdrop5 only (inputs/imp79_mdrop5_added.npz).
Stats: own r, diag - off, exact 720-permutation p, without CTCF (120), odd / even+X chromosome halves.
Outputs OUT/fig2_imputed_corr/: m_<arm>_<half>.tsv, summary.tsv; source copy for paper_figures/fig2/source_data/."""
import itertools
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/downstream/fig2_qc")
from fig2_common import INP, OUT, ROOT, TFS, bin_index  # noqa: E402

MASK = "Mdrop5"
OD = OUT / "fig2_imputed_corr"; OD.mkdir(parents=True, exist_ok=True)
SD = ROOT / "paper_figures/fig2/source_data"
G, NT, E = 30, 6, 1e-4
S = pd.read_csv(OUT / "round2/summits.tsv", sep="\t")
Bk = S[[f"bulk_{t}" for t in TFS]].values
bi, off, N = bin_index()
fr = np.load(INP / "frame.npz", allow_pickle=True)
tf79 = [str(t) for t in fr["tf79"]]; ncell = fr["ncells_imp79"]
b0 = np.array([off[c][0] for c in S.chrom]) + S.pos.values // 1000
nb = np.array([off[c][1] for c in S.chrom]); lo = np.array([off[c][0] for c in S.chrom])
FL = np.r_[-5:-1, 2:6]
fb = np.clip(b0[:, None] + FL[None, :], lo[:, None], lo[:, None] + nb[:, None] - 1)


def groups(half):
    cn = np.array([23 if c == "chrX" else int(c[3:]) for c in S.chrom])
    hm = {"all": np.ones(len(S), bool), "odd": (cn % 2 == 1) & (cn != 23), "even_X": (cn % 2 == 0) | (cn == 23)}[half]
    out = []
    for t in TFS:
        ix = np.flatnonzero((S.ref.values == t) & S.preferred.values.astype(bool))
        ix = ix[np.argsort(-S.signal.values[ix], kind="stable")][:5000]   # same 180 groups as the main matrix
        out += [g[hm[g]] for g in np.array_split(ix, G)]
    return [g for g in out if len(g)]


def perm(M):
    k = len(M); o = ~np.eye(k, dtype=bool); st = lambda A: A.diagonal().mean() - A[o].mean()
    obs = st(M); null = np.array([st(M[list(p)]) for p in itertools.permutations(range(k))])
    return float(obs), float(np.mean(null >= obs - 1e-12))


rows = []
for arm, fn in (("all", "imp79_mdrop5.npz"), ("added", "imp79_mdrop5_added.npz")):
    imp = sp.load_npz(INP / fn).tocsr()
    Cs = np.zeros((79, len(S))); Fs = np.zeros((79, len(S)))
    for k in range(79):
        v = imp[k].toarray().ravel() / ncell[k]
        Cs[k] = v[b0]; Fs[k] = v[fb].mean(1)
    Cp, Fp = Cs.sum(0), Fs.sum(0)
    pi = [tf79.index(t) for t in TFS]
    for half in ("all", "odd", "even_X"):
        gr = groups(half)
        gB = np.array([Bk[g].sum(0) for g in gr]); bulk = np.log2(gB / gB.sum(0, keepdims=True) * 1e6 + 1)
        sc = np.zeros((len(gr), NT))
        for i, k in enumerate(pi):
            for a, g in enumerate(gr):
                c, f = Cs[k, g].sum(), Fs[k, g].sum()
                cp, fp = Cp[g].sum() - c, Fp[g].sum() - f
                sc[a, i] = np.log2((c + E) / (f + E)) - np.log2((cp + E) / (fp + E))
        M = np.corrcoef(np.column_stack([bulk, sc]).T)[NT:, :NT]
        obs, p = perm(M); _, p5 = perm(M[1:, 1:])
        pd.DataFrame(M, [f"{t} sc" for t in TFS], [f"{t} ChIP" for t in TFS]).to_csv(OD / f"m_{arm}_{half}.tsv", sep="\t", float_format="%.5f")
        rows.append(dict(mask=MASK, arm=arm, half=half, n_groups=len(gr), diag_minus_off=obs, perm_p=p, noCTCF_perm_p=p5,
                         own_best_rows=int(sum(np.argmax(M[i]) == i for i in range(NT))), **{f"own_{t}": M[i, i] for i, t in enumerate(TFS)}))
        print(rows[-1], flush=True)
        if half == "all":
            pd.DataFrame(M, [f"{t} sc" for t in TFS], [f"{t} ChIP" for t in TFS]).to_csv(SD / f"corr_imputed_{MASK}_{arm}_6x6.tsv", sep="\t", float_format="%.5f")
T = pd.DataFrame(rows); T.to_csv(OD / "summary.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 250); print(T.round(3).to_string()); print("done")
