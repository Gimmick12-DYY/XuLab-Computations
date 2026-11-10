#!/usr/bin/env python
"""Round 4: signal (A) x standardisation (B) grid for the 6x6 sc-vs-bulk summit-group Pearson, with nulls.

A, signal versions (all six TFs always kept):
  depth     full (primary) or ZNF777 depth (one seed in the grid; 3 seeds for the reported top versions)
  groups    G = 5, 10, 20, 30 per reference (summits split in rank order)
  summits   strength-ranked preferred summits (bulk_spec > log2 fold, fold 1.5 or 2), top 5k/10k/20k;
            or specificity-ranked (bulk_spec = own bulk log CPM minus mean of the other 5), top 5k/10k/20k
  bulk      lcpm (log2 CPM+1 of ChIP reads +-250 bp) or spec (lcpm minus mean of the other 5 bulk)
  sc        poolratio: log2((C+.5)/(F+.5)) minus the same for the leave-one-out 79-TF pool (C +-100 bp, F 1-2 kb)
            poolenr:   log2((C + a)/(E + a)), E = leave-one-out pool centre reads scaled by depth, a = 2
                       (79-TF pool as an untargeted IgG-like control)
B, transforms of the 6x6 r block M (rows sc TF, cols bulk reference):
  B0 none; B1 row z-score; B2 column z-score; B3 double-centred, then z-scored; B4 Sinkhorn bistochastic
  scaling of exp(M); B5 within-row rank.
Per (A, B): own-max rows (own cell is the row maximum), own-max cols, diag - offdiag of the transformed matrix,
label-shuffle null (720 row shuffles of M before the transform: fraction with >= own-max rows) and, for the top
40 A versions, a pseudo-TF null (200 sets of 6 pseudo-TFs = the 79-TF pool thinned to each real TF's depth,
same pipeline; fraction with >= own-max rows). Ranked by own-max rows, then pseudo null exceedance, then
label null, then diag - off. Top 3: 3-seed ZNF777-depth check, chromosome halves (odd / even+X), heatmaps
(transformed next to B0). Outputs OUT/round4/.
"""
import itertools
import sys

import numpy as np
import pandas as pd

from fig2_common import DM_DEPTH, DM_SEEDS, OUT, ROOT, TFS

O4 = OUT / "round4"
O4.mkdir(parents=True, exist_ok=True)
NT = len(TFS)
R2 = OUT / "round2"
S = pd.read_csv(R2 / "summits.tsv", sep="\t", low_memory=False)
z = np.load(R2 / "summit_counts.npz", allow_pickle=True)
C79, F79, tf79, tot79 = z["C"].astype(np.int64), z["F"].astype(np.int64), list(z["tf79"]), z["tot79"].astype(float)
pi = [tf79.index(t) for t in TFS]
Cpool, Fpool, Rpool = C79.sum(0), F79.sum(0), tot79.sum()
Bk = S[[f"bulk_{t}" for t in TFS]].values
cn = S.chrom.str.replace("chr", "").replace("X", "23").astype(int).values
HALVES = {"all": np.ones(len(S), bool), "odd": (cn % 2 == 1) & (cn != 23), "even_X": (cn % 2 == 0) | (cn == 23)}
rng = np.random.default_rng(404)
NPSEUDO = 200


def tracks(depth, seed=DM_SEEDS[0]):
    """Returns (C, F, R): 6 x S thinned counts and per-track total reads."""
    if depth == "full":
        return C79[pi], F79[pi], tot79[pi]
    p = np.minimum(1, DM_DEPTH / tot79[pi])[:, None]
    g = np.random.default_rng(seed)
    return g.binomial(C79[pi], p), g.binomial(F79[pi], p), tot79[pi] * p[:, 0]


def make_groups(sel, G, half):
    lab = np.full(len(S), -1)
    k = 0
    for t in TFS:
        m = (S.ref.values == t) & HALVES[half]
        if sel[0] == "str":
            m &= S.bulk_spec.values > np.log2(sel[1])
            key = -S.signal.values
        else:
            key = -S.bulk_spec.values
        ix = np.flatnonzero(m)
        ix = ix[np.argsort(key[ix], kind="stable")][:sel[2]]
        for part in np.array_split(ix, G):
            if len(part):
                lab[part] = k
                k += 1
    return lab, k


def gsum(lab, k, X):
    m = lab >= 0
    if X.ndim == 1:
        return np.bincount(lab[m], X[m], k)
    return np.vstack([np.bincount(lab[m], x[m], k) for x in X])


def bulk_feat(lab, k, how):
    gB = gsum(lab, k, Bk.T).T
    L = np.log2(gB / gB.sum(0, keepdims=True) * 1e6 + 1)
    if how == "spec":
        L = np.column_stack([L[:, i] - np.delete(L, i, 1).mean(1) for i in range(NT)])
    return L


GP = {}


def pool_groups(lab, k):
    key = (id(lab), k)
    if key not in GP:
        GP[key] = (gsum(lab, k, Cpool), gsum(lab, k, Fpool), gsum(lab, k, C79[pi]), gsum(lab, k, F79[pi]))
    return GP[key]


def sc_feat(lab, k, how, C, F, R):
    gc, gf = gsum(lab, k, C).T, gsum(lab, k, F).T
    pc, pf, fc, ff = pool_groups(lab, k)
    oc, of = pc[:, None] - fc.T, pf[:, None] - ff.T                 # leave-one-out pool (full depth)
    if how == "poolratio":
        return np.log2((gc + .5) / (gf + .5)) - np.log2((oc + .5) / (of + .5))
    E = oc * (R / (Rpool - tot79[pi]))[None, :]
    return np.log2((gc + 2) / (E + 2))


def rowz(M):
    return (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-12)


def transform(M, b):
    if b == "B0":
        return M
    if b == "B1":
        return rowz(M)
    if b == "B2":
        return rowz(M.T).T
    if b == "B3":
        D = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()
        return D / (D.std() + 1e-12)
    if b == "B4":
        X = np.exp(M)
        for _ in range(200):
            X = X / X.sum(1, keepdims=True)
            X = X / X.sum(0, keepdims=True)
        return X
    if b == "B5":
        return np.argsort(np.argsort(M, 1), 1).astype(float) + 1
    raise ValueError(b)


BS = ["B0", "B1", "B2", "B3", "B4", "B5"]
PERMS = list(itertools.permutations(range(NT)))


def metrics(T):
    off = ~np.eye(NT, dtype=bool)
    return (int(sum(np.argmax(T[i]) == i for i in range(NT))), int(sum(np.argmax(T[:, j]) == j for j in range(NT))),
            float(T.diagonal().mean() - T[off].mean()))


def block(lab, k, bh, sh, C, F, R):
    b = bulk_feat(lab, k, bh)
    s = sc_feat(lab, k, sh, C, F, R)
    return np.corrcoef(np.column_stack([b, s]).T)[NT:, :NT]


def label_null(M, b, obs_rows):
    return float(np.mean([metrics(transform(M[list(p)], b))[0] >= obs_rows for p in PERMS]))


SELS = [("str", 1.5, n) for n in (5000, 10000, 20000)] + [("str", 2.0, n) for n in (5000, 10000, 20000)] + \
       [("spec", 0, n) for n in (5000, 10000, 20000)]
if __name__ == "__main__" and sys.argv[1:] == ["grid"]:
    rows = []
    for depth in ("full", "znf777"):
        C, F, R = tracks(depth)
        for sel, G in itertools.product(SELS, (5, 10, 20, 30)):
            lab, k = make_groups(sel, G, "all")
            GP.clear()
            for bh, sh in itertools.product(("lcpm", "spec"), ("poolratio", "poolenr")):
                M = block(lab, k, bh, sh, C, F, R)
                for b in BS:
                    r_, c_, d_ = metrics(transform(M, b))
                    rows.append(dict(depth=depth, sel=f"{sel[0]}{'' if sel[0] == 'spec' else sel[1]}_{sel[2]}",
                                     sel_t=sel, groups=G, bulk=bh, sc=sh, transform=b, own_max_rows=r_,
                                     own_max_cols=c_, diag_minus_off=d_, label_null=label_null(M, b, r_),
                                     M=M))
        print("grid done", depth, flush=True)
    df = pd.DataFrame(rows)
    # pseudo-TF null for the top 40 A versions (by best own-max rows over transforms)
    df["aver"] = df.depth + "|" + df.sel + "|" + df.groups.astype(str) + "|" + df.bulk + "|" + df.sc
    best_a = df.groupby("aver").apply(lambda g: (g.own_max_rows.max(), -g.label_null.min())).sort_values(
        ascending=False).index[:40]
    df["pseudo_null"] = np.nan
    for av in best_a:
        g0 = df[df.aver == av].iloc[0]
        lab, k = make_groups(g0.sel_t, g0.groups, "all")
        GP.clear()
        depth_reads = tot79[pi] if g0.depth == "full" else np.minimum(tot79[pi], DM_DEPTH)
        cnt = {b: [] for b in BS}
        for rep in range(NPSEUDO):
            p = (depth_reads / Rpool)[:, None]
            Cp, Fp = rng.binomial(Cpool[None, :].repeat(NT, 0), p), rng.binomial(Fpool[None, :].repeat(NT, 0), p)
            b_ = bulk_feat(lab, k, g0.bulk)
            gc, gf = gsum(lab, k, Cp).T, gsum(lab, k, Fp).T
            pc, pf = gsum(lab, k, Cpool), gsum(lab, k, Fpool)
            oc, of = pc[:, None] - gc, pf[:, None] - gf
            if g0.sc == "poolratio":
                s_ = np.log2((gc + .5) / (gf + .5)) - np.log2((oc + .5) / (of + .5))
            else:
                s_ = np.log2((gc + 2) / (oc * (depth_reads / (Rpool - depth_reads))[None, :] + 2))
            Mp = np.corrcoef(np.column_stack([b_, s_]).T)[NT:, :NT]
            for b in BS:
                cnt[b].append(metrics(transform(Mp, b))[0])
        for b in BS:
            i = df.index[(df.aver == av) & (df["transform"] == b)][0]
            df.at[i, "pseudo_null"] = float(np.mean(np.array(cnt[b]) >= df.at[i, "own_max_rows"]))
        print("pseudo", av, flush=True)
    df["pn"] = df.pseudo_null.fillna(1.0)
    df = df.sort_values(["own_max_rows", "pn", "label_null", "diag_minus_off"], ascending=[False, True, True, False])
    df.drop(columns=["M", "sel_t", "pn"]).to_csv(O4 / "r4_grid.tsv", sep="\t", index=False, float_format="%.4f")
    df.head(200).to_pickle(O4 / "r4_top.pkl")
    pd.set_option("display.width", 250)
    print(df.drop(columns=["M", "sel_t", "aver", "pn"]).head(40).round(4).to_string())
