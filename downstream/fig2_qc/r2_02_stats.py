#!/usr/bin/env python
"""Round 2: summit-centred centre/flank specificity tests (R2-1, R2-2, R2-3).

Statistic for sc TF i at reference j's summit set (counts pooled over the summits):
  L[i,j] = log2((C_i + .5) / (F_i + .5)) - log2((C_pool-i + .5) / (F_pool-i + .5))
C = read 5' ends within +-100 bp of the summit, F = within 1-2 kb on both sides, pool-i = all 79 TFs
minus TF i (full depth). The ratio cancels broad local accessibility and total depth.
Summit sets per reference: top N by bulk signalValue (N = 2000, 5000) among all / unique / preferred summits.
R2-1  6x6 L, double-centred D; diag-minus-offdiag of D, own-best rows/cols, exact p over 720 row shuffles;
      full depth and ZNF777 depth (binomial thinning of the 6 TFs' counts, 3 seeds, L averaged);
      bootstrap 95% CI over summits (200 reps); per-TF own-minus-off (row and column);
      leave-one-TF-out p (5x5, 120 shuffles).
R2-2  all 79 TFs' L at each set: rank of the own TF among 79 (full depth; and depth-matched with every TF
      thinned to <= ZNF777 depth), plus z of the own TF vs 20 depth-matched pseudo-TFs (pool thinned).
R2-3  R2-1 within promoter (TSS +-1 kb) and distal summits; CTCF and MAZ columns split by motif presence.
Outputs OUT/round2/: r2_summary.tsv, r2_L_*.tsv, r2_D_*.tsv, r2_pertf.tsv, r2_loo.tsv, r2_panel_rank.tsv,
r2_motif_strata.tsv.
"""
import itertools

import numpy as np
import pandas as pd

from fig2_common import DM_DEPTH, DM_SEEDS, OUT, TFS, write_mat

R2 = OUT / "round2"
S = pd.read_csv(R2 / "summits.tsv", sep="\t")
z = np.load(R2 / "summit_counts.npz", allow_pickle=True)
C79, F79, tf79, tot79 = z["C"].astype(np.int64), z["F"].astype(np.int64), list(z["tf79"]), z["tot79"]
NT = len(TFS)
pi = [tf79.index(t) for t in TFS]
Cpool, Fpool = C79.sum(0), F79.sum(0)
NBOOT, NPSEUDO = 200, 20
rng = np.random.default_rng(2026)


def make_sets():
    out = {}
    for n in (2000, 5000):
        for typ in ("all", "unique", "preferred"):
            d = {}
            for t in TFS:
                m = (S.ref.values == t)
                if typ != "all":
                    m &= S[typ].values.astype(bool)
                idx = np.flatnonzero(m)
                idx = idx[np.argsort(-S.signal.values[idx], kind="stable")][:n]
                d[t] = idx
            out[f"{typ}_top{n}"] = d
    return out


def L_of(Ci, Fi, Cfull, Ffull, idx):
    """Ci, Fi: counts of the scored track(s) (rows); Cfull/Ffull: full-depth counts of the same TF
    (for leave-one-out pool). Returns vector over rows."""
    c, f = Ci[:, idx].sum(1), Fi[:, idx].sum(1)
    cp = Cpool[idx].sum() - Cfull[:, idx].sum(1)
    fp = Fpool[idx].sum() - Ffull[:, idx].sum(1)
    return np.log2((c + .5) / (f + .5)) - np.log2((cp + .5) / (fp + .5))


def Lmat(sets, Cp, Fp, rows=pi):
    Cf, Ff = C79[rows], F79[rows]
    return np.column_stack([L_of(Cp, Fp, Cf, Ff, sets[t]) for t in TFS])


def thin(X, p, seed):
    g = np.random.default_rng(seed)
    return g.binomial(X, p[:, None] if np.ndim(p) else p)


def dcentre(L):
    return L - L.mean(1, keepdims=True) - L.mean(0, keepdims=True) + L.mean()


def stat(D):
    k = len(D)
    off = ~np.eye(k, dtype=bool)
    return D.diagonal().mean() - D[off].mean()


def perm_p(L):
    D = dcentre(L)
    obs = stat(D)
    null = np.array([stat(D[list(p)]) for p in itertools.permutations(range(len(D)))])
    return obs, float(np.mean(null >= obs - 1e-12)), float((obs - null.mean()) / null.std())


def Lversions(sets):
    full = Lmat(sets, C79[pi], F79[pi])
    pdm = np.minimum(1, DM_DEPTH / tot79[pi])
    dms = [Lmat(sets, thin(C79[pi], pdm, s), thin(F79[pi], pdm, s + 1)) for s in DM_SEEDS]
    return {"full": full, "znf777": np.mean(dms, 0)}


def boot_ci(sets, depth):
    vals = []
    pdm = np.minimum(1, DM_DEPTH / tot79[pi])
    for b in range(NBOOT):
        bs = {t: rng.choice(ix, len(ix), replace=True) for t, ix in sets.items()}
        if depth == "full":
            L = Lmat(bs, C79[pi], F79[pi])
        else:
            s = DM_SEEDS[b % 3]
            L = Lmat(bs, thin(C79[pi], pdm, s), thin(F79[pi], pdm, s + 1))
        vals.append(stat(dcentre(L)))
    return np.percentile(vals, [2.5, 97.5])


summ, pertf, loo, ranks, motif_rows = [], [], [], [], []
allsets = make_sets()
strata = {}
for name, sets in allsets.items():
    strata[name] = sets
    if name.endswith("top5000"):
        for lab, col in (("promoter", True), ("distal", False)):
            strata[f"{name}_{lab}"] = {t: ix[S.promoter.values[ix] == col] for t, ix in sets.items()}

for name, sets in strata.items():
    nmin = min(len(ix) for ix in sets.values())
    for depth, L in Lversions(sets).items():
        D = dcentre(L)
        obs, p, zz = perm_p(L)
        ci = boot_ci(sets, depth) if "_" not in name.replace("_top", "") else (np.nan, np.nan)
        tag = f"{name}_{depth}"
        write_mat(L, R2 / f"r2_L_{tag}.tsv")
        write_mat(D, R2 / f"r2_D_{tag}.tsv")
        row = dict(test="R2-1" if name.count("_") == 1 else "R2-3", set=name, depth=depth,
                   n_summits={t: len(ix) for t, ix in sets.items()}.__repr__(), n_min=nmin,
                   diag_minus_off=obs, perm_p=p, perm_z=zz,
                   own_best_row=int(sum(np.argmax(D[i]) == i for i in range(NT))),
                   own_best_col=int(sum(np.argmax(D[:, j]) == j for j in range(NT))),
                   boot_lo=ci[0], boot_hi=ci[1])
        summ.append(row)
        off = ~np.eye(NT, dtype=bool)
        for i, t in enumerate(TFS):
            pertf.append(dict(set=name, depth=depth, tf=t, L_own=L[i, i], D_own=D[i, i],
                              row_own_minus_off=D[i, i] - D[i][off[i]].mean(),
                              col_own_minus_off=D[i, i] - D[:, i][off[:, i]].mean()))
            keep = [k for k in range(NT) if k != i]
            o2, p2, _ = perm_p(L[np.ix_(keep, keep)])
            loo.append(dict(set=name, depth=depth, dropped=t, diag_minus_off=o2, perm_p_5x5=p2))
        print(f"{tag:35s} n_min={nmin:5d} d-o={obs:+.3f} p={p:.4f} rows={row['own_best_row']} "
              f"cols={row['own_best_col']} CI=[{ci[0]:+.3f},{ci[1]:+.3f}]", flush=True)

# ---- R2-2 panel rank ----
pdm79 = np.minimum(1, DM_DEPTH / tot79)
for name, sets in allsets.items():
    Lfull = Lmat(sets, C79, F79, rows=list(range(len(tf79))))
    Ldm = np.mean([Lmat(sets, thin(C79, pdm79, s), thin(F79, pdm79, s + 1), rows=list(range(len(tf79))))
                   for s in DM_SEEDS], 0)
    pz = []
    for k in range(NPSEUDO):
        g = np.random.default_rng(500 + k)
        cps, fps = g.binomial(Cpool, DM_DEPTH / tot79.sum()), g.binomial(Fpool, DM_DEPTH / tot79.sum())
        pz.append([np.log2((cps[ix].sum() + .5) / (fps[ix].sum() + .5)) -
                   np.log2((Cpool[ix].sum() + .5) / (Fpool[ix].sum() + .5)) for ix in (sets[t] for t in TFS)])
    pz = np.array(pz)
    for j, t in enumerate(TFS):
        k = tf79.index(t)
        ranks.append(dict(set=name, reference=t, L_own_full=Lfull[k, j],
                          rank_full_of79=int((Lfull[:, j] > Lfull[k, j]).sum() + 1),
                          L_own_dm=Ldm[k, j], rank_dm_of79=int((Ldm[:, j] > Ldm[k, j]).sum() + 1),
                          z_vs_pseudo=float((Ldm[k, j] - pz[:, j].mean()) / pz[:, j].std(ddof=1)),
                          top3_full=",".join(np.array(tf79)[np.argsort(-Lfull[:, j])[:3]])))
    # motif strata (CTCF, MAZ columns)
    if name.startswith("all") or name.startswith("unique"):
        for t in ("CTCF", "MAZ"):
            ix = sets[t]
            for lab, mm in (("motif+", S.motif.values[ix] == True), ("motif-", S.motif.values[ix] == False)):  # noqa: E712
                sub = ix[mm]
                lf = L_of(C79, F79, C79, F79, sub)
                k = tf79.index(t)
                panel = [tf79.index(u) for u in TFS]
                motif_rows.append(dict(set=name, reference=t, stratum=lab, n=len(sub), L_own=lf[k],
                                       rank_of79=int((lf > lf[k]).sum() + 1),
                                       rank_of6=int((lf[panel] > lf[k]).sum() + 1),
                                       **{f"L_{u}": lf[tf79.index(u)] for u in TFS}))

pd.DataFrame(summ).to_csv(R2 / "r2_summary.tsv", sep="\t", index=False, float_format="%.4g")
pd.DataFrame(pertf).to_csv(R2 / "r2_pertf.tsv", sep="\t", index=False, float_format="%.4g")
pd.DataFrame(loo).to_csv(R2 / "r2_loo.tsv", sep="\t", index=False, float_format="%.4g")
pd.DataFrame(ranks).to_csv(R2 / "r2_panel_rank.tsv", sep="\t", index=False, float_format="%.4g")
pd.DataFrame(motif_rows).to_csv(R2 / "r2_motif_strata.tsv", sep="\t", index=False, float_format="%.4g")
pd.set_option("display.width", 250)
print(pd.DataFrame(ranks).to_string())
print(pd.DataFrame(motif_rows).to_string())
