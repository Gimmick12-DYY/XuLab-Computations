#!/usr/bin/env python
"""Round 4 follow-up.
1. Selection-corrected null: rerun the whole 1,728-combination search (r4_grid.py: depth x summits x groups x
   bulk x sc x transform) and keep the best own-max-rows, under (a) 100 random sc label permutations
   (exact: rows of each cached 6x6 r block permuted) and (b) 100 sets of 6 depth-matched pseudo-TFs
   (79-TF pool thinned to each real TF's depth at full and ZNF777 depth; same pipeline).
   Reports the fraction of searches whose best reaches >= 5/6 and >= 4/6 rows.
2. Stability of the top-1 spec (preferred fold 2, top 10k, bulk spec, sc pool-ratio, ZNF777 depth, column
   z-score): own-max rows per seed and seeds averaged, all / odd autosomes / even autosomes + chrX,
   for 5 and 10 groups; plain r (B0) and column z (B2).
3. Margins of top 1: own cell minus the best other cell in each row, plain r and column z.
Outputs OUT/round4/followup/."""
import itertools

import numpy as np
import pandas as pd

import r4_grid as g
from fig2_common import DM_DEPTH, DM_SEEDS, TFS

OF = g.O4 / "followup"
OF.mkdir(exist_ok=True)
NT = len(TFS)
NREP = 100
BS = g.BS
rng = np.random.default_rng(777)

# cache group labels and real M for all 288 signal versions
versions = []
for depth in ("full", "znf777"):
    C, F, R = g.tracks(depth)
    for sel, G in itertools.product(g.SELS, (5, 10, 20, 30)):
        lab, k = g.make_groups(sel, G, "all")
        g.GP.clear()
        pc, pf = g.gsum(lab, k, g.Cpool), g.gsum(lab, k, g.Fpool)
        for bh, sh in itertools.product(("lcpm", "spec"), ("poolratio", "poolenr")):
            versions.append(dict(depth=depth, lab=lab, k=k, bh=bh, sh=sh, pc=pc, pf=pf,
                                 b=g.bulk_feat(lab, k, bh), M=g.block(lab, k, bh, sh, C, F, R)))
print("versions", len(versions), flush=True)


def best_rows(Ms):
    return max(g.metrics(g.transform(M, b))[0] for M in Ms for b in BS)


real_best = best_rows([v["M"] for v in versions])
lab_best = []
for r in range(NREP):
    p = rng.permutation(NT)
    lab_best.append(best_rows([v["M"][p] for v in versions]))
print("label search done", flush=True)

pseudo_best = []
for r in range(NREP):
    Ms = []
    pseudo = {}
    for depth in ("full", "znf777"):
        dr = g.tot79[g.pi] if depth == "full" else np.minimum(g.tot79[g.pi], DM_DEPTH)
        p = (dr / g.Rpool)[:, None]
        pseudo[depth] = (rng.binomial(g.Cpool[None, :].repeat(NT, 0), p),
                         rng.binomial(g.Fpool[None, :].repeat(NT, 0), p), dr)
    for v in versions:
        Cp, Fp, dr = pseudo[v["depth"]]
        gc, gf = g.gsum(v["lab"], v["k"], Cp).T, g.gsum(v["lab"], v["k"], Fp).T
        oc, of = v["pc"][:, None] - gc, v["pf"][:, None] - gf
        if v["sh"] == "poolratio":
            s = np.log2((gc + .5) / (gf + .5)) - np.log2((oc + .5) / (of + .5))
        else:
            s = np.log2((gc + 2) / (oc * (dr / (g.Rpool - dr))[None, :] + 2))
        Ms.append(np.corrcoef(np.column_stack([v["b"], s]).T)[NT:, :NT])
    pseudo_best.append(best_rows(Ms))
    if r % 10 == 0:
        print("pseudo search", r, flush=True)
lb, pb = np.array(lab_best), np.array(pseudo_best)
sel_null = pd.DataFrame([dict(null="label permutation", n=NREP, real_best=real_best,
                              frac_ge5=float((lb >= 5).mean()), frac_ge4=float((lb >= 4).mean()),
                              frac_ge_real=float((lb >= real_best).mean()), dist=str(np.bincount(lb, minlength=7).tolist())),
                         dict(null="pseudo-TF sets", n=NREP, real_best=real_best,
                              frac_ge5=float((pb >= 5).mean()), frac_ge4=float((pb >= 4).mean()),
                              frac_ge_real=float((pb >= real_best).mean()), dist=str(np.bincount(pb, minlength=7).tolist()))])
sel_null.to_csv(OF / "selection_null.tsv", sep="\t", index=False, float_format="%.3f")
print(sel_null.to_string(), flush=True)

# 2. stability
sel = ("str", 2.0, 10000)
stab = []
for G in (5, 10):
    for half in ("all", "odd", "even_X"):
        lab, k = g.make_groups(sel, G, half)
        g.GP.clear()
        Ms = [g.block(lab, k, "spec", "poolratio", *g.tracks("znf777", s)) for s in DM_SEEDS]
        for name, M in [(f"seed{s}", m) for s, m in zip(DM_SEEDS, Ms)] + [("seeds_avg", np.mean(Ms, 0))]:
            stab.append(dict(groups=G, half=half, seed=name, rows_B0=g.metrics(M)[0],
                             rows_B2=g.metrics(g.transform(M, "B2"))[0],
                             own_max_B2=",".join(t for i, t in enumerate(TFS)
                                                 if np.argmax(g.transform(M, "B2")[i]) == i)))
st = pd.DataFrame(stab)
st.to_csv(OF / "stability.tsv", sep="\t", index=False)
print(st.to_string(), flush=True)

# 3. margins for top 1 (5 groups, seeds averaged, all summits)
lab, k = g.make_groups(sel, 5, "all")
g.GP.clear()
M = np.mean([g.block(lab, k, "spec", "poolratio", *g.tracks("znf777", s)) for s in DM_SEEDS], 0)
T = g.transform(M, "B2")
mg = []
for i, t in enumerate(TFS):
    for nm, X in (("plain_r", M), ("column_z", T)):
        oth = np.delete(X[i], i)
        mg.append(dict(tf=t, scale=nm, own=X[i, i], best_other=oth.max(), best_other_ref=np.delete(np.array(TFS), i)[oth.argmax()],
                       margin=X[i, i] - oth.max()))
mg = pd.DataFrame(mg)
mg.to_csv(OF / "top1_margins.tsv", sep="\t", index=False, float_format="%.3f")
print(mg.round(3).to_string())
