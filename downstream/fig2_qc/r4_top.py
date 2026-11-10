#!/usr/bin/env python
"""Round 4 report: top 3 (signal version x transform) from r4_grid.py (distinct signal versions).
For each: 6x6 r block M (full depth, or ZNF777 depth averaged over 3 seeds), transformed matrix T, own-max
rows/cols, label and pseudo nulls (from the grid), the same version at ZNF777 depth (3 seeds) and on the two
chromosome halves (odd autosomes; even autosomes + chrX). Heatmaps: B0 next to the transform.
Outputs OUT/round4/top/: r4_top_summary.tsv, r4_top<k>_{M,T}.tsv, r4_top<k>.png/.pdf."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import r4_grid as g
from fig2_common import DM_SEEDS, ROOT, TFS

sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402

P = rp.use()
INK = P.neutral.ink
NT = len(TFS)
OT = g.O4 / "top"
OT.mkdir(exist_ok=True)
top = pd.read_pickle(g.O4 / "r4_top.pkl").drop_duplicates("aver").head(3)
TNAME = {"B0": "none", "B1": "row z-score", "B2": "column z-score", "B3": "double-centred, z-scored",
         "B4": "Sinkhorn bistochastic of exp(r)", "B5": "within-row rank"}


def Mfor(r, depth, half="all"):
    lab, k = g.make_groups(r.sel_t, r.groups, half)
    g.GP.clear()
    if depth == "full":
        return g.block(lab, k, r.bulk, r.sc, *g.tracks("full"))
    return np.mean([g.block(lab, k, r.bulk, r.sc, *g.tracks("znf777", s)) for s in DM_SEEDS], 0)


def heat(ax, M, title, lim=None):
    lim = lim or max(1e-6, np.abs(M).max())
    im = ax.imshow(M, cmap=P.div_cmap(), vmin=-lim, vmax=lim)
    for a in range(NT):
        for b in range(NT):
            ax.text(b, a, f"{M[a, b]:+.2f}", ha="center", va="center", fontsize=5,
                    color="white" if abs(M[a, b]) > 0.6 * lim else INK)
        ax.add_patch(Rectangle((a - .5, a - .5), 1, 1, fill=False, ec=INK, lw=0.6))
    ax.set_xticks(range(NT), [f"Bulk {t}" for t in TFS], rotation=90)
    ax.set_yticks(range(NT), [f"sc {t}" for t in TFS])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=5.5)
    return im


rows = []
for kk, r in enumerate(top.itertuples(), 1):
    M = Mfor(r, r.depth)
    T = g.transform(M, r.transform)
    or_, oc_, do_ = g.metrics(T)
    other = "znf777" if r.depth == "full" else "full"
    Mo = Mfor(r, other)
    halves = {h: g.metrics(g.transform(Mfor(r, r.depth, h), r.transform))[0] for h in ("odd", "even_X")}
    rows.append(dict(rank=kk, depth=r.depth, summits=r.sel, groups=r.groups, bulk=r.bulk, sc=r.sc,
                     transform=f"{r.transform} ({TNAME[r.transform]})", own_max_rows=or_, own_max_cols=oc_,
                     diag_minus_off=do_, label_null=r.label_null, pseudo_null=r.pseudo_null,
                     B0_own_max_rows=g.metrics(M)[0], **{f"{other}_own_max_rows": g.metrics(g.transform(Mo, r.transform))[0]},
                     odd_own_max_rows=halves["odd"], evenX_own_max_rows=halves["even_X"]))
    lab = pd.Index([f"sc {t}" for t in TFS])
    pd.DataFrame(M, lab, [f"Bulk {t}" for t in TFS]).to_csv(OT / f"r4_top{kk}_M.tsv", sep="\t", float_format="%.4f")
    pd.DataFrame(T, lab, [f"Bulk {t}" for t in TFS]).to_csv(OT / f"r4_top{kk}_T.tsv", sep="\t", float_format="%.4f")
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.8), layout="constrained")
    dep = "full depth" if r.depth == "full" else "ZNF777 depth"
    desc = f"{r.sel} summits, {r.groups} groups/ref, bulk {r.bulk}, sc {r.sc}, {dep}"
    i0 = heat(axs[0], M, f"Pearson r, no transform\nown max: rows {g.metrics(M)[0]}/6, cols {g.metrics(M)[1]}/6")
    fig.colorbar(i0, ax=axs[0], fraction=0.046, pad=0.03, label="Pearson r")
    tl = None if r.transform != "B5" else 6
    Tp = T if r.transform != "B5" else T - 3.5
    i1 = heat(axs[1], Tp, f"Transform: {TNAME[r.transform]}\nown max: rows {or_}/6, cols {oc_}/6; "
              f"null ≥ rows: pseudo {r.pseudo_null:.3f}, labels {r.label_null:.3f}", lim=None if tl is None else 2.5)
    fig.colorbar(i1, ax=axs[1], fraction=0.046, pad=0.03, label=TNAME[r.transform])
    fig.suptitle(desc, fontsize=6)
    for ext in ("png", "pdf"):
        fig.savefig(OT / f"r4_top{kk}.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
    print(f"top{kk}", rows[-1], "\nM\n", pd.DataFrame(M, TFS, TFS).round(3), "\nT\n", pd.DataFrame(T, TFS, TFS).round(2),
          flush=True)
pd.DataFrame(rows).to_csv(OT / "r4_top_summary.tsv", sep="\t", index=False, float_format="%.4f")
