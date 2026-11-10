#!/usr/bin/env python
"""Held-out confirmation of the summit-group Pearson test (round 3).

Split: every reference's summits by chromosome. Discovery = odd autosomes (chr1, 3, ..., 21);
test = even autosomes + chrX. Statistic, as in r2_05_pearson.py: per reference, the chosen summit set
(top N by bulk signalValue) is split into G groups; per group bulk = log2(CPM + 1) of ChIP reads in
summit +-250 bp; sc = log2((C + .5)/(F + .5)) - same for the leave-one-out 79-TF pool (C = +-100 bp,
F = 1-2 kb flanks); sc thinned to ZNF777 depth (3 seeds; r averaged). 6x6 sc-vs-bulk Pearson block,
diag - offdiag, exact p over 720 sc-label shuffles.
  discover : grid over set (all, unique, preferred fold 1.5, preferred fold 2) x top N (1000, 2500 = half of
               2000/5000) x G (20, 30, 50) on the discovery half; best = smallest p, ties broken by larger
               diag-off; writes OUT/round3_pearson/heldout/locked_spec.json and discovery_grid.tsv.
  test     : applies locked_spec.json once to the test half (plus the published spec preferred fold 2,
               top 2500, G 30 as a secondary result); writes test_result.tsv, test_pertf.tsv, test_per_seed.tsv,
               test_corr_*.tsv and block PNGs.
"""
import argparse
import itertools
import json
import sys

import numpy as np
import pandas as pd

from fig2_common import DM_DEPTH, DM_SEEDS, OUT, ROOT, TFS

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["discover", "test"])
mode = ap.parse_args().mode
HO = OUT / "round3_pearson" / "heldout"
HO.mkdir(parents=True, exist_ok=True)
NT = len(TFS)
R2 = OUT / "round2"
S = pd.read_csv(R2 / "summits.tsv", sep="\t", low_memory=False)
z = np.load(R2 / "summit_counts.npz", allow_pickle=True)
C79, F79, tf79, tot79 = z["C"].astype(np.int64), z["F"].astype(np.int64), list(z["tf79"]), z["tot79"]
pi = [tf79.index(t) for t in TFS]
Cpool, Fpool = C79.sum(0), F79.sum(0)
Bk = S[[f"bulk_{t}" for t in TFS]].values
chrom_num = S.chrom.str.replace("chr", "").replace("X", "23").astype(int).values
HALF = {"discovery": (chrom_num % 2 == 1) & (chrom_num != 23), "test": (chrom_num % 2 == 0) | (chrom_num == 23)}
pdm = np.minimum(1, DM_DEPTH / tot79[pi])[:, None]
THIN = [(np.random.default_rng(s).binomial(C79[pi], pdm), np.random.default_rng(s + 1).binomial(F79[pi], pdm))
        for s in DM_SEEDS]


def summit_set(half, typ, fold, n):
    out = []
    for t in TFS:
        m = (S.ref.values == t) & HALF[half]
        if typ == "unique":
            m &= S["unique"].values.astype(bool)
        elif typ == "preferred":
            m &= S.bulk_spec.values > np.log2(fold)
        ix = np.flatnonzero(m)
        out.append(ix[np.argsort(-S.signal.values[ix], kind="stable")][:n])
    return out


def block(sets, G, Cs, Fs):
    gr = [g for ix in sets for g in np.array_split(ix, G) if len(g)]
    gB = np.array([Bk[g].sum(0) for g in gr])
    bulk = np.log2(gB / gB.sum(0, keepdims=True) * 1e6 + 1)
    sc = np.zeros((len(gr), NT))
    for i in range(NT):
        k = pi[i]
        for a, g in enumerate(gr):
            cp, fp = Cpool[g].sum() - C79[k, g].sum(), Fpool[g].sum() - F79[k, g].sum()
            sc[a, i] = np.log2((Cs[i, g].sum() + .5) / (Fs[i, g].sum() + .5)) - np.log2((cp + .5) / (fp + .5))
    R = np.corrcoef(np.column_stack([bulk, sc]).T)
    return R, R[NT:, :NT]


def perm(B):
    k = len(B)
    off = ~np.eye(k, dtype=bool)
    st = lambda M: M.diagonal().mean() - M[off].mean()
    obs = st(B)
    null = np.array([st(B[list(q)]) for q in itertools.permutations(range(k))])
    return float(obs), float(np.mean(null >= obs - 1e-12))


def run(half, spec):
    sets = summit_set(half, spec["set"], spec.get("fold", 0), spec["top_n"])
    Rs, seeds = [], []
    for si, (Cs, Fs) in enumerate(THIN):
        R, blk = block(sets, spec["groups"], Cs, Fs)
        Rs.append(R)
        o, p = perm(blk)
        seeds.append(dict(seed=si, diag_minus_off=o, perm_p=p))
    R = np.mean(Rs, 0)
    return R, sets, seeds


if mode == "discover":
    grid = []
    for typ, fold in (("all", 0), ("unique", 0), ("preferred", 1.5), ("preferred", 2.0)):
        for n in (1000, 2500):
            for G in (20, 30, 50):
                spec = dict(set=typ, fold=fold, top_n=n, groups=G)
                R, sets, _ = run("discovery", spec)
                o, p = perm(R[NT:, :NT])
                grid.append(dict(**spec, n_min=min(len(s) for s in sets), diag_minus_off=o, perm_p=p))
                print(grid[-1], flush=True)
    g = pd.DataFrame(grid)
    g.to_csv(HO / "discovery_grid.tsv", sep="\t", index=False, float_format="%.4f")
    best = g.sort_values(["perm_p", "diag_minus_off"], ascending=[True, False]).iloc[0]
    locked = dict(set=best.set, fold=float(best.fold), top_n=int(best.top_n), groups=int(best.groups),
                  depth="ZNF777 (420,339 reads), seeds " + ",".join(map(str, DM_SEEDS)),
                  discovery_half="odd autosomes", test_half="even autosomes + chrX",
                  discovery_diag_minus_off=float(best.diag_minus_off), discovery_perm_p=float(best.perm_p))
    (HO / "locked_spec.json").write_text(json.dumps(locked, indent=2) + "\n")
    print("LOCKED", locked)
    sys.exit(0)

# ---- test ----
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402
P = rp.use()
INK = P.neutral.ink
locked = json.loads((HO / "locked_spec.json").read_text())
specs = {"locked": {k: locked[k] for k in ("set", "fold", "top_n", "groups")},
         "published": dict(set="preferred", fold=2.0, top_n=2500, groups=30)}
res, pertf, seeds_all = [], [], []
labs = [f"Bulk {t}" for t in TFS] + [f"sc {t}" for t in TFS]
for name, spec in specs.items():
    R, sets, seeds = run("test", spec)
    blk = R[NT:, :NT]
    o, p = perm(blk)
    keep = list(range(1, NT))
    o5, p5 = perm(blk[np.ix_(keep, keep)])
    own = [int(np.argmax(blk[i]) == i) for i in range(NT)]
    res.append(dict(spec_name=name, **spec, n_min=min(len(s) for s in sets), diag_minus_off=o, perm_p=p,
                    noCTCF_diag_minus_off=o5, noCTCF_perm_p=p5, own_best_row=sum(own),
                    own_best_col=int(sum(np.argmax(blk[:, j]) == j for j in range(NT)))))
    for s in seeds:
        seeds_all.append(dict(spec_name=name, **s))
    off = ~np.eye(NT, dtype=bool)
    for i, t in enumerate(TFS):
        pertf.append(dict(spec_name=name, tf=t, r_own=blk[i, i], row_off_mean=blk[i][off[i]].mean(),
                          own_minus_off=blk[i, i] - blk[i][off[i]].mean(),
                          best_bulk=TFS[int(np.argmax(blk[i]))]))
    pd.DataFrame(R, labs, labs).to_csv(HO / f"test_corr_{name}.tsv", sep="\t", float_format="%.4f")
    fig, ax = plt.subplots(figsize=(2.7, 2.5))
    lim = max(0.1, np.abs(blk).max())
    im = ax.imshow(blk, cmap=P.div_cmap(), vmin=-lim, vmax=lim)
    for a in range(NT):
        for b in range(NT):
            ax.text(b, a, f"{blk[a, b]:+.2f}", ha="center", va="center", fontsize=5,
                    color="white" if abs(blk[a, b]) > 0.6 * lim else INK)
        ax.add_patch(Rectangle((a - .5, a - .5), 1, 1, fill=False, ec=INK, lw=0.8))
    ax.set_xticks(range(NT), [f"Bulk {t}" for t in TFS], rotation=90)
    ax.set_yticks(range(NT), [f"sc {t}" for t in TFS])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Pearson r")
    ax.set_title(f"Held-out test half ({name} spec), ZNF777 depth\nrows {sum(own)}/6, p = {p:.3f} "
                 f"(no CTCF p = {p5:.3f})", fontsize=6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HO / f"test_block_{name}.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
pd.DataFrame(res).to_csv(HO / "test_result.tsv", sep="\t", index=False, float_format="%.4f")
pd.DataFrame(pertf).to_csv(HO / "test_pertf.tsv", sep="\t", index=False, float_format="%.4f")
pd.DataFrame(seeds_all).to_csv(HO / "test_per_seed.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 250)
print(pd.DataFrame(res).round(4).to_string())
print(pd.DataFrame(pertf).round(3).to_string())
print(pd.DataFrame(seeds_all).round(4).to_string())
