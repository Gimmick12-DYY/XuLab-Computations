#!/usr/bin/env python
"""Imputed tracks vs bulk ChIP through the pre-specified raw recipes (no version search).

Imputed = cells positive per 1 kb bin (unified imputation, masked to the ATAC union). Features declared before
looking at results:
  primary    per summit: log2((imp_i[b] + 1) / (mean of the other 78 imputed tracks at b + 1)), b = the summit's bin;
             per group = mean over summits.
  secondary  primary minus the mean of the same log ratio at the two flanking bins (b-1, b+1), only for summits
             whose two flanking bins are both inside the ATAC mask.
Recipes (bulk = log2(CPM + 1) of ChIP reads +-250 bp, as for raw):
  R-A  HEK293T-only 2x2 (GSE103651 CTCF nexus; GSM2466515 ZNF282): summits of r5_hek293t_2x2.py, own - other > 1,
       top 5000, 30 groups per reference; all chromosomes, odd autosomes, even autosomes + chrX (top 5000 each).
  R-B  6x6 (references: CTCF mixed HEK293/HEK293T, MAZ/ZBTB7A/ZIC2/ZNF777 HEK293, ZNF282 HEK293T): round-2
       summits, preferred 2x vs the other 5, top 5000, 30 groups; halves use top 2500 (as in the held-out test).
Reports plain Pearson blocks, own > other per row/col, own minus mean other per TF, exact label-permutation p
for the 6x6 (720; 120 without CTCF). Figures: raw (from the existing raw matrices) next to imputed primary and secondary.
Outputs OUT/imputed_samerecipe/.
"""
import itertools
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib.patches import Rectangle

from fig2_common import D, INP, OUT, ROOT, TFS, bin_index

sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402

P = rp.use()
INK = P.neutral.ink
O6 = OUT / "imputed_samerecipe"
O6.mkdir(parents=True, exist_ok=True)
G = 30
bi, off, N = bin_index()
fr = np.load(INP / "frame.npz", allow_pickle=True)
tf79 = list(fr["tf79"])
atac = fr["atac_union"]
imp = sp.load_npz(INP / "imp79.npz").tocsr()
cstart = np.zeros(N, np.int64); cend = np.zeros(N, np.int64)
for c, (o, n) in off.items():
    cstart[o:o + n] = o
    cend[o:o + n] = o + n
DEN = {k: imp[tf79.index(k)].toarray().ravel().astype(float) for k in TFS}
POOL = np.asarray(imp.sum(0)).ravel().astype(float)
NOTH = len(tf79) - 1


def lr(t, j):
    return np.log2((DEN[t][j] + 1) / ((POOL[j] - DEN[t][j]) / NOTH + 1))


def per_summit(t, b):
    prim = lr(t, b)
    l, r = b - 1, b + 1
    ok = (l >= cstart[b]) & (r < cend[b])
    ok[ok] &= atac[l[ok]] & atac[r[ok]]
    sec = np.full(len(b), np.nan)
    sec[ok] = prim[ok] - (lr(t, l[ok]) + lr(t, r[ok])) / 2
    return prim, sec


def perm(B):
    k = len(B)
    off_ = ~np.eye(k, dtype=bool)
    st = lambda M: M.diagonal().mean() - M[off_].mean()
    obs = st(B)
    null = np.array([st(B[list(q)]) for q in itertools.permutations(range(k))])
    return float(obs), float(np.mean(null >= obs - 1e-12))


def groups(S, tfs, mask, half, n):
    cn = S.chrom.str.replace("chr", "").replace("X", "23").astype(int).values
    h = {"all": np.ones(len(S), bool), "odd": (cn % 2 == 1) & (cn != 23), "even_X": (cn % 2 == 0) | (cn == 23)}[half]
    gr = []
    for t in tfs:
        ix = np.flatnonzero((S.ref.values == t) & mask & h)
        ix = ix[np.argsort(-S.signal.values[ix], kind="stable")][:n]
        gr += [g for g in np.array_split(ix, G) if len(g)]
    return gr


def run(S, Bk, tfs, mask, half, n):
    b = np.array([off[c][0] for c in S.chrom.values]) + S.pos.values // 1000
    gr = groups(S, tfs, mask, half, n)
    gB = np.array([Bk[g].sum(0) for g in gr])
    bulk = np.log2(gB / gB.sum(0, keepdims=True) * 1e6 + 1)
    F1 = np.zeros((len(gr), len(tfs))); F2 = F1.copy()
    for i, t in enumerate(tfs):
        p1, p2 = per_summit(t, b)
        for a, g in enumerate(gr):
            F1[a, i] = p1[g].mean()
            F2[a, i] = np.nanmean(p2[g]) if np.isfinite(p2[g]).any() else np.nan
    k = len(tfs)
    Rp = np.corrcoef(np.column_stack([bulk, F1]).T)[k:, :k]
    keep = np.isfinite(F2).all(1)
    Rs = np.corrcoef(np.column_stack([bulk[keep], F2[keep]]).T)[k:, :k]
    frac_flank = float(np.mean([np.isfinite(per_summit(tfs[0], b[g])[1]).mean() for g in gr]))
    return Rp, Rs, len(gr), int(keep.sum()), float(np.mean([atac[b[g]].mean() for g in gr])), frac_flank


def stats(B):
    k = len(B)
    off_ = ~np.eye(k, dtype=bool)
    d = dict(own_max_rows=int(sum(np.argmax(B[i]) == i for i in range(k))),
             own_max_cols=int(sum(np.argmax(B[:, j]) == j for j in range(k))),
             diag_minus_off=float(B.diagonal().mean() - B[off_].mean()))
    return d


REFLAB6 = ["CTCF (mixed)", "MAZ (HEK293)", "ZBTB7A (HEK293)", "ZIC2 (HEK293)", "ZNF282 (HEK293T)", "ZNF777 (HEK293)"]
rows, pertf = [], []

# R-B 6x6
S6 = pd.read_csv(OUT / "round2/summits.tsv", sep="\t", low_memory=False)
Bk6 = S6[[f"bulk_{t}" for t in TFS]].values
mask6 = S6.bulk_spec.values > 1
blocks6 = {}
for half, n in (("all", 5000), ("odd", 2500), ("even_X", 2500)):
    Rp, Rs, ng, ngs, ina, ff = run(S6, Bk6, TFS, mask6, half, n)
    blocks6[half] = (Rp, Rs)
    for feat, B in (("primary", Rp), ("secondary", Rs)):
        o, p = perm(B)
        keep = list(range(1, 6))
        o5, p5 = perm(B[np.ix_(keep, keep)])
        rows.append(dict(recipe="R-B_6x6", feature=feat, half=half, n_groups=ng, n_groups_used=ng if feat == "primary" else ngs,
                         frac_summit_bin_in_ATAC=ina, frac_summits_flanks_in_ATAC=ff, perm_p=p, noCTCF_diag_minus_off=o5,
                         noCTCF_perm_p=p5, **stats(B)))
        pd.DataFrame(B, [f"imputed {t}" for t in TFS], REFLAB6).to_csv(O6 / f"Rb_6x6_{feat}_{half}.tsv", sep="\t", float_format="%.4f")
        off_ = ~np.eye(6, dtype=bool)
        for i, t in enumerate(TFS):
            pertf.append(dict(recipe="R-B_6x6", feature=feat, half=half, tf=t, r_own=B[i, i], mean_other=B[i][off_[i]].mean(),
                              own_minus_mean_other=B[i, i] - B[i][off_[i]].mean(), best_bulk=TFS[int(np.argmax(B[i]))]))

# R-A 2x2
S2 = pd.read_csv(OUT / "hek293t_2x2/summits.tsv", sep="\t", low_memory=False)
T2 = ["CTCF", "ZNF282"]
BAM2 = [ROOT / "chip_gse103651/work/bam/gse103651.pooled.bam", D / "ZNF282_HEK293.bam"]
bb = O6 / "w250.bed"
o_ = np.lexsort((S2.pos.values, S2.chrom.values))
pd.DataFrame({0: S2.chrom.values[o_], 1: S2.pos.values[o_] - 250, 2: S2.pos.values[o_] + 250, 3: o_}).to_csv(
    bb, sep="\t", header=False, index=False)
out = subprocess.run(["bedtools", "multicov", "-bams", *map(str, BAM2), "-bed", str(bb)], check=True,
                     capture_output=True, text=True).stdout
Bk2 = np.zeros((len(S2), 2))
for l in out.splitlines():
    p_ = l.split("\t")
    Bk2[int(p_[3])] = [float(x) for x in p_[4:]]
mask2 = S2.preferred.values.astype(bool)
blocks2 = {}
for half in ("all", "odd", "even_X"):
    Rp, Rs, ng, ngs, ina, ff = run(S2, Bk2, T2, mask2, half, 5000)
    blocks2[half] = (Rp, Rs)
    for feat, B in (("primary", Rp), ("secondary", Rs)):
        rows.append(dict(recipe="R-A_2x2", feature=feat, half=half, n_groups=ng, n_groups_used=ng if feat == "primary" else ngs,
                         frac_summit_bin_in_ATAC=ina, frac_summits_flanks_in_ATAC=ff, **stats(B)))
        pd.DataFrame(B, ["imputed CTCF", "imputed ZNF282"], ["HEK293T ChIP CTCF", "HEK293T ChIP ZNF282"]).to_csv(
            O6 / f"Ra_2x2_{feat}_{half}.tsv", sep="\t", float_format="%.4f")
        for i, t in enumerate(T2):
            pertf.append(dict(recipe="R-A_2x2", feature=feat, half=half, tf=t, r_own=B[i, i], mean_other=B[i, 1 - i],
                              own_minus_mean_other=B[i, i] - B[i, 1 - i], best_bulk=T2[int(np.argmax(B[i]))]))
pd.DataFrame(rows).to_csv(O6 / "summary.tsv", sep="\t", index=False, float_format="%.4f")
pd.DataFrame(pertf).to_csv(O6 / "pertf.tsv", sep="\t", index=False, float_format="%.4f")


def heat(ax, M, xl, yl, title, lim):
    im = ax.imshow(M, cmap=P.div_cmap(), vmin=-lim, vmax=lim)
    for a in range(M.shape[0]):
        for b in range(M.shape[1]):
            ax.text(b, a, f"{M[a, b]:+.2f}", ha="center", va="center", fontsize=5,
                    color="white" if abs(M[a, b]) > 0.6 * lim else INK)
        ax.add_patch(Rectangle((a - .5, a - .5), 1, 1, fill=False, ec=INK, lw=0.6))
    ax.set_xticks(range(M.shape[1]), xl, rotation=90)
    ax.set_yticks(range(M.shape[0]), yl)
    ax.tick_params(length=0)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.set_title(title, fontsize=6)
    return im


raw6 = pd.read_csv(R2P := OUT / "round2/pearson/r2p_corr_preferred_top5000_znf777.tsv", sep="\t", index_col=0).values[6:, :6]
raw2 = pd.read_csv(OUT / "hek293t_2x2/corr4x4_all.tsv", sep="\t", index_col=0).values[2:, :2]
for nm, rawB, (Bp, Bs), xl, tfs, fs in (("Rb_6x6", raw6, blocks6["all"], REFLAB6, TFS, (7.2, 2.9)),
                                        ("Ra_2x2", raw2, blocks2["all"], ["HEK293T ChIP CTCF", "HEK293T ChIP ZNF282"], T2, (6.0, 2.5))):
    lim = max(0.1, np.abs(rawB).max(), np.abs(Bp).max(), np.abs(Bs).max())
    fig, axs = plt.subplots(1, 3, figsize=fs, layout="constrained")
    heat(axs[0], rawB, xl, [f"raw {t}" for t in tfs], "Raw (ZNF777 depth)\ncentre/flank vs pool", lim)
    heat(axs[1], Bp, xl, [f"imputed {t}" for t in tfs], "Imputed, primary\nsummit bin vs other 78 TFs", lim)
    im = heat(axs[2], Bs, xl, [f"imputed {t}" for t in tfs], "Imputed, secondary\nprimary minus flanking bins", lim)
    fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02, label="Pearson r")
    for ext in ("png", "pdf"):
        fig.savefig(O6 / f"{nm}_raw_vs_imputed.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).round(3).to_string())
print(pd.DataFrame(pertf).round(3).to_string())
