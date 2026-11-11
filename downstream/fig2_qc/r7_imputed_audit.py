#!/usr/bin/env python
"""Audit of the imputed-vs-bulk analysis (r6_imputed_samerecipe.py).

Checks
 1 Track identity: on ATAC-union bins, Pearson of each panel imputed track (log1p) with all 79 raw tracks (log1p);
   the own raw track should rank 1st among the 79 (or near it).
 2 Regression: recompute the raw 6x6 with this script's code and compare with the published raw matrix.
 3 Scale artefact of the r6 primary feature: log2((imp_i+1)/(mean other 78 + 1)) is not normalised for each TF's cell
   number. Per TF, correlate that feature with log2 of the group's mean pooled imputed signal (accessibility).
Corrected features (group level, scale-free; no version search):
 B  raw-recipe analogue: log2((C_i+.5)/(F_i+.5)) - log2((C_pool-i+.5)/(F_pool-i+.5)); C = summit's 1 kb bin, F = bins b-2, b+2
    (cells positive, summed over the group's summits)
 A  bin enrichment over pool, normalised by cell number: log2(((C_i+.5)/n_cells_i) / ((C_pool-i+.5)/n_cells_others))
Recipes as r6: 6x6 (preferred 2x, top 5000, 30 groups; halves top 2500) and HEK293T 2x2 (own - other > 1, top 5000, 30 groups).
Outputs OUT/imputed_audit/.
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

from fig2_common import COUNTS, D, DM_DEPTH, DM_SEEDS, INP, OUT, ROOT, TFS, bin_index

sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402

P = rp.use()
INK = P.neutral.ink
OA = OUT / "imputed_audit"
OA.mkdir(parents=True, exist_ok=True)
G = 30
bi, off, N = bin_index()
fr = np.load(INP / "frame.npz", allow_pickle=True)
tf79 = list(fr["tf79"])
atac, uni = fr["atac_union"], fr["universe"]
ncell = fr["ncells_imp79"].astype(float)
imp = sp.load_npz(INP / "imp79.npz").tocsr()
raw = sp.load_npz(COUNTS / "raw.npz").tocsr()
cstart = np.zeros(N, np.int64); cend = np.zeros(N, np.int64)
for c, (o, n) in off.items():
    cstart[o:o + n] = o
    cend[o:o + n] = o + n

# ---- 1 identity ----
ab = np.flatnonzero(atac & uni)
X_imp = np.log1p(imp[:, ab].toarray())
X_raw = np.log1p(raw[:, ab].toarray())
Zi = (X_imp - X_imp.mean(1, keepdims=True)) / (X_imp.std(1, keepdims=True) + 1e-12)
Zr = (X_raw - X_raw.mean(1, keepdims=True)) / (X_raw.std(1, keepdims=True) + 1e-12)
C_ir = Zi @ Zr.T / Zi.shape[1]
idr = []
for t in tf79:
    k = tf79.index(t)
    order = np.argsort(-C_ir[k])
    idr.append(dict(tf=t, in_panel=t in TFS, own_r=C_ir[k, k], rank_own_among_79_raw=int(np.where(order == k)[0][0] + 1),
                    best_raw=tf79[order[0]], best_r=C_ir[k, order[0]]))
idr = pd.DataFrame(idr)
idr.to_csv(OA / "check1_track_identity.tsv", sep="\t", index=False, float_format="%.4f")
print(idr[idr.in_panel].round(3).to_string())
print("all 79: own rank 1:", int((idr.rank_own_among_79_raw == 1).sum()), "median own rank", idr.rank_own_among_79_raw.median(), flush=True)

DEN = {k: imp[tf79.index(k)].toarray().ravel().astype(float) for k in TFS}
POOL = np.asarray(imp.sum(0)).ravel().astype(float)
NC = {k: ncell[tf79.index(k)] for k in TFS}
NTOT = ncell.sum()


def cf(track, b):
    fl = np.zeros(len(b))
    for d in (-2, 2):
        j = b + d
        ok = (j >= cstart[b]) & (j < cend[b])
        fl[ok] += track[j[ok]]
    return track[b], fl


def bins(S):
    return np.array([off[c][0] for c in S.chrom.values]) + S.pos.values // 1000


def groups(S, tfs, mask, half, n):
    cn = S.chrom.str.replace("chr", "").replace("X", "23").astype(int).values
    h = {"all": np.ones(len(S), bool), "odd": (cn % 2 == 1) & (cn != 23), "even_X": (cn % 2 == 0) | (cn == 23)}[half]
    gr = []
    for t in tfs:
        ix = np.flatnonzero((S.ref.values == t) & mask & h)
        ix = ix[np.argsort(-S.signal.values[ix], kind="stable")][:n]
        gr += [g for g in np.array_split(ix, G) if len(g)]
    return gr


def feats(S, tfs, gr):
    b = bins(S)
    Fb, Fa, Old = (np.zeros((len(gr), len(tfs))) for _ in range(3))
    access = np.array([np.log2(POOL[b[g]].mean() + 1) for g in gr])
    for i, t in enumerate(tfs):
        ci, fi = cf(DEN[t], b)
        cp, fp = cf(POOL - DEN[t], b)
        old = np.log2((DEN[t][b] + 1) / ((POOL[b] - DEN[t][b]) / (len(tf79) - 1) + 1))
        for a, g in enumerate(gr):
            Fb[a, i] = np.log2((ci[g].sum() + .5) / (fi[g].sum() + .5)) - np.log2((cp[g].sum() + .5) / (fp[g].sum() + .5))
            Fa[a, i] = np.log2(((ci[g].sum() + .5) / NC[t]) / ((cp[g].sum() + .5) / (NTOT - NC[t])))
            Old[a, i] = old[g].mean()
    return Fb, Fa, Old, access


def bulkf(Bk, gr):
    gB = np.array([Bk[g].sum(0) for g in gr])
    return np.log2(gB / gB.sum(0, keepdims=True) * 1e6 + 1)


def block(b_, s_):
    k = b_.shape[1]
    return np.corrcoef(np.column_stack([b_, s_]).T)[k:, :k]


def perm(B):
    k = len(B)
    off_ = ~np.eye(k, dtype=bool)
    st = lambda M: M.diagonal().mean() - M[off_].mean()
    obs = st(B)
    null = np.array([st(B[list(q)]) for q in itertools.permutations(range(k))])
    return float(obs), float(np.mean(null >= obs - 1e-12))


S6 = pd.read_csv(OUT / "round2/summits.tsv", sep="\t", low_memory=False)
Bk6 = S6[[f"bulk_{t}" for t in TFS]].values
mask6 = S6.bulk_spec.values > 1
REFLAB6 = ["CTCF (mixed)", "MAZ (HEK293)", "ZBTB7A (HEK293)", "ZIC2 (HEK293)", "ZNF282 (HEK293T)", "ZNF777 (HEK293)"]
rows, art = [], []

# ---- 2 raw regression ----
z = np.load(OUT / "round2/summit_counts.npz", allow_pickle=True)
C79, F79, tot79 = z["C"].astype(np.int64), z["F"].astype(np.int64), z["tot79"].astype(float)
pi = [tf79.index(t) for t in TFS]
Cpool, Fpool = C79.sum(0), F79.sum(0)
gr = groups(S6, TFS, mask6, "all", 5000)
pdm = np.minimum(1, DM_DEPTH / tot79[pi])[:, None]
Rs = []
for s in DM_SEEDS:
    Cs, Fs = np.random.default_rng(s).binomial(C79[pi], pdm), np.random.default_rng(s + 1).binomial(F79[pi], pdm)
    sc = np.zeros((len(gr), 6))
    for a, g in enumerate(gr):
        cp = Cpool[g].sum() - C79[pi][:, g].sum(1); fp = Fpool[g].sum() - F79[pi][:, g].sum(1)
        sc[a] = np.log2((Cs[:, g].sum(1) + .5) / (Fs[:, g].sum(1) + .5)) - np.log2((cp + .5) / (fp + .5))
    Rs.append(block(bulkf(Bk6, gr), sc))
raw_B = np.mean(Rs, 0)
pub = pd.read_csv(OUT / "round2/pearson/r2p_corr_preferred_top5000_znf777.tsv", sep="\t", index_col=0).values[6:, :6]
print("raw regression, max abs difference from published block:", float(np.abs(raw_B - pub).max()), flush=True)
pd.DataFrame(raw_B, TFS, TFS).to_csv(OA / "check2_raw_block_recomputed.tsv", sep="\t", float_format="%.4f")

B6, A6 = {}, {}
for half, n in (("all", 5000), ("odd", 2500), ("even_X", 2500)):
    gr = groups(S6, TFS, mask6, half, n)
    Fb, Fa, Old, access = feats(S6, TFS, gr)
    b_ = bulkf(Bk6, gr)
    if half == "all":
        for i, t in enumerate(TFS):
            art.append(dict(tf=t, n_cells=NC[t], r_oldfeature_vs_group_accessibility=np.corrcoef(Old[:, i], access)[0, 1],
                            r_B_vs_accessibility=np.corrcoef(Fb[:, i], access)[0, 1],
                            r_A_vs_accessibility=np.corrcoef(Fa[:, i], access)[0, 1]))
    for nm, Fx, D_ in (("B_centre_flank", Fb, B6), ("A_enrich_cellnorm", Fa, A6)):
        B = block(b_, Fx)
        D_[half] = B
        o, p = perm(B)
        keep = list(range(1, 6))
        o5, p5 = perm(B[np.ix_(keep, keep)])
        rows.append(dict(recipe="6x6", feature=nm, half=half, n_groups=len(gr), diag_minus_off=o, perm_p=p, noCTCF_perm_p=p5,
                         own_max_rows=int(sum(np.argmax(B[i]) == i for i in range(6))),
                         own_max_cols=int(sum(np.argmax(B[:, j]) == j for j in range(6))),
                         **{f"r_own_{t}": B[i, i] for i, t in enumerate(TFS)}))
        pd.DataFrame(B, [f"imputed {t}" for t in TFS], REFLAB6).to_csv(OA / f"6x6_{nm}_{half}.tsv", sep="\t", float_format="%.4f")
art = pd.DataFrame(art)
art.to_csv(OA / "check3_scale_artefact.tsv", sep="\t", index=False, float_format="%.4f")
print(art.round(3).to_string(), flush=True)

# HEK293T 2x2
S2 = pd.read_csv(OUT / "hek293t_2x2/summits.tsv", sep="\t", low_memory=False)
T2 = ["CTCF", "ZNF282"]
BAM2 = [ROOT / "chip_gse103651/work/bam/gse103651.pooled.bam", D / "ZNF282_HEK293.bam"]
bb = OA / "w250.bed"
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
B2 = {}
for half in ("all", "odd", "even_X"):
    gr = groups(S2, T2, mask2, half, 5000)
    Fb, Fa, _, _ = feats(S2, T2, gr)
    b_ = bulkf(Bk2, gr)
    for nm, Fx in (("B_centre_flank", Fb), ("A_enrich_cellnorm", Fa)):
        B = block(b_, Fx)
        B2[(nm, half)] = B
        rows.append(dict(recipe="2x2", feature=nm, half=half, n_groups=len(gr), diag_minus_off=float((B[0, 0] + B[1, 1]) / 2 - (B[0, 1] + B[1, 0]) / 2),
                         own_max_rows=int(B[0, 0] > B[0, 1]) + int(B[1, 1] > B[1, 0]),
                         own_max_cols=int(B[0, 0] > B[1, 0]) + int(B[1, 1] > B[0, 1]),
                         r_own_CTCF=B[0, 0], r_own_ZNF282=B[1, 1], r_CTCF_vs_ZNF282ChIP=B[0, 1], r_ZNF282_vs_CTCFChIP=B[1, 0]))
sm = pd.DataFrame(rows)
sm.to_csv(OA / "summary.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 260)
print(sm.round(3).to_string())
print(pd.DataFrame(B6["all"], TFS, TFS).round(2).to_string())
print(pd.DataFrame(A6["all"], TFS, TFS).round(2).to_string())


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
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(title, fontsize=6)
    return im


lim = max(0.1, np.abs(raw_B).max(), np.abs(B6["all"]).max(), np.abs(A6["all"]).max())
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), layout="constrained")
heat(axs[0], raw_B, REFLAB6, [f"raw {t}" for t in TFS], "Raw (ZNF777 depth)\ncentre/flank vs pool", lim)
heat(axs[1], B6["all"], REFLAB6, [f"imputed {t}" for t in TFS], "Imputed, centre/flank vs pool\n(1 kb bins)", lim)
im = heat(axs[2], A6["all"], REFLAB6, [f"imputed {t}" for t in TFS], "Imputed, bin enrichment vs pool\n(cell-number normalised)", lim)
fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02, label="Pearson r")
for ext in ("png", "pdf"):
    fig.savefig(OA / f"6x6_raw_vs_imputed_corrected.{ext}", dpi=600 if ext == "png" else None)
