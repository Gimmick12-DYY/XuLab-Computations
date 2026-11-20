#!/usr/bin/env python
"""Round 6 (differential peaks), step 2: differential sets D-A / D-B, metrics M1 / M2a / M2b, statistics, figures
(round6_SPEC.md). Outputs OUT/round6_diffpeaks/: summary.tsv/.xlsx, pertf.tsv, diffset_counts.tsv, m_<...>.tsv, figs/*.png/pdf/svg."""
import itertools
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from scipy.stats import binom

from fig2_common import OUT, ROOT, TFS

O6 = OUT / "round6_diffpeaks"
(O6 / "figs").mkdir(exist_ok=True)
G = 30
R = pd.read_csv(O6 / "rows.tsv", sep="\t")
z = np.load(O6 / "regions.npz", allow_pickle=True)
SC, TOT, tf79, BK, MAPPED, OV, AT, GC, COLS = (z[k] for k in ("SC", "TOT", "tf79", "BK", "MAPPED", "OV", "AT", "GC", "cols"))
tf79, COLS = list(tf79), list(COLS)
SC = SC.astype(np.int64)
POOL = SC.sum(0)
cn = R.chrom.str.replace("chr", "").replace("X", "23").astype(int).values
HALF = {"all": np.ones(len(R), bool), "odd": (cn % 2 == 1) & (cn != 23), "even_X": (cn % 2 == 0) | (cn == 23)}
cand = np.flatnonzero(R.kind.values == "cand")
CELL = {"CTCF": "HEK293", "MAZ": "HEK293", "ZBTB7A": "HEK293", "ZIC2": "HEK293", "ZNF282": "HEK293T", "ZNF777": "HEK293"}


def lfc(k, idx):
    ni, no = SC[k, idx], POOL[idx] - SC[k, idx]
    Ri, Ro = TOT[k], TOT.sum() - TOT[k]
    return np.log2(((ni + .5) / Ri) / ((no + .5) / Ro))


# ---- differential sets ----
SETS, counts = {}, []
for t in TFS:
    k = tf79.index(t)
    ni = SC[k, cand]; n = POOL[cand]
    p0 = TOT[k] / TOT.sum()
    ok = n > 0
    p = np.ones(len(cand)); p[ok] = binom.sf(ni[ok] - 1, n[ok], p0)
    m = ok.sum(); o = np.argsort(p[ok]); q = np.ones(len(cand))
    qq = p[ok][o] * m / np.arange(1, m + 1); qq = np.minimum.accumulate(qq[::-1])[::-1]
    tmpq = np.empty(m); tmpq[o] = np.minimum(qq, 1); q[ok] = tmpq
    sig = np.flatnonzero((q < 0.05) & (lfc(k, cand) > 1))
    sig = sig[np.argsort(p[sig], kind="stable")]
    da = np.flatnonzero(R.kind.values == f"DA_{t}")
    da = da[np.argsort(-R.p.values[da], kind="stable")]
    for N in (1000, 5000):
        SETS[("D-B", N, t)] = cand[sig[:N]]
        SETS[("D-A", N, t)] = da[:N]
    counts.append(dict(tf=t, DB_significant=len(sig), DB_top1000=min(1000, len(sig)), DB_top5000=min(5000, len(sig)),
                       DA_q05_kept=len(da), DA_top1000=min(1000, len(da)), DA_top5000=min(5000, len(da))))
np.savez(O6 / "diff_sets.npz", **{f"{d}_{n}_{t}": v for (d, n, t), v in SETS.items()})
cdf = pd.DataFrame(counts)
cdf.to_csv(O6 / "diffset_counts.tsv", sep="\t", index=False)
print(cdf.to_string(), flush=True)


def deciles(x, base):
    qs = np.unique(np.quantile(x[base], np.linspace(0, 1, 11)[1:-1]))
    return np.searchsorted(qs, x, side="right")


STRAT = deciles(AT, cand) * 10 + deciles(GC, cand)
rng = np.random.default_rng(20261120)


def perm(B):
    k = len(B)
    o_ = ~np.eye(k, dtype=bool)
    st = lambda M: M.diagonal().mean() - M[o_].mean()
    obs = st(B)
    null = np.array([st(B[list(q)]) for q in itertools.permutations(range(k))])
    return float(obs), float(np.mean(null >= obs - 1e-12))


def colidx(ctcf):
    return [COLS.index("CTCF_nexus" if (t == "CTCF" and ctcf == "nexus") else t) for t in TFS]


def matrices(dset, N, half, ctcf):
    ci = colidx(ctcf)
    sel = {t: SETS[(dset, N, t)][HALF[half][SETS[(dset, N, t)]]] for t in TFS}
    M1 = np.zeros((6, 6)); M2a = np.zeros((6, 6))
    cpool = cand[HALF[half][cand]]
    for i, t in enumerate(TFS):
        s = sel[t]
        bg_pool = np.setdiff1d(cpool, s)
        bs = STRAT[bg_pool]; order = np.argsort(bs, kind="stable"); bg_pool, bs = bg_pool[order], bs[order]
        bg = []
        for st_ in STRAT[s]:
            lo, hi = np.searchsorted(bs, st_), np.searchsorted(bs, st_, side="right")
            if hi > lo:
                bg.append(bg_pool[rng.integers(lo, hi, 10)])
        bg = np.concatenate(bg) if bg else np.array([], int)
        k = tf79.index(t)
        x = lfc(k, s)
        for j, c in enumerate(ci):
            a = OV[s, c].sum(); b = len(s) - a; cc = OV[bg, c].sum(); d = len(bg) - cc
            M1[i, j] = np.log2(((a + .5) / (b + .5)) / ((cc + .5) / (d + .5)))
            y = np.log2(BK[s, c] / MAPPED[c] * 1e6 + 1)
            M2a[i, j] = np.corrcoef(x, y)[0, 1] if len(s) > 2 else np.nan
    gr = []
    for t in TFS:
        s = sel[t]
        if len(s) == 0:
            continue
        gr += [g for g in np.array_split(s, G) if len(g)]
    gB = np.array([BK[g][:, ci].sum(0) for g in gr])
    bulk = np.log2(gB / gB.sum(0, keepdims=True) * 1e6 + 1)
    sc = np.zeros((len(gr), 6))
    for i, t in enumerate(TFS):
        k = tf79.index(t)
        for a_, g in enumerate(gr):
            ni = SC[k, g].sum(); no = POOL[g].sum() - ni
            sc[a_, i] = np.log2(((ni + .5) / TOT[k]) / ((no + .5) / (TOT.sum() - TOT[k])))
    M2b = np.corrcoef(np.column_stack([bulk, sc]).T)[6:, :6]
    return {"M1": M1, "M2a": M2a, "M2b": M2b}, {t: len(sel[t]) for t in TFS}


rows, pertf, store = [], [], {}
nver = 0
for dset, N, ctcf in itertools.product(("D-A", "D-B"), (1000, 5000), ("ENCODE_HEK293", "nexus")):
    for half in ("all", "odd", "even_X"):
        Ms, ns = matrices(dset, N, half, ctcf)
        for met, B in Ms.items():
            if half == "all":
                nver += 1
            store[(dset, N, ctcf, met, half)] = B
            Bf = np.nan_to_num(B, nan=-9)
            o, p = perm(Bf); kk = list(range(1, 6)); o5, p5 = perm(Bf[np.ix_(kk, kk)])
            rows.append(dict(diffset=dset, top_n=N, ctcf_ref=ctcf, metric=met, half=half, own_best_rows=int(sum(np.argmax(Bf[i]) == i for i in range(6))),
                             own_best_cols=int(sum(np.argmax(Bf[:, j]) == j for j in range(6))), diag_minus_off=o, perm_p=p,
                             noCTCF_perm_p=p5, **{f"n_{t}": ns[t] for t in TFS}))
            for i, t in enumerate(TFS):
                pertf.append(dict(diffset=dset, top_n=N, ctcf_ref=ctcf, metric=met, half=half, tf=t, own=B[i, i],
                                  best_other=float(np.nanmax(np.delete(B[i], i))), best_other_ref=np.delete(np.array(TFS), i)[int(np.nanargmax(np.delete(B[i], i)))]))
            pd.DataFrame(B, [f"{t} sc" for t in TFS], [f"{t} ChIP" for t in TFS]).to_csv(
                O6 / f"m_{dset}_top{N}_{ctcf}_{met}_{half}.tsv", sep="\t", float_format="%.4f")
sm = pd.DataFrame(rows); pt = pd.DataFrame(pertf)
sm.to_csv(O6 / "summary.tsv", sep="\t", index=False, float_format="%.4f")
pt.to_csv(O6 / "pertf.tsv", sep="\t", index=False, float_format="%.4f")
try:
    with pd.ExcelWriter(O6 / "round6_results.xlsx") as xw:
        cdf.to_excel(xw, sheet_name="diffset_counts", index=False); sm.to_excel(xw, sheet_name="summary", index=False); pt.to_excel(xw, sheet_name="per_TF", index=False)
except Exception as e:
    print("xlsx not written:", e)
print("matrices (all chromosomes):", nver)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(sm[sm.half == "all"].drop(columns=[c for c in sm if c.startswith("n_")]).round(3).to_string())

for f in (ROOT.parent / "fonts/arial").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402
P = rp.use()
plt.rcParams["font.sans-serif"] = ["Arial"] + plt.rcParams["font.sans-serif"]
plt.rcParams["svg.fonttype"] = "none"
LAB = {"M1": "log2 odds ratio (overlap vs matched background)", "M2a": "Pearson r (sc log2FC vs bulk, per region)",
       "M2b": "Pearson r (summit groups)"}


def draw(ax, B, met, ctcf, title):
    lim = max(0.1, float(np.nanmax(np.abs(B))))
    lim = float(np.ceil(lim * 10) / 10)
    im = ax.imshow(B, cmap=P.div_cmap(), vmin=-lim, vmax=lim)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{B[i, j]:+.2f}", ha="center", va="center", fontsize=5.5, color="white" if abs(B[i, j]) > 0.6 * lim else P.neutral.ink)
        ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=P.neutral.ink, lw=0.8))
    cl = dict(CELL); cl["CTCF"] = "HEK293T nexus" if ctcf == "nexus" else "HEK293"
    ax.set_xticks(range(6), [f"{t} ChIP\n({cl[t]})" for t in TFS], rotation=90, fontsize=5.5)
    ax.set_yticks(range(6), [f"{t} sc diff." for t in TFS], fontsize=6.5)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(title, fontsize=6)
    return im, lim


for dset, N, ctcf in itertools.product(("D-A", "D-B"), (1000, 5000), ("ENCODE_HEK293", "nexus")):
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.2), layout="constrained")
    for ax, met in zip(axs, ("M1", "M2a", "M2b")):
        B = store[(dset, N, ctcf, met, "all")]
        r = sm[(sm.diffset == dset) & (sm.top_n == N) & (sm.ctcf_ref == ctcf) & (sm.metric == met) & (sm.half == "all")].iloc[0]
        im, lim = draw(ax, B, met, ctcf, f"{dset}, top {N}, {met}\nown best {r.own_best_rows}/6, p = {r.perm_p:.3f} (no CTCF {r.noCTCF_perm_p:.3f})")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label(f"{LAB[met]}; range ±{lim:g}", fontsize=5.5)
        cb.ax.tick_params(labelsize=5.5, length=2)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(O6 / "figs" / f"{dset}_top{N}_{ctcf}.{ext}", dpi=400)
    plt.close(fig)
print("done")
