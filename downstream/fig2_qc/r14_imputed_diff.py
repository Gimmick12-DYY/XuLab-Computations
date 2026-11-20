#!/usr/bin/env python
"""Round 6 imputed versions (round6_SPEC.md addendum): D-I differential sets from imputed counts on the D-B candidate
regions; I1 = D-I sets with imputed scores (M1, M2a, M2b); I2 = raw-called D-B sets with imputed scores (M2a, M2b);
I3 = D-I sets with raw scores (M2a, M2b). 2 sizes x 2 CTCF references; halves. Outputs OUT/round6_diffpeaks/imputed/."""
import itertools
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from scipy.stats import binom

from fig2_common import INP, OUT, ROOT, TFS, bin_index

O6 = OUT / "round6_diffpeaks"
OI = O6 / "imputed"
(OI / "figs").mkdir(parents=True, exist_ok=True)
G = 30
R = pd.read_csv(O6 / "rows.tsv", sep="\t")
z = np.load(O6 / "regions.npz", allow_pickle=True)
SC, TOT, tf79, BK, MAPPED, OV, AT, GC, COLS = (z[k] for k in ("SC", "TOT", "tf79", "BK", "MAPPED", "OV", "AT", "GC", "cols"))
tf79, COLS = list(tf79), list(COLS)
SC = SC.astype(np.int64); POOL = SC.sum(0)
cand = np.flatnonzero(R.kind.values == "cand")
cn = R.chrom.str.replace("chr", "").replace("X", "23").astype(int).values
HALF = {"all": np.ones(len(R), bool), "odd": (cn % 2 == 1) & (cn != 23), "even_X": (cn % 2 == 0) | (cn == 23)}
CELL = {"CTCF": "HEK293", "MAZ": "HEK293", "ZBTB7A": "HEK293", "ZIC2": "HEK293", "ZNF282": "HEK293T", "ZNF777": "HEK293"}
DB = np.load(O6 / "diff_sets.npz")

# imputed counts per row (sum of cells positive over overlapped 1 kb bins)
bi, off, N = bin_index()
fr = np.load(INP / "frame.npz", allow_pickle=True)
assert list(fr["tf79"]) == tf79
imp = sp.load_npz(INP / "imp79.npz").tocsr()
o = np.array([off[c][0] for c in R.chrom.values])
a = o + R.start.values // 1000
b = o + (R.end.values - 1) // 1000
IM = np.zeros((79, len(R)))
ITOT = np.zeros(79)
for k in range(79):
    v = imp[k].toarray().ravel().astype(float)
    ITOT[k] = v.sum()
    cs = np.concatenate([[0], np.cumsum(v)])
    IM[k] = cs[b + 1] - cs[a]
IPOOL = IM.sum(0)
print("imputed counts done", flush=True)


def lfc_from(Xk, Xpool, Tk, Ttot, idx):
    ni, no = Xk[idx], Xpool[idx] - Xk[idx]
    return np.log2(((ni + .5) / Tk) / ((no + .5) / (Ttot - Tk)))


SETS, counts = {}, []
for t in TFS:
    k = tf79.index(t)
    ni = np.round(IM[k, cand]).astype(np.int64); n = np.round(IPOOL[cand]).astype(np.int64)
    p0 = ITOT[k] / ITOT.sum()
    ok = n > 0
    p = np.ones(len(cand)); p[ok] = binom.sf(ni[ok] - 1, n[ok], p0)
    m = ok.sum(); od = np.argsort(p[ok]); q = np.ones(len(cand))
    qq = p[ok][od] * m / np.arange(1, m + 1); qq = np.minimum.accumulate(qq[::-1])[::-1]
    tq = np.empty(m); tq[od] = np.minimum(qq, 1); q[ok] = tq
    l = lfc_from(IM[k], IPOOL, ITOT[k], ITOT.sum(), cand)
    sig = np.flatnonzero((q < 0.05) & (l > 1)); sig = sig[np.argsort(p[sig], kind="stable")]
    for Nn in (1000, 5000):
        SETS[("D-I", Nn, t)] = cand[sig[:Nn]]
        SETS[("D-B", Nn, t)] = DB[f"D-B_{Nn}_{t}"]
    counts.append(dict(tf=t, DI_significant=len(sig), DI_top1000=min(1000, len(sig)), DI_top5000=min(5000, len(sig))))
cdf = pd.DataFrame(counts); print(cdf.to_string(), flush=True)
np.savez(OI / "DI_sets.npz", **{f"{d}_{n}_{t}": v for (d, n, t), v in SETS.items() if d == "D-I"})


def deciles(x, base):
    qs = np.unique(np.quantile(x[base], np.linspace(0, 1, 11)[1:-1]))
    return np.searchsorted(qs, x, side="right")


STRAT = deciles(AT, cand) * 10 + deciles(GC, cand)
rng = np.random.default_rng(20261120)


def perm(B):
    kk = len(B); o_ = ~np.eye(kk, dtype=bool)
    st = lambda M: M.diagonal().mean() - M[o_].mean()
    obs = st(B)
    null = np.array([st(B[list(q)]) for q in itertools.permutations(range(kk))])
    return float(obs), float(np.mean(null >= obs - 1e-12))


def score(kind, k, idx):
    if kind == "imputed":
        return lfc_from(IM[k], IPOOL, ITOT[k], ITOT.sum(), idx)
    return lfc_from(SC[k].astype(float), POOL.astype(float), TOT[k], TOT.sum(), idx)


def grp_score(kind, k, g):
    X, Xp, T, Tt = (IM[k], IPOOL, ITOT[k], ITOT.sum()) if kind == "imputed" else (SC[k], POOL, TOT[k], TOT.sum())
    ni = X[g].sum(); no = Xp[g].sum() - ni
    return np.log2(((ni + .5) / T) / ((no + .5) / (Tt - T)))


def matrices(dset, Nn, half, ctcf, kind, mets):
    ci = [COLS.index("CTCF_nexus" if (t == "CTCF" and ctcf == "nexus") else t) for t in TFS]
    sel = {t: SETS[(dset, Nn, t)][HALF[half][SETS[(dset, Nn, t)]]] for t in TFS}
    out = {}
    if "M1" in mets:
        M1 = np.zeros((6, 6)); cpool = cand[HALF[half][cand]]
        for i, t in enumerate(TFS):
            s = sel[t]; bgp = np.setdiff1d(cpool, s); bs = STRAT[bgp]; od = np.argsort(bs, kind="stable"); bgp, bs = bgp[od], bs[od]
            bg = [bgp[rng.integers(np.searchsorted(bs, x), np.searchsorted(bs, x, side="right"), 10)] for x in STRAT[s]
                  if np.searchsorted(bs, x, side="right") > np.searchsorted(bs, x)]
            bg = np.concatenate(bg) if bg else np.array([], int)
            for j, c in enumerate(ci):
                A = OV[s, c].sum(); Bn = len(s) - A; C = OV[bg, c].sum(); Dn = len(bg) - C
                M1[i, j] = np.log2(((A + .5) / (Bn + .5)) / ((C + .5) / (Dn + .5)))
        out["M1"] = M1
    if "M2a" in mets:
        M2a = np.full((6, 6), np.nan)
        for i, t in enumerate(TFS):
            s = sel[t]
            if len(s) > 2:
                x = score(kind, tf79.index(t), s)
                for j, c in enumerate(ci):
                    M2a[i, j] = np.corrcoef(x, np.log2(BK[s, c] / MAPPED[c] * 1e6 + 1))[0, 1]
        out["M2a"] = M2a
    if "M2b" in mets:
        gr = [g for t in TFS for g in np.array_split(sel[t], G) if len(g)]
        gB = np.array([BK[g][:, ci].sum(0) for g in gr]); bulk = np.log2(gB / gB.sum(0, keepdims=True) * 1e6 + 1)
        scv = np.array([[grp_score(kind, tf79.index(t), g) for t in TFS] for g in gr])
        out["M2b"] = np.corrcoef(np.column_stack([bulk, scv]).T)[6:, :6]
    return out, {t: len(sel[t]) for t in TFS}


PLAN = [("I1", "D-I", "imputed", ("M1", "M2a", "M2b")), ("I2", "D-B", "imputed", ("M2a", "M2b")), ("I3", "D-I", "raw", ("M2a", "M2b"))]
rows, pertf, store, nv = [], [], {}, 0
for ver, dset, kind, mets in PLAN:
    for Nn, ctcf in itertools.product((1000, 5000), ("ENCODE_HEK293", "nexus")):
        for half in ("all", "odd", "even_X"):
            Ms, ns = matrices(dset, Nn, half, ctcf, kind, mets)
            for met, Bm in Ms.items():
                nv += half == "all"
                store[(ver, Nn, ctcf, met, half)] = Bm
                Bf = np.nan_to_num(Bm, nan=-9); ob, pv = perm(Bf); kk = list(range(1, 6)); _, p5 = perm(Bf[np.ix_(kk, kk)])
                rows.append(dict(version=ver, diffset=dset, scores=kind, top_n=Nn, ctcf_ref=ctcf, metric=met, half=half,
                                 own_best_rows=int(sum(np.argmax(Bf[i]) == i for i in range(6))), own_best_cols=int(sum(np.argmax(Bf[:, j]) == j for j in range(6))),
                                 diag_minus_off=ob, perm_p=pv, noCTCF_perm_p=p5, **{f"n_{t}": ns[t] for t in TFS}))
                for i, t in enumerate(TFS):
                    oth = np.delete(Bm[i], i)
                    pertf.append(dict(version=ver, top_n=Nn, ctcf_ref=ctcf, metric=met, half=half, tf=t, own=Bm[i, i], best_other=np.nanmax(oth),
                                      best_other_ref=np.delete(np.array(TFS), i)[int(np.nanargmax(oth))]))
                pd.DataFrame(Bm, [f"{t} sc" for t in TFS], [f"{t} ChIP" for t in TFS]).to_csv(OI / f"m_{ver}_top{Nn}_{ctcf}_{met}_{half}.tsv", sep="\t", float_format="%.4f")
sm = pd.DataFrame(rows); pt = pd.DataFrame(pertf)
sm.to_csv(OI / "summary.tsv", sep="\t", index=False, float_format="%.4f"); pt.to_csv(OI / "pertf.tsv", sep="\t", index=False, float_format="%.4f")
with pd.ExcelWriter(OI / "round6_imputed_results.xlsx") as xw:
    cdf.to_excel(xw, sheet_name="DI_counts", index=False); sm.to_excel(xw, sheet_name="summary", index=False); pt.to_excel(xw, sheet_name="per_TF", index=False)
print("matrices (all chromosomes):", nv)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(sm.drop(columns=[c for c in sm if c.startswith("n_")]).round(3).to_string())

for f in (ROOT.parent / "fonts/arial").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402
P = rp.use()
plt.rcParams["font.sans-serif"] = ["Arial"] + plt.rcParams["font.sans-serif"]; plt.rcParams["svg.fonttype"] = "none"
LAB = {"M1": "log2 odds ratio (overlap vs matched background)", "M2a": "Pearson r (per region)", "M2b": "Pearson r (summit groups)"}
VN = {"I1": "imputed-called sets, imputed scores", "I2": "raw-called sets, imputed scores", "I3": "imputed-called sets, raw scores"}
for ver, dset, kind, mets in PLAN:
    for Nn, ctcf in itertools.product((1000, 5000), ("ENCODE_HEK293", "nexus")):
        fig, axs = plt.subplots(1, len(mets), figsize=(3.2 * len(mets), 3.2), layout="constrained")
        for ax, met in zip(np.atleast_1d(axs), mets):
            Bm = store[(ver, Nn, ctcf, met, "all")]
            r = sm[(sm.version == ver) & (sm.top_n == Nn) & (sm.ctcf_ref == ctcf) & (sm.metric == met) & (sm.half == "all")].iloc[0]
            lim = float(np.ceil(max(0.1, np.nanmax(np.abs(Bm))) * 10) / 10)
            im = ax.imshow(Bm, cmap=P.div_cmap(), vmin=-lim, vmax=lim)
            for i in range(6):
                for j in range(6):
                    ax.text(j, i, f"{Bm[i, j]:+.2f}", ha="center", va="center", fontsize=5.5, color="white" if abs(Bm[i, j]) > 0.6 * lim else P.neutral.ink)
                ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=P.neutral.ink, lw=0.8))
            cl = dict(CELL); cl["CTCF"] = "HEK293T nexus" if ctcf == "nexus" else "HEK293"
            ax.set_xticks(range(6), [f"{t} ChIP\n({cl[t]})" for t in TFS], rotation=90, fontsize=5.5)
            ax.set_yticks(range(6), [f"{t} sc {'imputed' if kind == 'imputed' else 'raw'}" for t in TFS], fontsize=6)
            ax.tick_params(length=0)
            for s_ in ax.spines.values():
                s_.set_visible(False)
            ax.set_title(f"{VN[ver]}, top {Nn}, {met}\nown best {r.own_best_rows}/6, p = {r.perm_p:.3f} (no CTCF {r.noCTCF_perm_p:.3f})", fontsize=6)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label(f"{LAB[met]}; range ±{lim:g}", fontsize=5.5); cb.ax.tick_params(labelsize=5.5)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(OI / "figs" / f"{ver}_top{Nn}_{ctcf}.{ext}", dpi=400)
        plt.close(fig)
print("done")
