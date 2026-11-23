#!/usr/bin/env python
"""Round 8 (spec round8_mdrop5_diff/SPEC.md): differential bins called only inside the Mdrop5 accessibility mask, for
raw (R), Mdrop5 imputed all calls (I) and model-added calls only (Ia); scored vs bulk ChIP by summit groups (M2b) and
peak overlap (M1). Outputs OUT/round8_mdrop5_diff/: summary.tsv, counts.tsv, m_*.tsv, round8_results.xlsx, figs/*."""
import itertools
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
import scipy.sparse as sp
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.stats import binom

sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/downstream/fig2_qc")
from fig2_common import BLACKLIST, COUNTS, INP, OUT, REF, ROOT, TFS, bin_index, interval_mask, read_bed  # noqa: E402

MASK = "Mdrop5"
O8 = OUT / "round8_mdrop5_diff"; (O8 / "figs").mkdir(parents=True, exist_ok=True)
G, NT = 30, 6
CH = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
CELL = {"CTCF": "HEK293", "MAZ": "HEK293", "ZBTB7A": "HEK293", "ZIC2": "HEK293", "ZNF282": "HEK293T", "ZNF777": "HEK293"}
PEAKS = dict(REF); PEAKS["CTCF"] = ROOT / "downstream/tracks/ctcf_bulk_sources/1_ENCODE_HEK293_ENCFF314ZAL_peaks.bed"
bi, off, N = bin_index()
fr = np.load(INP / "frame.npz", allow_pickle=True)
tf79 = [str(t) for t in fr["tf79"]]; ncell = fr["ncells_imp79"]; atac = fr["atac_union"].astype(bool)
pi = [tf79.index(t) for t in TFS]

# universe: Mdrop5 bins on the frame, chr1-22 + X, minus blacklist
mk = np.load(ROOT / "regulatome/m46_stage1_imputed/work/masks_extra.npz")[MASK]
reg = pd.read_csv(ROOT / "unified/work/nfya/impute/regions.tsv", header=None)[0].str.extract(r"^(chr[^:]+):(\d+)-")
kp = reg[0].isin(off).values
idx = np.array([off[c][0] if c in off else 0 for c in reg[0]]) + reg[1].astype(np.int64).values // 1000
umask = np.zeros(N, bool); umask[idx[kp]] = mk[kp]
chrom_of = np.empty(N, dtype=object)
for c, (o, nb) in off.items():
    chrom_of[o:o + nb] = c
inCH = np.isin(chrom_of, CH)
bl = interval_mask(read_bed(BLACKLIST), off, N)
U = np.flatnonzero(umask & inCH & ~bl)
cn = np.array([23 if c == "chrX" else int(c[3:]) for c in chrom_of[U]])
print("universe bins", len(U), flush=True)

# bulk ChIP CPM per 1 kb bin and reference peak masks
BW = OUT / "round6_diffpeaks/heatmaps/bw"
bulk = np.zeros((NT, N), np.float32)
for j, t in enumerate(TFS):
    bw = pyBigWig.open(str(BW / f"{t}.bw"))
    for c in CH:
        o, nb = off[c]; L = bw.chroms().get(c)
        if L is None:
            continue
        v = np.array(bw.stats(c, 0, nb * 1000 if nb * 1000 <= L else L, nBins=nb if nb * 1000 <= L else nb - 1, type="mean"), dtype=float)
        bulk[j, o:o + len(v)] = np.nan_to_num(v)
pkm = np.vstack([interval_mask(read_bed(PEAKS[t]), off, N) for t in TFS])

FL = np.r_[-5:-1, 2:6]


def flank_idx(b):
    o = np.array([off[c][0] for c in chrom_of[b]]); nb = np.array([off[c][1] for c in chrom_of[b]])
    return np.clip(b[:, None] + FL[None, :], o[:, None], (o + nb - 1)[:, None])


def load_variant(v):
    if v == "R":
        M = sp.load_npz(COUNTS / "raw.npz").tocsr()
        stats = pd.read_csv(COUNTS / "tf_stats.tsv", sep="\t"); assert list(stats.tf) == tf79
        return M, 0.5
    M = sp.load_npz(INP / ("imp79_mdrop5.npz" if v == "I" else "imp79_mdrop5_added.npz")).tocsr()
    return M, None


def perm(M):
    k = len(M); o = ~np.eye(k, dtype=bool); st = lambda A: np.nanmean(np.diag(A)) - np.nanmean(A[o])
    obs = st(M); null = np.array([st(M[list(p)]) for p in itertools.permutations(range(k))])
    return float(obs), float(np.mean(null >= obs - 1e-12))


rows, counts, store = [], [], {}
for v in ("R", "I", "Ia"):
    M, e = load_variant(v)
    X = np.vstack([M[k, U].toarray().ravel() for k in range(79)]).astype(float)   # counts in universe bins
    tot = X.sum(1); n = X.sum(0)
    sets = {}
    for i, t in enumerate(TFS):
        k = pi[i]; x = X[k]; p0 = tot[k] / tot.sum()
        ok = n > 0
        p = np.ones(len(U)); p[ok] = binom.sf(np.round(x[ok]) - 1, np.round(n[ok]).astype(np.int64), p0)
        m = ok.sum(); od = np.argsort(p[ok]); qq = p[ok][od] * m / np.arange(1, m + 1)
        qq = np.minimum.accumulate(qq[::-1])[::-1]; q = np.ones(len(U)); tq = np.empty(m); tq[od] = np.minimum(qq, 1); q[ok] = tq
        lfc = np.log2(((x + .5) / tot[k]) / ((n - x + .5) / (tot.sum() - tot[k])))
        sig = np.flatnonzero((q < 0.05) & (lfc > 1)); sig = sig[np.argsort(p[sig], kind="stable")]
        sets[t] = sig
        counts.append(dict(variant=v, TF=t, significant=len(sig), in_ATAC_peak_union_top1000=float(atac[U[sig[:1000]]].mean()) if len(sig) else np.nan))
    # signal for sc scores (full genome frame, the variant's own data)
    full = M if e is not None else None
    if e is None:
        e = 1e-4
        V = np.vstack([M[k].toarray().ravel() / ncell[k] for k in pi])
        Vpool = np.asarray(M.multiply(1 / ncell[:, None]).sum(0)).ravel() if False else sum(M[k].toarray().ravel() / ncell[k] for k in range(79))
    else:
        V = np.vstack([M[k].toarray().ravel() for k in pi]).astype(float)
        Vpool = np.asarray(M.sum(0)).ravel().astype(float)
    for Nn in (1000, 5000):
        for half in ("all", "odd", "even_X"):
            hm = {"all": np.ones(len(U), bool), "odd": (cn % 2 == 1) & (cn != 23), "even_X": (cn % 2 == 0) | (cn == 23)}[half]
            sel = {t: sets[t][:Nn][hm[sets[t][:Nn]]] for t in TFS}
            gr = [U[g] for t in TFS for g in np.array_split(sel[t], G) if len(g)]
            if len(gr) < 12:
                continue
            gB = np.array([bulk[:, g].sum(1) for g in gr]); B = np.log2(gB / gB.sum(0, keepdims=True) * 1e6 + 1)
            sc = np.zeros((len(gr), NT))
            for a, g in enumerate(gr):
                fb = flank_idx(g)
                cp, fp = Vpool[g].sum(), Vpool[fb].mean(1).sum()
                for i in range(NT):
                    c, f = V[i, g].sum(), V[i][fb].mean(1).sum()
                    sc[a, i] = np.log2((c + e) / (f + e)) - np.log2((cp - c + e) / (fp - f + e))
            M2b = np.corrcoef(np.column_stack([B, sc]).T)[NT:, :NT]
            # empty sets -> NaN rows
            for i, t in enumerate(TFS):
                if len(sel[t]) == 0:
                    M2b[i] = np.nan
            M1 = np.full((NT, NT), np.nan)
            for i, t in enumerate(TFS):
                s_ = U[sel[t]]
                if len(s_) == 0:
                    continue
                for j in range(NT):
                    a_ = pkm[j, s_].sum(); b_ = len(s_) - a_; c_ = pkm[j, U[hm]].sum(); d_ = hm.sum() - c_
                    M1[i, j] = np.log2(((a_ + .5) / (b_ + .5)) / ((c_ + .5) / (d_ + .5)))
            for met, Mx in (("M2b", M2b), ("M1", M1)):
                Bf = np.nan_to_num(Mx, nan=-9)
                obs, p = perm(Bf); _, p5 = perm(Bf[1:, 1:])
                store[(v, Nn, met, half)] = Mx
                rows.append(dict(variant=v, top_n=Nn, metric=met, half=half, n_groups=len(gr), diag_minus_off=obs, perm_p=p, noCTCF_perm_p=p5,
                                 own_best_rows=int(sum(np.isfinite(Mx[i]).any() and np.nanargmax(Mx[i]) == i for i in range(NT))),
                                 own_is_row_max=",".join(t for i, t in enumerate(TFS) if np.isfinite(Mx[i]).any() and np.nanargmax(Mx[i]) == i),
                                 **{f"own_{t}": Mx[i, i] for i, t in enumerate(TFS)}))
                pd.DataFrame(Mx, [f"{t} sc" for t in TFS], [f"{t} ChIP" for t in TFS]).to_csv(O8 / f"m_{v}_top{Nn}_{met}_{half}.tsv", sep="\t", float_format="%.4f")
            print(v, Nn, half, "M2b p", round(rows[-2]["perm_p"], 3), "M1 p", round(rows[-1]["perm_p"], 3), flush=True)
    del X
T = pd.DataFrame(rows); C = pd.DataFrame(counts)
T.to_csv(O8 / "summary.tsv", sep="\t", index=False, float_format="%.4f"); C.to_csv(O8 / "counts.tsv", sep="\t", index=False, float_format="%.4f")
with pd.ExcelWriter(O8 / "round8_results.xlsx") as xw:
    C.to_excel(xw, sheet_name="counts", index=False); T.to_excel(xw, sheet_name="summary", index=False)

sys.path.insert(0, str(ROOT / "paper_figures"))
from common import style  # noqa: E402
P = style.use()
DIV = LinearSegmentedColormap.from_list("div", ["#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B"])
DIV.set_bad("#E3E3E3")
VN = {"R": "raw", "I": "imputed (raw-positive + model calls)", "Ia": "imputed, model-added only"}
for (v, Nn, met, half), Mx in store.items():
    if half != "all":
        continue
    r = T[(T.variant == v) & (T.top_n == Nn) & (T.metric == met) & (T.half == "all")].iloc[0]
    lim = 0.2 if met == "M2b" else 2.0
    fig = style.figure_mm(89, 82)
    ax = fig.add_axes([0.27, 0.30, 0.54, 0.52])
    im = ax.imshow(np.ma.masked_invalid(np.clip(Mx, -lim, lim)), cmap=DIV, vmin=-lim, vmax=lim)
    for i in range(NT):
        for j in range(NT):
            val = Mx[i, j]
            ax.text(j, i, "n/a" if not np.isfinite(val) else f"{val:+.2f}", ha="center", va="center", fontsize=6,
                    color="white" if np.isfinite(val) and abs(val) > 0.75 * lim else P.neutral.ink)
        ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=P.neutral.ink, lw=0.8))
    ax.set_xticks(range(NT), [f"{t} ChIP\n({CELL[t]})" for t in TFS], rotation=90)
    ax.set_yticks(range(NT), [f"{t} sc" for t in TFS]); ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(f"{MASK} differential bins, {VN[v]}, top {Nn:,}, {'summit groups r' if met == 'M2b' else 'peak overlap log2 OR'}\n"
                 f"own best {r.own_best_rows}/6, p = {r.perm_p:.3f} (without CTCF {r.noCTCF_perm_p:.3f})", fontsize=6)
    cax = fig.add_axes([0.84, 0.30, 0.03, 0.52]); cb = fig.colorbar(im, cax=cax); cb.outline.set_linewidth(0.4)
    cb.set_label("Pearson r (colour clipped at ±0.2)" if met == "M2b" else "log2 odds ratio (±2)")
    style.save(fig, O8 / "figs", f"r8_{v}_top{Nn}_{met}")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(C.pivot_table(index="TF", columns="variant", values="significant").to_string())
print(T[T.half == "all"].drop(columns=[c for c in T if c.startswith("own_") and c != "own_is_row_max"]).round(3).to_string())
print("done")
