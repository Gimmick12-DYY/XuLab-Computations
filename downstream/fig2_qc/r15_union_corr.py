#!/usr/bin/env python
"""Round 6c: sc vs bulk correlation over union peaks (round6_SPEC.md addendum 6c). Outputs OUT/round6_diffpeaks/union_6c/."""
import itertools
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from scipy.stats import rankdata

from fig2_common import BAM, D, DM_DEPTH, DM_SEEDS, INP, OUT, REF, ROOT, TFS, bin_index

OU = OUT / "round6_diffpeaks" / "union_6c"
(OU / "figs").mkdir(parents=True, exist_ok=True)
tmp = OU / "tmp"; tmp.mkdir(exist_ok=True)
CH = [f"chr{i}" for i in list(range(1, 23)) + ["X"]]
REFS = dict(REF); REFS["CTCF"] = D / "CTCF_HEK293_peaks.bed"
BAMS = dict(BAM); BAMS["CTCF"] = D / "ENCFF139DCW.bam"
LAB = "/vast/som/xujie_lab/TAGATG-293T/bigwig/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"
CELL = {"CTCF": "HEK293", "MAZ": "HEK293", "ZBTB7A": "HEK293", "ZIC2": "HEK293", "ZNF282": "HEK293T", "ZNF777": "HEK293"}
G = 33


def sh(*a):
    return subprocess.run(a, check=True, capture_output=True, text=True).stdout


def bed3(p, q05=False):
    d = pd.read_csv(p, sep="\t", header=None, usecols=range(9) if q05 else [0, 1, 2])
    if q05:
        d = d[d[8] >= 1.30103]
    return d[[0, 1, 2]]


def region_set(name, parts):
    c = pd.concat(parts); c = c[c[0].isin(CH)].sort_values([0, 1])
    c.to_csv(tmp / f"{name}_raw.bed", sep="\t", header=False, index=False)
    (tmp / f"{name}.bed").write_text(sh("bedtools", "merge", "-i", str(tmp / f"{name}_raw.bed")))
    return pd.read_csv(tmp / f"{name}.bed", sep="\t", header=None)


bulk_parts = [bed3(REFS[t]) for t in TFS]
SETS = {"U1": region_set("U1", bulk_parts), "U2": region_set("U2", bulk_parts + [bed3(LAB.format(t=t), True) for t in TFS])}
bi, off, N = bin_index()
fr = np.load(INP / "frame.npz", allow_pickle=True)
tf79 = [str(t) for t in fr["tf79"]]; atac = fr["atac_union"]
imp = sp.load_npz(INP / "imp79.npz").tocsr()
IMPCS = {t: np.concatenate([[0], np.cumsum(imp[tf79.index(t)].toarray().ravel().astype(float))]) for t in TFS}
ATCS = np.concatenate([[0], np.cumsum(atac.astype(int))])
five = {t: dict(np.load(INP / "five" / f"{t}.npz")) for t in TFS}
tot = {t: sum(len(v) for v in five[t].values()) for t in TFS}
cn_of = lambda ch: np.array([int(c.replace("chr", "").replace("X", "23")) for c in ch])


def lc(X):
    return np.log2(X / X.sum(0, keepdims=True) * 1e6 + 1)


def perm(Bm):
    k = len(Bm); o_ = ~np.eye(k, dtype=bool)
    st = lambda M: M.diagonal().mean() - M[o_].mean()
    obs = st(Bm); null = np.array([st(Bm[list(q)]) for q in itertools.permutations(range(k))])
    return float(obs), float(np.mean(null >= obs - 1e-12))


def corr(Xb, Xs, method):
    if method == "spearman":
        Xb = np.apply_along_axis(rankdata, 0, Xb); Xs = np.apply_along_axis(rankdata, 0, Xs)
    return np.corrcoef(np.column_stack([Xb, Xs]).T)[6:, :6]


rows, store, nreport = [], {}, []
for sname, reg in SETS.items():
    bed = tmp / f"{sname}.bed"
    out = sh("bedtools", "multicov", "-bams", *[str(BAMS[t]) for t in TFS], "-bed", str(bed))
    Bc = np.array([[float(x) for x in l.split("\t")[3:]] for l in out.splitlines()])
    Sc = np.zeros((len(reg), 6))
    for k, t in enumerate(TFS):
        for ch in CH:
            m = np.flatnonzero(reg[0].values == ch)
            if len(m):
                p = five[t][ch]; Sc[m, k] = np.searchsorted(p, reg[2].values[m]) - np.searchsorted(p, reg[1].values[m])
    o = np.array([off[c][0] for c in reg[0]]); a = o + reg[1].values // 1000; b = o + (reg[2].values - 1) // 1000
    Ic = np.column_stack([IMPCS[t][b + 1] - IMPCS[t][a] for t in TFS])
    inmask = (ATCS[b + 1] - ATCS[a]) > 0
    ovl = {}
    for t in TFS:
        rb = tmp / f"ref_{t}.bed"; bed3(REFS[t]).sort_values([0, 1]).to_csv(rb, sep="\t", header=False, index=False)
        hit = sh("bedtools", "intersect", "-u", "-a", str(bed), "-b", str(rb))
        hs = set((l.split("\t")[0], int(l.split("\t")[1])) for l in hit.splitlines())
        ovl[t] = np.array([(c, s) in hs for c, s in zip(reg[0], reg[1])])
    cnum = cn_of(reg[0].values)
    H = {"all": np.ones(len(reg), bool), "odd": (cnum % 2 == 1) & (cnum != 23), "even_X": (cnum % 2 == 0) | (cnum == 23)}
    nreport.append(dict(region_set=sname, n_peaks=len(reg), n_in_ATAC_mask=int(inmask.sum())))
    pdm = np.array([min(1, DM_DEPTH / tot[t]) for t in TFS])
    arms = {"raw_full": [Sc], "raw_znf777": [np.random.default_rng(s).binomial(Sc.astype(np.int64), pdm).astype(float) for s in DM_SEEDS],
            "imputed": [Ic]}
    for half in ("all", "odd", "even_X"):
        for arm, reps in arms.items():
            keep = H[half] & (inmask if arm == "imputed" else True)
            methods = [("a", "pearson"), ("b", "pearson"), ("c", "pearson")] + ([("a", "spearman"), ("b", "spearman")] if arm.startswith("raw") else [])
            for mth, cm in methods:
                Ms = []
                for X in reps:
                    if mth in ("a", "b"):
                        Xb, Xs = lc(Bc[keep]), lc(X[keep])
                        if mth == "b":
                            Xb = Xb - Xb.mean(1, keepdims=True); Xs = Xs - Xs.mean(1, keepdims=True)
                        Ms.append(corr(Xb, Xs, cm))
                    else:
                        gr = []
                        for j, t in enumerate(TFS):
                            ix = np.flatnonzero(keep & ovl[t]); ix = ix[np.argsort(-Bc[ix, j], kind="stable")]
                            gr += [g for g in np.array_split(ix, G) if len(g)]
                        gB = np.array([Bc[g].sum(0) for g in gr]); gS = np.array([X[g].sum(0) for g in gr])
                        Ms.append(corr(lc(gB), lc(gS), cm))
                Bm = np.mean(Ms, 0); tag = f"{sname}_{arm}_{mth}_{cm}"
                store[(tag, half)] = Bm
                ob, pv = perm(Bm); kk = list(range(1, 6)); _, p5 = perm(Bm[np.ix_(kk, kk)])
                rows.append(dict(region_set=sname, sc=arm, method=mth, corr=cm, half=half, n_peaks=int(keep.sum()),
                                 own_best_rows=int(sum(np.argmax(Bm[i]) == i for i in range(6))), own_best_cols=int(sum(np.argmax(Bm[:, j]) == j for j in range(6))),
                                 diag_minus_off=ob, perm_p=pv, noCTCF_perm_p=p5, **{f"own_{t}": Bm[i, i] for i, t in enumerate(TFS)}))
                pd.DataFrame(Bm, [f"{t} sc" for t in TFS], [f"{t} ChIP" for t in TFS]).to_csv(OU / f"m_{tag}_{half}.tsv", sep="\t", float_format="%.4f")
    print(sname, "done", flush=True)
sm = pd.DataFrame(rows); nr = pd.DataFrame(nreport)
sm.to_csv(OU / "summary.tsv", sep="\t", index=False, float_format="%.4f")
with pd.ExcelWriter(OU / "union_6c_results.xlsx") as xw:
    nr.to_excel(xw, sheet_name="regions", index=False); sm.to_excel(xw, sheet_name="summary", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(nr.to_string()); print(sm[sm.half == "all"].round(3).to_string())

for f in (ROOT.parent / "fonts/arial").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402
P = rp.use(); plt.rcParams["font.sans-serif"] = ["Arial"] + plt.rcParams["font.sans-serif"]; plt.rcParams["svg.fonttype"] = "none"
MN = {"a": "plain", "b": "peak-centred (shared signal removed)", "c": "grouped (198 groups)"}
for sname in SETS:
    for arm in ("raw_full", "raw_znf777", "imputed"):
        keys = [k for (k, h) in store if h == "all" and k.startswith(f"{sname}_{arm}_")]
        fig, axs = plt.subplots(1, len(keys), figsize=(3.1 * len(keys), 3.1), layout="constrained")
        for ax, k in zip(np.atleast_1d(axs), sorted(keys)):
            Bm = store[(k, "all")]; _, _, mth, cm = k.split("_", 3) if arm != "raw_full" and arm != "raw_znf777" else (None, None, k.split("_")[-2], k.split("_")[-1])
            mth, cm = k.split("_")[-2], k.split("_")[-1]
            r = sm[(sm.region_set == sname) & (sm.sc == arm) & (sm.method == mth) & (sm["corr"] == cm) & (sm.half == "all")].iloc[0]
            lim = float(np.ceil(max(0.1, np.abs(Bm).max()) * 10) / 10)
            im = ax.imshow(Bm, cmap=P.div_cmap(), vmin=-lim, vmax=lim)
            for i in range(6):
                for j in range(6):
                    ax.text(j, i, f"{Bm[i, j]:+.2f}", ha="center", va="center", fontsize=5.5, color="white" if abs(Bm[i, j]) > 0.6 * lim else P.neutral.ink)
                ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=P.neutral.ink, lw=0.8))
            ax.set_xticks(range(6), [f"{t} ChIP\n({CELL[t]})" for t in TFS], rotation=90, fontsize=5.5)
            ax.set_yticks(range(6), [f"{t} sc" for t in TFS], fontsize=6); ax.tick_params(length=0)
            for s_ in ax.spines.values():
                s_.set_visible(False)
            ax.set_title(f"{sname}, {arm}, {MN[mth]}, {cm}\nn = {r.n_peaks:,}; own best {r.own_best_rows}/6, p = {r.perm_p:.3f} (no CTCF {r.noCTCF_perm_p:.3f})", fontsize=5.5)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label(f"{cm.capitalize()} r; range ±{lim:g}", fontsize=5.5)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(OU / "figs" / f"{sname}_{arm}.{ext}", dpi=400)
        plt.close(fig)
print("done")
