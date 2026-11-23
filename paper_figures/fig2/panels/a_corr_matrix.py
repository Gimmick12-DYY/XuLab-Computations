"""Fig 2 corr_matrix_6x6: summit-group Pearson r, single-cell raw (rows) vs bulk ChIP (columns), colour clipped at +-0.2
(diverging, centred at 0; every cell annotated). Diagonal-minus-off-diagonal p from all 720 row-label permutations,
also without CTCF (120). Input source_data/corr_matrix_6x6.tsv.
Supplement: the same rendering for the genome-wide 10 kb Pearson 6x6 (source_data/corr_10kb_6x6.tsv), colour +-0.6."""
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import style  # noqa: E402

CELL = {"CTCF": "HEK293", "MAZ": "HEK293", "ZBTB7A": "HEK293", "ZIC2": "HEK293", "ZNF282": "HEK293T", "ZNF777": "HEK293"}
DIV = ["#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B"]


def perm_p(M):
    k = len(M); off = ~np.eye(k, dtype=bool); st = lambda A: A.diagonal().mean() - A[off].mean()
    obs = st(M); null = np.array([st(M[list(p)]) for p in itertools.permutations(range(k))])
    return obs, float(np.mean(null >= obs - 1e-12))


def main(src, out):
    standardised(src, out)
    dropped(src, out)
    draw(src, out, "corr_matrix_6x6.tsv", "fig2_corr_matrix_6x6", 0.2, "Pearson r (colour clipped at ±0.2)", None)
    draw(src, out, "corr_10kb_6x6.tsv", "fig2_supp_corr_10kb_6x6", 0.6, "Pearson r, genome-wide 10 kb bins",
         "log2 CPM per 10 kb bin, genome-wide (Paired-Tag style)")
    for arm, lab in (("all", "imputed (raw-positive cells + model calls)"), ("added", "imputed, model-added calls only")):
        f = Path(src) / f"corr_imputed_Mdrop5_{arm}_6x6.tsv"
        if f.exists():
            draw(src, out, f.name, f"fig2_supp_corr_6x6_imputed_mdrop5{'' if arm == 'all' else '_added'}", 0.2,
                 "Pearson r (colour clipped at ±0.2)", f"sc {lab}, Mdrop5 mask, 1 kb")


def draw(src, out, fn, name, lim, cblabel, subtitle):
    P = style.use()
    D = pd.read_csv(Path(src) / fn, sep="\t", index_col=0)
    tfs = ["CTCF", "MAZ", "ZBTB7A", "ZIC2", "ZNF282", "ZNF777"]   # row/column order of the source table (TFS)
    M = D.values.astype(float)
    obs, p = perm_p(M); _, p5 = perm_p(M[1:, 1:])
    fig = style.figure_mm(89, 80)
    ax = fig.add_axes([0.25, 0.30, 0.56, 0.56])
    cm = LinearSegmentedColormap.from_list("div", DIV)
    im = ax.imshow(np.clip(M, -lim, lim), cmap=cm, vmin=-lim, vmax=lim)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(M[i, j]) > 0.75 * lim else P.neutral.ink)
        ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=P.neutral.ink, lw=0.8))
    ax.set_xticks(range(6), [f"{t} ChIP\n({CELL[t]})" for t in tfs], rotation=90)
    ax.set_yticks(range(6), [f"{t} sc{' imputed' if 'imputed' in fn else ''}" for t in tfs])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ttl = f"diagonal − off-diagonal = {obs:+.3f}, p = {p:.3f}\n(without CTCF p = {p5:.3f}; 720 / 120 permutations)"
    ax.set_title((subtitle + "\n" if subtitle else "") + ttl, fontsize=6)
    cax = fig.add_axes([0.84, 0.30, 0.03, 0.56])
    cb = fig.colorbar(im, cax=cax); cb.set_ticks(np.round(np.linspace(-lim, lim, 5), 2)); cb.outline.set_linewidth(0.4)
    cb.set_label(cblabel)
    style.save(fig, out, name)
    print("own r:", dict(zip(tfs, np.diag(M).round(3))), "p", round(p, 4), "noCTCF", round(p5, 4))


# ---- display-only standardised versions of the main matrix (same values; permutation p of the raw r matrix,
#      diag - off = +0.151, p = 0.006, without CTCF p = 0.033, goes in the legend, not on the panel) ----
STANDARDIZE = ("rowz", "rowminmax", "colz")   # rowz = 11-12 S2, colz = 11-12 S3 (same definitions, population SD)


def standardised(src, out, which=STANDARDIZE):
    P = style.use()
    D = pd.read_csv(Path(src) / "corr_matrix_6x6.tsv", sep="\t", index_col=0)
    M = D.values.astype(float)
    tfs = ["CTCF", "MAZ", "ZBTB7A", "ZIC2", "ZNF282", "ZNF777"]
    spec = {
        "rowz": ((M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True), "div", 2.0,
                 "row z-score of Pearson r\n(per sc row)"),
        "rowminmax": ((M - M.min(1, keepdims=True)) / (M.max(1, keepdims=True) - M.min(1, keepdims=True)), "seq", 1.0,
                      "row min–max scaled r\n(per sc row, 0–1)"),
        "colz": ((M - M.mean(0, keepdims=True)) / M.std(0, keepdims=True), "div", 2.0,
                 "column z-score of Pearson r\n(per ChIP column)"),
    }
    res = {}
    for w in which:
        X, kind, lim, lab = spec[w]
        res[w] = X
        fig = style.figure_mm(89, 80)
        ax = fig.add_axes([0.22, 0.30, 0.54, 0.56])
        if kind == "div":
            cm = LinearSegmentedColormap.from_list("div", DIV); vmin, vmax = -lim, lim
        else:
            cm = style.seq_cmap(); vmin, vmax = 0, lim
        im = ax.imshow(np.clip(X, vmin, vmax), cmap=cm, vmin=vmin, vmax=vmax)
        for i in range(6):
            for j in range(6):
                dark = (abs(X[i, j]) > 0.75 * lim) if kind == "div" else (X[i, j] > 0.6)
                ax.text(j, i, f"{X[i, j]:+.2f}" if kind == "div" else f"{X[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if dark else P.neutral.ink)
            ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=P.neutral.ink, lw=0.8))
        ax.set_xticks(range(6), [f"{t} ChIP\n({CELL[t]})" for t in tfs], rotation=90)
        ax.set_yticks(range(6), [f"{t} sc" for t in tfs]); ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cax = fig.add_axes([0.79, 0.30, 0.03, 0.56])
        cb = fig.colorbar(im, cax=cax); cb.outline.set_linewidth(0.4); cb.set_label(lab)
        style.save(fig, out, f"fig2_corr_matrix_6x6_{w}")
        rank = [int(1 + (X[i] > X[i, i]).sum()) for i in range(6)] if w != "colz" else [int(1 + (X[:, i] > X[i, i]).sum()) for i in range(6)]
        print(w, "own rank per", "row" if w != "colz" else "column", dict(zip(tfs, rank)))
        print(pd.DataFrame(X, [f"{t} sc" for t in tfs], [f"{t} ChIP" for t in tfs]).round(2).to_string())
    return res


# ---- reduced matrices: drop TFs (both the sc row and the ChIP column), then re-standardise over what remains ----
DROP = ["MAZ"]


def dropped(src, out, drop=DROP):
    P = style.use()
    allt = ["CTCF", "MAZ", "ZBTB7A", "ZIC2", "ZNF282", "ZNF777"]
    M0 = pd.read_csv(Path(src) / "corr_matrix_6x6.tsv", sep="\t", index_col=0).values.astype(float)
    keep = [i for i, t in enumerate(allt) if t not in drop]
    tfs = [allt[i] for i in keep]
    M = M0[np.ix_(keep, keep)]
    k = len(tfs)
    obs, p = perm_p(M); obs1, p1 = perm_p(M[1:, 1:])
    tag = "no" + "".join(drop)
    spec = {"r": (M, 0.2, "Pearson r (colour clipped at ±0.2)"),
            "rowz": ((M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True), 2.0, f"row z-score of Pearson r\n(per sc row, {k}×{k})"),
            "colz": ((M - M.mean(0, keepdims=True)) / M.std(0, keepdims=True), 2.0, f"column z-score of Pearson r\n(per ChIP column, {k}×{k})")}
    for w, (X, lim, lab) in spec.items():
        fig = style.figure_mm(89, 80)
        ax = fig.add_axes([0.24, 0.30, 0.52, 0.56])
        im = ax.imshow(np.clip(X, -lim, lim), cmap=LinearSegmentedColormap.from_list("div", DIV), vmin=-lim, vmax=lim)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{X[i, j]:+.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(X[i, j]) > 0.75 * lim else P.neutral.ink)
            ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=P.neutral.ink, lw=0.8))
        ax.set_xticks(range(k), [f"{t} ChIP\n({CELL[t]})" for t in tfs], rotation=90)
        ax.set_yticks(range(k), [f"{t} sc" for t in tfs]); ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cax = fig.add_axes([0.79, 0.30, 0.03, 0.56]); cb = fig.colorbar(im, cax=cax); cb.outline.set_linewidth(0.4); cb.set_label(lab)
        style.save(fig, out, f"fig2_corr_matrix_{k}x{k}_{tag}_{w}")
        rr = [int(1 + (X[i] > X[i, i]).sum()) for i in range(k)]; rc = [int(1 + (X[:, i] > X[i, i]).sum()) for i in range(k)]
        print(w, "own rank per row", dict(zip(tfs, rr)), "per column", dict(zip(tfs, rc)))
        print(pd.DataFrame(X, [f"{t} sc" for t in tfs], [f"{t} ChIP" for t in tfs]).round(3).to_string())
    print(f"{k}x{k} without {drop}: diag-off {obs:+.4f}, p = {p:.4f} ({int(round(p * math.factorial(k)))}/{math.factorial(k)});"
          f" without CTCF diag-off {obs1:+.4f}, p = {p1:.4f} ({int(round(p1 * math.factorial(k - 1)))}/{math.factorial(k - 1)}; minimum 1/{math.factorial(k - 1)})")
