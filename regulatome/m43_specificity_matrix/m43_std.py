# m43 Amendment 1: standardised square specificity matrices S1-S3 (rule A).
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/figures_paletteC"); import regulome_palette as rp
P = rp.use(); INK = P.neutral.ink; PALE = P.neutral.pale
D = "/work/users/d/y/dyy12/XuLab/regulatome/m43_specificity_matrix"; O = f"{D}/out"; PL = f"{D}/plots"; CH = f"{PL}/for_chat"
TO = pd.read_csv(f"{O}/tf_order.tsv", sep="\t"); ORDER = list(TO.TF); BROAD = set(TO.TF[TO.broad])
def S1(M): A = M - M.mean(1).values[:, None]; A = A - A.mean(0).values[None, :]; return A / np.nanstd(A.values)
def S2(M): A = (M - M.mean(0)) / M.std(0, ddof=0); A = A.sub(A.mean(1), axis=0).div(A.std(1, ddof=0), axis=0); return A
def S3(M): A = M - M.mean(0); return A.rank(axis=1, method="average").sub(1).div(A.shape[1] - 1) * 100
STD = {"S1": (S1, "double-centred share found (overall SD units)", "div", (-3, 3), "double_centred"), "S2": (S2, "share found, z-scored within column then within row", "div", (-3, 3), "double_zscore"),
       "S3": (S3, "rank of this list within the row (100 = finds the largest share)", "seq", (0, 100), "row_rank")}
def summ(Q):
    d = np.diag(Q.values); off = Q.values[~np.eye(len(Q), dtype=bool)]; rk = [1 + int((Q.values[i] > Q.values[i, i]).sum() - 0) for i in range(len(Q))]
    return np.median(d), np.median(off), sum(r == 1 for r in rk), sum(r <= 5 for r in rk), [Q.index[i] for i in range(len(Q)) if rk[i] <= 5], rk
def draw(Q, title, sub, cbl, cmap, lim, path, cell=.088, fs=5.5, dpi=600, outline=False, exts=("png", "pdf", "svg")):
    n = Q.shape[0]; m = Q.shape[1]; W = 1.0 + m * cell + .3; H = .8 + n * cell + .75 + .85
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([1.0 / W, 1.6 / H, m * cell / W, n * cell / H])
    cm = P.div_cmap() if cmap == "div" else P.seq_cmap(); im = ax.imshow(Q.values, cmap=cm, vmin=lim[0], vmax=lim[1], aspect="auto", interpolation="nearest")
    if outline:
        for r, t in enumerate(Q.index):
            if t in Q.columns: j = list(Q.columns).index(t); ax.add_patch(Rectangle((j - .5, r - .5), 1, 1, fill=False, ec=INK, lw=.45))
    ax.set_xticks(range(m)); ax.set_xticklabels(Q.columns, rotation=90, fontsize=fs); ax.set_yticks(range(n)); ax.set_yticklabels(Q.index, fontsize=fs); ax.tick_params(length=0, pad=1.5); ax.tick_params(axis="y", pad=8)
    [s.set_visible(False) for s in ax.spines.values()]
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        if lab.get_text() in BROAD: lab.set_fontweight("bold")
    pos = ax.get_position(); a1 = fig.add_axes([pos.x0, pos.y1 + .004, pos.width, .05 / H]); a1.imshow([[t in BROAD for t in Q.columns]], cmap=ListedColormap([PALE, INK]), vmin=0, vmax=1, aspect="auto"); a1.axis("off")
    a2 = fig.add_axes([pos.x0 - .05 / W - .004, pos.y0, .05 / W, pos.height]); a2.imshow([[t in BROAD] for t in Q.index], cmap=ListedColormap([PALE, INK]), vmin=0, vmax=1, aspect="auto"); a2.axis("off")
    cax = fig.add_axes([(1.0 + m * cell * .15) / W, .5 / H, m * cell * .7 / W, .09 / H]); cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_label(cbl, fontsize=6.5); cb.ax.tick_params(labelsize=6, width=.5, length=2); cb.outline.set_linewidth(.4)
    fig.text(1.0 / W, (H - .2) / H, title, fontsize=8, ha="left", va="top"); fig.text(1.0 / W, (H - .42) / H, sub, fontsize=6, ha="left", va="top", color=P.neutral.dark, wrap=True)
    for e in exts: fig.savefig(f"{path}.{e}", dpi=dpi if e == "png" else None, metadata={"Creator": None} if e in ("pdf", "svg") else None)
    plt.close(fig)
SUM = []; CHAT = {}
for c in (2, 3, 4, 5):
    R0 = pd.read_csv(f"{O}/matrix_raw_ruleA_z{c}.tsv", sep="\t", index_col=0)
    for setn, keep in (("all TFs", lambda t: True), ("without broad TFs", lambda t: t not in BROAD)):
        sq = [t for t in ORDER if t in R0.index and keep(t)]; M = R0.loc[sq, sq]
        for k, (f, cbl, cmap, lim, fn) in STD.items():
            Q = f(M); tag = f"specificity_square_{fn}_ruleA_absz{c}" + ("" if setn == "all TFs" else "_without_broad_TFs"); Q.to_csv(f"{O}/{tag}.tsv", sep="\t", float_format="%.4g")
            md, mo, n1, n5, top, rk = summ(Q)
            SUM.append({"standardisation": k, "cutoff": f"|z| >= {c}", "TF set": setn, "TFs": len(sq), "median diagonal": md, "median off-diagonal": mo, "diagonal highest in row": n1, "diagonal in top 5 of row": n5, "those TFs": ",".join(top), "median rank of diagonal in row": float(np.median(rk))})
            title = {"S1": "Own list vs other lists, double-centred", "S2": "Own list vs other lists, double z-score", "S3": "Rank of each list within each knocked-down TF's row"}[k]
            sub = f"HEK293T, top-25% sgRNA cells, current rule, |z| >= {c}, {setn}: {len(sq)} TFs (rows = columns, same order). Diagonal (own list) median {md:.2f} vs off-diagonal {mo:.2f}; own list highest in its row for {n1} TFs, in the top 5 for {n5}. Bold / dark bar: broad TFs."
            draw(Q, title, sub, cbl, cmap, lim, f"{PL}/{tag}_paper")
            CHAT[(k, c, setn)] = (Q, title, sub, cbl, cmap, lim)
    print(c, flush=True)
def chat(key, name):
    Q, title, sub, cbl, cmap, lim = CHAT[key]; n = Q.shape[1]; cell = 6.6 / max(n, 30); fs = min(8, cell * 72 * .78)
    W = 1.0 + n * cell + .3; dpi = int(1600 / max(W, .8 + Q.shape[0] * cell + 1.6)); draw(Q, title, sub, cbl, cmap, lim, f"{CH}/{name}", cell=cell, fs=fs, dpi=dpi, exts=("png",))
chat(("S2", 2, "all TFs"), "1_double_zscore_absz2_all_TFs"); chat(("S2", 4, "all TFs"), "2_double_zscore_absz4_all_TFs"); chat(("S2", 4, "without broad TFs"), "3_double_zscore_absz4_without_broad_TFs"); chat(("S3", 4, "without broad TFs"), "4_row_rank_absz4_without_broad_TFs")
A = pd.read_csv(f"{O}/matrix_adjusted_ruleA_z4.tsv", sep="\t", index_col=0); NOB = [t for t in ORDER if t not in BROAD]; Qa = A.loc[[t for t in NOB if t in A.index], NOB]; v = np.abs(Qa.values); lim = float(np.percentile(v[np.isfinite(v)], 98))
cell = 6.6 / 55; W = 1.0 + 55 * cell + .3; dpi = int(1600 / W)
draw(Qa, "Own list vs other lists (adjusted share), broad TFs removed", f"current rule, |z| >= 4; {Qa.shape[0]} knocked-down TFs x {Qa.shape[1]} Regulome lists; outlined: own list (as in the earlier figure, panel b)",
     "share found minus share of the knocked-down TF's\nnon-changing genes found in this list (points)", "div", (-lim, lim), f"{CH}/5_adjusted_absz4_without_broad_TFs_panel_b", cell=cell, fs=min(8, cell * 72 * .78), dpi=dpi, outline=True, exts=("png",))
S = pd.DataFrame(SUM); S.to_csv(f"{O}/standardised_square_summary.tsv", sep="\t", index=False, float_format="%.4g")
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 70); print(S.round(2).to_string(index=False)); print("done")
