# m43 plots: specificity matrices (paper and slide versions) + diagonal summaries.
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/figures_paletteC"); import regulome_palette as rp
P = rp.use(); INK = P.neutral.ink; PALE = P.neutral.pale; MID = P.neutral.mid
D = "/work/users/d/y/dyy12/XuLab/regulatome/m43_specificity_matrix"; O = f"{D}/out"; PL = f"{D}/plots"; os.makedirs(PL, exist_ok=True)
TO = pd.read_csv(f"{O}/tf_order.tsv", sep="\t"); ORDER = list(TO.TF); BROAD = set(TO.TF[TO.broad])
RULEN = {"A": "current rule (promoter + ABC)", "G": "rule G (promoter + ABC + 10 kb + coordinated binding)"}
CB = {"raw": "share of the knocked-down TF's Perturb-seq genes\nfound in this TF's Regulome list (%)", "adjusted": "share found minus share of the knocked-down TF's\nnon-changing genes found in this list (points)",
      "columnz": "share found, standardised within each column (z)", "spearman": "Spearman correlation across genes: this TF's binding\nscore vs the knocked-down TF's |z|"}
def load(kind, rule, c, cols):
    M = pd.read_csv(f"{O}/matrix_{kind}_rule{rule}_z{c}.tsv" if kind != "spearman" else f"{O}/matrix_spearman_ruleA.tsv", sep="\t", index_col=0)
    return M.loc[[t for t in M.index if t in cols], cols]
def scale(kind, M):
    v = M.values[np.isfinite(M.values)]
    if kind == "raw": return P.seq_cmap(), 0, float(np.percentile(v, 99))
    m = float(np.percentile(np.abs(v), 98)) if kind != "columnz" else 3.0; return P.div_cmap(), -m, m
def heat(fig, ax, M, kind, fs, show_rows, lw, cax, cbfs, ypad=8):
    cm, lo, hi = scale(kind, M); nr, ncl = M.shape
    im = ax.imshow(M.values, cmap=cm, vmin=lo, vmax=hi, aspect="auto", interpolation="nearest")
    for r, t in enumerate(M.index):
        if t in M.columns: j = list(M.columns).index(t); ax.add_patch(Rectangle((j - .5, r - .5), 1, 1, fill=False, ec=INK, lw=lw))
    ax.set_xticks(range(ncl)); ax.set_xticklabels(M.columns, rotation=90, fontsize=fs); ax.set_yticks(range(nr)); ax.set_yticklabels(M.index if show_rows else [], fontsize=fs)
    ax.tick_params(length=0, pad=1.5 if fs < 8 else 3); ax.tick_params(axis="y", pad=ypad); [s.set_visible(False) for s in ax.spines.values()]
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        if lab.get_text() in BROAD: lab.set_fontweight("bold")
    cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_label(CB[kind], fontsize=cbfs); cb.ax.tick_params(labelsize=cbfs, width=.5, length=2); cb.outline.set_linewidth(.4)
    return im
def bars(fig, ax, M, w):
    pos = ax.get_position(); nr, ncl = M.shape
    a1 = fig.add_axes([pos.x0, pos.y1 + .004, pos.width, w[1]]); a1.imshow([[1 if t in BROAD else 0 for t in M.columns]], cmap=matplotlib.colors.ListedColormap([PALE, INK]), vmin=0, vmax=1, aspect="auto"); a1.axis("off")
    a2 = fig.add_axes([pos.x0 - w[0] - .004, pos.y0, w[0], pos.height]); a2.imshow([[1 if t in BROAD else 0] for t in M.index], cmap=matplotlib.colors.ListedColormap([PALE, INK]), vmin=0, vmax=1, aspect="auto"); a2.axis("off")
    return a2
def summary(M):
    d, off, rk, top = [], [], {}, []
    for r, t in enumerate(M.index):
        if t not in M.columns: continue
        row = M.loc[t]; dv = row[t]; d.append(dv); off += list(row.drop(t).values); rk[t] = int(1 + (row.drop(t) > dv).sum())
        if rk[t] <= 5: top.append(t)
    return np.median(d), np.median(off), rk, top
SUM = []
def figure(kinds, rule, c, cols, tag, title):
    Ms = [load(k, rule, c, cols) for k in kinds]; nr, ncl = Ms[0].shape; cs = .088; fs = 5.5
    W = .95 + len(kinds) * ncl * cs + (len(kinds) - 1) * .8 + .25; H = .75 + .1 + nr * cs + .62 + 1.0
    fig = plt.figure(figsize=(W, H))
    for k, (kind, M) in enumerate(zip(kinds, Ms)):
        x0 = .95 + k * (ncl * cs + .8); ax = fig.add_axes([x0 / W, (.62 + 1.0) / H, ncl * cs / W, nr * cs / H]); cax = fig.add_axes([(x0 + ncl * cs * .1) / W, .55 / H, ncl * cs * .8 / W, .09 / H])
        heat(fig, ax, M, kind, fs, True, .45, cax, 6); a2 = bars(fig, ax, M, (.05 / W, .05 / H))
        fig.text((x0 - .02) / W, (H - .62) / H, "ab"[k] if len(kinds) > 1 else "", fontsize=8, fontweight="bold", ha="right", va="bottom")
        md, mo, rk, top = summary(M)
        SUM.append({"figure": tag, "value": kind, "rule": rule, "cutoff": f"|z| >= {c}", "rows (knocked-down TFs)": nr, "columns (Regulome lists)": ncl, "median diagonal": md, "median off-diagonal": mo,
                    "TFs with own list ranked first in its row": sum(v == 1 for v in rk.values()), "TFs with own list in top 5 of its row": len(top), "those TFs": ",".join(top), "median rank of own list in its row": float(np.median(list(rk.values())))})
    fig.text(.95 / W, (H - .22) / H, title, fontsize=8, ha="left", va="top")
    fig.text(.95 / W, (H - .42) / H, f"HEK293T, top-25% sgRNA cells, {RULEN[rule]}; Perturb-seq genes |z| >= {c}; rows: {nr} knocked-down TFs with >= 10 Perturb-seq genes; columns: {ncl} Regulome lists; outlined: own list; bold / dark bar: broad TFs (list > 4,000 genes)",
             fontsize=6, ha="left", va="top", color=P.neutral.dark)
    for ext, dpi in (("png", 600), ("pdf", None), ("svg", None)): fig.savefig(f"{PL}/{tag}_paper.{ext}", dpi=dpi, metadata={"Creator": None} if ext in ("pdf", "svg") else None)
    plt.close(fig)
def slide(rule, c, cols, tag, title, kind="adjusted"):
    M = load(kind, rule, c, cols); nr, ncl = M.shape; W, H = 13.33, 7.5; cs = min(5.6 / nr, 6.4 / ncl); fs = min(cs * 72 * .85, 12)
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([1.0 / W, (H - 1.1 - nr * cs) / H, ncl * cs / W, nr * cs / H]); cax = fig.add_axes([(1.2 + ncl * cs) / W + .02, .62, .22, .035])
    heat(fig, ax, M, kind, fs, True, .8, cax, 12, ypad=12); bars(fig, ax, M, (.07 / W, .07 / H))
    md, mo, rk, top = summary(M)
    fig.text(.03, .965, title, fontsize=18, ha="left", va="top")
    fig.text(.03, .915, f"{RULEN[rule]}; |z| >= {c}; {nr} knocked-down TFs x {ncl} Regulome lists; outlined: own list; dark bar: broad TFs", fontsize=12, ha="left", va="top", color=P.neutral.dark)
    tx = (1.2 + ncl * cs) / W + .02
    fig.text(tx, .5, f"Own list (outlined)\nmedian {md:.1f}\nOther lists\nmedian {mo:.1f}\n\nOwn list in top 5\nof its row: {len(top)} of {nr}", fontsize=13, ha="left", va="top")
    if top: fig.text(tx, .2, "\n".join([", ".join(top[k:k + 4]) for k in range(0, len(top), 4)]), fontsize=12, ha="left", va="top", color=P.neutral.dark)
    fig.savefig(f"{PL}/{tag}_slide.png", dpi=300); plt.close(fig)
NOB = [t for t in ORDER if t not in BROAD]
for rule in ("A", "G"):
    for c in (2, 3, 4, 5):
        base = f"specificity_matrix_rule{rule}_absz{c}"
        figure(["raw", "adjusted"], rule, c, ORDER, base, "Does each TF's own Regulome list find its own Perturb-seq genes? (a) raw share, (b) adjusted for each list's background")
        figure(["raw", "adjusted"], rule, c, NOB, base + "_without_broad_TFs", "Same, with the 10 broad TFs removed from rows and columns")
        figure(["columnz"], rule, c, ORDER, base + "_column_standardised", "Share found, standardised within each column")
        if c in (2, 4): slide(rule, c, ORDER, base, "Own Regulome list vs other TFs' lists (adjusted share)"); slide(rule, c, NOB, base + "_without_broad_TFs", "Own list vs other lists, broad TFs removed")
        print(rule, c, flush=True)
# Spearman
for cols, tg in ((ORDER, ""), (NOB, "_without_broad_TFs")):
    M = load("spearman", "A", 0, cols); nr, ncl = M.shape; cs = .088; W = .95 + ncl * cs + .25; H = .75 + .1 + nr * cs + .62 + 1.0
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([.95 / W, 1.62 / H, ncl * cs / W, nr * cs / H]); cax = fig.add_axes([(.95 + ncl * cs * .1) / W, .55 / H, ncl * cs * .8 / W, .09 / H])
    heat(fig, ax, M, "spearman", 5.5, True, .45, cax, 6); bars(fig, ax, M, (.05 / W, .05 / H)); md, mo, rk, top = summary(M)
    SUM.append({"figure": "spearman_matrix_ruleA" + tg, "value": "spearman", "rule": "A", "cutoff": "all genes", "rows (knocked-down TFs)": nr, "columns (Regulome lists)": ncl, "median diagonal": md, "median off-diagonal": mo,
                "TFs with own list ranked first in its row": sum(v == 1 for v in rk.values()), "TFs with own list in top 5 of its row": len(top), "those TFs": ",".join(top), "median rank of own list in its row": float(np.median(list(rk.values())))})
    fig.text(.95 / W, (H - .22) / H, "Binding score vs knockdown response, Spearman correlation", fontsize=8, ha="left", va="top")
    fig.text(.95 / W, (H - .42) / H, f"HEK293T, top-25% sgRNA cells, current rule; binding score = peak signal x ABC (0 if not linked); {nr} x {ncl} TFs", fontsize=6, ha="left", va="top", color=P.neutral.dark)
    for ext, dpi in (("png", 600), ("pdf", None), ("svg", None)): fig.savefig(f"{PL}/spearman_matrix_ruleA{tg}_paper.{ext}", dpi=dpi, metadata={"Creator": None} if ext in ("pdf", "svg") else None)
    plt.close(fig)
S = pd.DataFrame(SUM); S.to_csv(f"{O}/diagonal_summary.tsv", sep="\t", index=False, float_format="%.4g")
sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
with pd.ExcelWriter(f"{O}/specificity_matrix_tables.xlsx") as w:
    S.to_excel(w, sheet_name="diagonal_summary", index=False); TO.to_excel(w, sheet_name="tf_order", index=False)
    for rule in ("A", "G"):
        for c in (2, 4):
            for k in ("raw", "adjusted"): pd.read_csv(f"{O}/matrix_{k}_rule{rule}_z{c}.tsv", sep="\t", index_col=0).to_excel(w, sheet_name=f"{k}_rule{rule}_absz{c}")
    pd.read_csv(f"{O}/matrix_spearman_ruleA.tsv", sep="\t", index_col=0).to_excel(w, sheet_name="spearman_ruleA")
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 80); print(S.round(2).to_string(index=False)); print("done")
