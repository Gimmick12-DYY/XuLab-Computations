# Figures for "Complex slides Pearson.pptx": Pearson percentile matrices (Spearman dropped),
# each in its own dendrogram order, with the known complexes that sit close together annotated.
# Close = mean pairwise Pearson percentile >= 0.90 (same cut as the earlier Pearson deck).
import json
import numpy as np
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import fcluster
exec(open("sp_scan.py").read().split('\nif __name__ == "__main__":')[0])
S = scan()


def wang_style_order():
    # Dr. Wang's figure (pheatmap defaults): rows of the r matrix clustered by Euclidean distance, complete linkage
    from scipy.spatial.distance import pdist
    tfs = S["tfs"]; R = mat("wang18", "pearson")
    Z = linkage(pdist(R), "complete"); o = list(leaves_list(Z)); cl = fcluster(Z, 3, "maxclust")
    # dendrogram branches can be flipped freely: lay out the three clusters as in his figure
    # (MEF2A cluster first, CTCF cluster last), keeping the leaf order within each cluster
    first, last = cl[tfs.index("mef2a")], cl[tfs.index("ctcf")]
    rank = {first: 0, last: 2}
    o = sorted(o, key=lambda i: (rank.get(cl[i], 1), o.index(i)))
    Lo = [tfs[i] for i in o]
    res = S["mats"]["wang18"]["pearson"]; res["order"] = Lo
    for k, name, mem in KNOWN:
        pos = sorted(Lo.index(t) for t in mem); c = res["complex"][k]
        c.update(pos=pos, span=pos[-1] - pos[0], adj=pos[-1] - pos[0] == len(pos) - 1)


wang_style_order()
json.dump(S, open("sp_scan_spqn.json", "w"))
RAW = {"wang18": (0.65, 1.0)}      # shown as raw r on Dr. Wang's colour scale
CUT = 0.90
COLS = ["#000000", "#6A1B9A", "#00695C", "#C2185B", "#1565C0", "#E65100", "#2E7D32"]
SHORT = {"prc2": "PRC2", "nurd": "NuRD", "nurd_z": "NuRD+Z512B", "cohesin": "Cohesin–CTCF", "maz": "MAZ–ZNF143",
         "g9a": "G9a/GLP", "nfy": "NF-Y/MLX", "mef2": "MEF2", "znf512": "ZNF512", "pcgf6": "PRC1.6", "chahp": "ChAHP",
         "hp1setdb1": "HP1/SETDB1", "e2f3rbbp4": "E2F3–RBBP4", "hox": "HOX"}
LAB = {"spearman": "Percentile of TF–TF Spearman ρ", "pearson": "Percentile of TF–TF Pearson r"}


def close(mk):
    return [(k, n, m) for k, n, m in KNOWN if S["mats"][mk]["pearson"]["complex"][k]["pct"] >= CUT]


def mark(ax, pos, col, lw=1.8, d=0.0):
    rr = runs(pos)
    for a, b in rr:
        if b > a:
            ax.add_patch(Polygon([(a - .5, a - .5), (b + .5, a - .5), (b + .5, b + .5)], closed=True, fill=False,
                                 edgecolor=col, lw=lw, zorder=5))
    for i in pos:
        for j in pos:
            if i < j and not any(a <= i and j <= b for a, b in rr):
                ax.add_patch(Rectangle((j - .5 + d, i - .5 + d), 1 - 2 * d, 1 - 2 * d, fill=False, edgecolor=col,
                                       lw=lw * 0.75, zorder=6))


def panel(mk, metric, groups, out, fs=4.6, size=6.4, labels="all", legend=True, title=None):
    tfs = S["tfs"]; R = mat(mk, metric); P = R if mk in RAW else pct(R)
    Lo = S["mats"][mk][metric]["order"]; o = [tfs.index(t) for t in Lo]; Po = P[np.ix_(o, o)]; n = len(Lo)
    fig = plt.figure(figsize=(size, size * 0.93), dpi=300)
    ax = fig.add_axes([0.13, 0.06, 0.74, 0.80])
    vmin, vmax = RAW.get(mk, (0, 1))
    im = ax.imshow(Po, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xlim(-.5, n - .5); ax.set_ylim(n - .5, -.5)
    allm = set(); handles = []
    for c, (k, name, mem) in enumerate(groups):
        col = COLS[c % len(COLS)] if len(groups) > 1 else "black"
        pos = sorted(Lo.index(t) for t in mem); allm |= set(pos)
        mark(ax, pos, col, lw=2.2 if size > 5 else 1.4, d=0.22 if k == "nurd_z" else 0.0)
        handles.append(Line2D([0], [0], color=col, lw=2.2, label=SHORT[k]))
        if len(groups) > 1:                 # coloured edge ticks so members are findable on a crowded matrix
            off = 1.4 + 0.9 * c
            for p_ in pos:
                ax.plot([p_], [-off], marker="s", ms=2.2, color=col, clip_on=False, zorder=7)
                ax.plot([n - 1 + off], [p_], marker="s", ms=2.2, color=col, clip_on=False, zorder=7)
        if len(groups) == 1:
            for p_ in pos:
                ax.plot([p_], [-1.6], marker="v", ms=4.2 if size > 5 else 3, color=RED, clip_on=False, zorder=7)
                ax.plot([n + .6], [p_], marker="<", ms=4.2 if size > 5 else 3, color=RED, clip_on=False, zorder=7)
    if labels == "all":
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels([t.upper() for t in Lo], rotation=90, fontsize=fs)
        ax.set_yticklabels([t.upper() for t in Lo], fontsize=fs)
    else:                                   # members only
        pm = sorted(allm)
        ax.set_xticks(pm); ax.set_yticks(pm)
        ax.set_xticklabels([Lo[i].upper() for i in pm], rotation=90, fontsize=fs)
        ax.set_yticklabels([Lo[i].upper() for i in pm], fontsize=fs)
    for ll in (ax.get_xticklabels(), ax.get_yticklabels()):
        for lab in ll:
            if Lo.index(lab.get_text().lower()) in allm:
                lab.set_fontweight("bold"); lab.set_color(RED)
    ax.tick_params(length=1.2, width=0.3, pad=1)
    for s_ in ax.spines.values(): s_.set_linewidth(0.4)
    if title: ax.set_title(title, fontsize=9 if size > 5 else 7.5, pad=12 if len(groups) == 1 else 4 + 2.6 * len(groups))
    cax = fig.add_axes([0.905 + (0.012 * len(groups) if len(groups) > 1 else 0), 0.30, 0.018, 0.36]); cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6, width=0.3, length=2); cb.outline.set_linewidth(0.3)
    cb.set_label("Pearson r" if mk in RAW else LAB[metric], fontsize=7 if size > 5 else 6)
    if legend and len(groups) > 1:
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.885 + 0.012 * len(groups), 0.04), fontsize=6.5, frameon=False,
                   handlelength=1.5, title="Close (r pct ≥ 0.90)", title_fontsize=6.5)
    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.04); plt.close(fig)


if __name__ == "__main__":
    for mk, mt in SP_MATS:
        g = close(mk)
        panel(mk, "pearson", g, f"pc_ov_{mk}.png", title="Pearson r")
        for k, name, mem in g:
            panel(mk, "pearson", [(k, name, mem)], f"pc_bm_{mk}_{k}.png", title="Pearson r")
        print(mk, [k for k, _, _ in g])
