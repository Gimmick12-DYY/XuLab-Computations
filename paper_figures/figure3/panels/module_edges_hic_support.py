"""Figure 3d: Hi-C loop support of higher-order coordinated binding module edges, one TF per row (rows ordered by number of
modules with an edge > 25 kb; TFs with >= 20 such modules).

Left: all module edges > 25 kb of each TF in 1 Mb bins of the edge midpoint (columns clustered by Jaccard distance on edge
presence, average linkage, branches rotated so bins used by higher rows come first; tree not drawn). Cell colour = fraction of
that TF's edges in that bin on a Peakachu 0.7 loop (same classes as the bars); white = no edge. Top strip = TFs per bin.
Middle: modules with >= 1 edge > 25 kb, stacked by the fraction of their edges on a loop (bars capped at 2,500; percent =
modules with any loop edge). Right: observed % of module edges on a loop vs a per-edge shuffle null (10,000 shuffles; box =
IQR, line = 95 % interval); filled = BH q < 0.05 (TFs with >= 100 edges).

Inputs (source_data/): d_module_edge_bins_1000kb.tsv.gz, d_module_loop_coverage.tsv.gz, d_module_edge_null.npz
Output: module_edges_hic_support.{pdf,svg,png}, 183 x 95 mm (print size). Heatmap cells are an embedded raster image.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from common import style  # noqa: E402

LABS = ["no loop edge", ">0–25%", "25–50%", "50–75%", "75–<100%", "100% (all edges)"]
CAP = 2500          # bar axis cap (modules)


def bh(p):
    """Benjamini-Hochberg adjusted p values (same as statsmodels multipletests(method="fdr_bh"))."""
    p = np.asarray(p, float)
    o = np.argsort(p)[::-1]
    q = np.minimum.accumulate(p[o] * len(p) / np.arange(len(p), 0, -1))
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def strata(n_loop, frac):
    return np.select([n_loop == 0, frac <= .25, frac <= .5, frac <= .75, frac < 1], [0, 1, 2, 3, 4], 5)


def main(src, out):
    P = style.use()
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    plt.rcParams["axes.linewidth"] = 0.6
    N, cols = P.neutral, P.loop_strata

    # bars: modules per TF by loop coverage
    C = pd.read_csv(src / "d_module_loop_coverage.tsv.gz", sep="\t")
    C["stratum"] = [LABS[i] for i in strata(C.loop_edges.values, C.coverage.values)]
    Pt = C.groupby(["tf", "stratum"]).size().unstack(fill_value=0).reindex(columns=LABS, fill_value=0)
    Pt = Pt[Pt.sum(1) >= 20]
    # rows by number of modules (ascending = bottom to top); ties broken alphabetically
    Pt = Pt.loc[sorted(Pt.index, key=lambda t: (Pt.sum(1)[t], t))]
    rows = list(Pt.index)

    # per-edge shuffle null
    Z_ = np.load(src / "d_module_edge_null.npz")
    tfs_null = [t for t in rows if f"null_{t}" in Z_.files]
    E = pd.DataFrame({"edge_obs": [float(Z_[f"obs_{t}"]) for t in tfs_null]}, index=tfs_null)
    nul = {t: Z_[f"null_{t}"] for t in tfs_null}
    E["edge_p"] = [(1 + (nul[t] >= E.edge_obs[t]).sum()) / (1 + len(nul[t])) for t in E.index]
    E["q"] = bh(E.edge_p)
    E.to_csv(Path(out) / "module_edges_hic_support_null.tsv", sep="\t")

    # heatmap: TF x 1 Mb bin, coloured by loop stratum of that TF's edges in the bin
    BD = pd.read_csv(src / "d_module_edge_bins_1000kb.tsv.gz", sep="\t")
    ri = {t: i for i, t in enumerate(rows)}
    BD = BD[BD.tf.isin(ri)]
    CHO = {f"chr{i}": i for i in range(1, 23)} | {"chrX": 23, "chrY": 24}
    bins = BD[["chrom", "bin"]].drop_duplicates()
    bins = bins.assign(o=bins.chrom.map(CHO)).sort_values(["o", "bin"]).reset_index(drop=True)
    bi = {(c, x): j for j, (c, x) in enumerate(zip(bins.chrom, bins.bin))}
    S = np.full((len(rows), len(bins)), -1, int)
    for t, c, x, v in zip(BD.tf, BD.chrom, BD.bin, strata(BD.n_loop.values, (BD.n_loop / BD.n_edges).values)):
        S[ri[t], bi[(c, x)]] = v
    Z = linkage(pdist((S >= 0).T, "jaccard"), "average")
    nl = S.shape[1]
    members = {i: [i] for i in range(nl)}
    toprow = -(len(rows) - 1 - ((S >= 0)[::-1].argmax(0))).astype(float)
    for k in range(len(Z)):                      # rotate branches: bins used by higher rows first
        l, r = int(Z[k, 0]), int(Z[k, 1])
        if toprow[members[r]].mean() < toprow[members[l]].mean():
            Z[k, 0], Z[k, 1] = r, l
        members[nl + k] = members[int(Z[k, 0])] + members[int(Z[k, 1])]
    S = S[:, leaves_list(Z)]
    ntfb = (S >= 0).sum(0)

    fig = style.figure_mm(183, 95)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.1, 0.9], height_ratios=[0.07, 1], wspace=0.05, hspace=0.03,
                          left=0.065, right=0.995, top=0.985, bottom=0.085)
    hm = fig.add_subplot(gs[1, 0])
    top = fig.add_subplot(gs[0, 0], sharex=hm)
    ax = fig.add_subplot(gs[1, 1], sharey=hm)
    b = fig.add_subplot(gs[1, 2], sharey=hm)

    PAL = np.array([to_rgb(c) for c in cols] + [(1, 1, 1)])
    SUB, FILL = 25, 18                           # each row = 25 sub-pixels, 18 coloured (= bar height 0.72)
    img = np.ones((len(rows) * SUB, S.shape[1], 3))
    for i in range(len(rows)):
        y0 = i * SUB + (SUB - FILL) // 2
        img[y0:y0 + FILL] = PAL[S[i]][None, :, :]
    hm.imshow(img, aspect="auto", origin="lower", interpolation="nearest", extent=(-0.5, S.shape[1] - 0.5, -0.5, len(rows) - 0.5))
    hm.set_xlim(-0.5, S.shape[1] - 0.5)
    hm.set_xticks([])
    ne, nl_ = int(BD.n_edges.sum()), int(BD.n_loop.sum())
    hm.set_xlabel(f"Module edges >25 kb, 1 Mb bins (n = {S.shape[1]:,}, clustered)\n{ne:,} edges, {100 * nl_ / ne:.0f}% on a loop")
    hm.tick_params(axis="y", length=0, pad=1.5)
    for sp in ("top", "right", "bottom"):
        hm.spines[sp].set_visible(False)
    top.bar(range(len(ntfb)), ntfb, width=1.0, color=N.mid, lw=0)
    top.set_xlim(-0.5, len(ntfb) - 0.5)
    top.set_ylim(0, ntfb.max())
    top.set_yticks([0, int(ntfb.max())])
    top.tick_params(labelbottom=False, bottom=False, labelsize=5, length=1.5, pad=1)
    top.set_ylabel("TFs per\nbin", fontsize=5, rotation=0, ha="right", va="center", labelpad=2)
    for sp in ("top", "right", "bottom"):
        top.spines[sp].set_visible(False)

    y = np.arange(len(Pt))
    left = np.zeros(len(Pt))
    scale = np.minimum(1.0, CAP / Pt.sum(1).values)
    for l, c in zip(LABS, cols):
        v = Pt[l].values.astype(float) * scale
        ax.barh(y, v, left=left, color=c, height=0.72, label=("No loop edge" if l == LABS[0] else l), edgecolor="#4D4D4D", linewidth=0.3)
        left += v
    tot = Pt.sum(1).values
    sup = tot - Pt[LABS[0]].values
    for i, (t, s_) in enumerate(zip(tot, sup)):
        ax.text(CAP * 1.01 if t > CAP else t + 25, i, (f"// {t:,} ({100 * s_ / t:.0f}%)" if t > CAP else f"{100 * s_ / t:.0f}%"),
                va="center", fontsize=5, color=N.dark)
    ax.set_xlim(0, CAP * 1.38)
    hm.set_yticks(y)
    hm.set_yticklabels(Pt.index, fontsize=5)
    hm.set_ylim(-0.7, len(Pt) - 0.3)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xlabel("Modules with ≥1 edge >25 kb")
    ax.legend(title="Module edges on a loop\n(per module in bars, per bin in heatmap)", loc="lower right", fontsize=6, title_fontsize=6,
              handlelength=1.0, handleheight=0.8, labelspacing=0.25, borderaxespad=0.2, alignment="left")

    for i, tf in enumerate(Pt.index):
        if tf not in E.index:
            b.text(13.0, i, "n/a", va="center", fontsize=5, color=N.mid)
            continue
        v = nul[tf] * 100
        lo, q1, md, q3, hi = np.percentile(v, [2.5, 25, 50, 75, 97.5])
        b.plot([lo, hi], [i, i], color=N.mid, lw=0.6, solid_capstyle="butt")
        b.add_patch(plt.Rectangle((q1, i - 0.28), q3 - q1, 0.56, color=N.light, lw=0))
        b.plot([md, md], [i - 0.3, i + 0.3], color=N.dark, lw=0.6)
        b.plot([md, E.edge_obs[tf] * 100], [i, i], color=N.light, lw=0.5, zorder=1)
        sig = E.q[tf] < 0.05
        b.scatter([E.edge_obs[tf] * 100], [i], s=5, color=N.ink if sig else "white", edgecolors=N.ink, linewidths=0 if sig else 0.5, zorder=3)
    b.set_xlim(12, max(E.edge_obs.max() * 100, max(np.percentile(nul[t] * 100, 97.5) for t in E.index)) + 1)
    b.set_xlabel("Module edges >25 kb on a loop (%)")
    b.xaxis.grid(True, color=N.faint, lw=0.4)
    b.set_axisbelow(True)
    b.tick_params(axis="y", left=False, labelleft=False)
    b.spines["left"].set_visible(False)
    h = [plt.Line2D([], [], ls="", marker="o", ms=2.6, color=N.ink), plt.Line2D([], [], ls="", marker="o", ms=2.6, mfc="white", mec=N.ink, mew=0.5),
         plt.Rectangle((0, 0), 1, 1, color=N.light)]
    b.legend(h, ["Observed, q < 0.05", "Observed, n.s.", "Shuffle null"], loc="lower right", fontsize=6, handlelength=1.0, labelspacing=0.25,
             borderaxespad=0.2, facecolor="white", frameon=True, framealpha=1, edgecolor="none")
    print(f"d: {len(rows)} rows, {S.shape[1]:,} bins, {ne:,} edges; {int((E.q < 0.05).sum())} of {len(E)} TFs q < 0.05")
    style.save(fig, out, "module_edges_hic_support")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
