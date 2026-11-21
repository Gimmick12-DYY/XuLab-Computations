"""Supplementary to Figure 3j (earlier version of the GO comparison): pooled GO biological-process enrichment of module promoter genes,
TF-specific (modules of 1-2 TFs) vs conserved (modules of >= 3 TFs), each group with its own top 10 terms by pooled p (hypergeometric,
background = all network promoter genes; source_scripts/go_per_tf.py). A term is skipped when its background gene set overlaps a
better-ranked term with Jaccard > 0.6 (source_scripts/go_term_jaccard.py). x = -log10 p (shared axis); dashed line = p of the last
term with BH q < 0.05 (0.05 / m when none); teal = q < 0.05, hollow grey = q >= 0.05; size = genes in the term.

Inputs (source_data/): go_pooled_{specific,conserved}.tsv.gz, go_summary_{specific,conserved}.tsv.gz, go_top_term_jaccard.tsv.gz
Output: supplementary/S_j_gene_set_dotplot.{pdf,svg,png}, 183 x 66 mm (print size), and S_j_gene_set_dotplot.tsv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from common import style  # noqa: E402

TEAL, GREY = "#2C7A6B", "#9E9E9E"
SHORT = {"dolichol-linked oligosaccharide biosynthetic process": "dolichol oligosaccharide biosynthesis", "regulation of monoatomic ion transmembrane transport": "reg. of ion transmembrane transport",
         "monoatomic ion transmembrane transport": "ion transmembrane transport", "monoatomic cation transmembrane transport": "cation transmembrane transport",
         "regulation of mononuclear cell migration": "reg. of mononuclear cell migration", "positive regulation of cell cycle phase transition": "pos. reg. of cell cycle phase transition",
         "positive regulation of triglyceride biosynthetic process": "pos. reg. of triglyceride biosynthesis", "positive regulation of TORC1 signaling": "pos. reg. of TORC1 signaling",
         "positive regulation of catabolic process": "pos. reg. of catabolic process", "regulation of cellular response to stress": "reg. of response to stress", "endoplasmic reticulum to Golgi vesicle-mediated transport": "ER to Golgi vesicle transport", "negative regulation of protein localization to plasma membrane": "neg. reg. of protein localization to PM", "inner ear receptor cell stereocilium organization": "stereocilium organization (inner ear)", "entrainment of circadian clock by photoperiod": "circadian entrainment by photoperiod"}


def cap(s):
    t = SHORT.get(s, s)
    return t[0].upper() + t[1:]


def top(src, grp, JAC, k=10):
    pq = pd.read_csv(src / f"go_pooled_{grp}.tsv.gz", sep="\t", index_col=0).sort_values("p", kind="stable")
    sm = pd.read_csv(src / f"go_summary_{grp}.tsv.gz", sep="\t", index_col=0)
    keep = []
    for t, r in pq.head(40).iterrows():
        if r.k == 0 or any(JAC.get(frozenset((t, u)), 0) > 0.6 for u in keep):
            continue
        keep.append(t)
        if len(keep) == k:
            break
    sigp = pq.p[pq.q < 0.05]
    T = pq.loc[keep].copy()
    T["thr"] = float(sigp.max()) if len(sigp) else 0.05 / len(pq)
    T["n_tfs_sig"] = sm.n_tfs_sig.reindex(keep).fillna(0).astype(int).values
    T["n_tfs_tested"] = int(sm.n_tfs_tested.iloc[0])
    return T


def main(src, out):
    style.use(base_pt=6, small_pt=6)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.labelsize": 6.5, "axes.linewidth": 0.6})
    out = Path(out) / "supplementary"
    out.mkdir(parents=True, exist_ok=True)
    JJ = pd.read_csv(src / "go_top_term_jaccard.tsv.gz", sep="\t")
    JAC = {frozenset((a, b)): j for a, b, j in zip(JJ.term_a, JJ.term_b, JJ.jaccard)}
    SP, CO = top(src, "specific", JAC), top(src, "conserved", JAC)
    sz = lambda n: 4 + 0.35 * n
    xmax = np.ceil(max(-np.log10(SP.p.min()), -np.log10(CO.p.min()), -np.log10(min(SP.thr.iloc[0], CO.thr.iloc[0]))) + 0.5)
    fig = style.figure_mm(183, 66)
    W = 0.22
    axs = [fig.add_axes([0.245, 0.24, W, 0.62]), fig.add_axes([0.73, 0.24, W, 0.62])]
    for ax, T, ttl in ((axs[0], SP, "TF-specific module genes"), (axs[1], CO, "Conserved module genes")):
        y = np.arange(len(T))
        x = -np.log10(T.p.values)
        sig = (T.q < 0.05).values
        ax.scatter(x[~sig], y[~sig], s=sz(T.k.values[~sig]), facecolors="white", edgecolors=GREY, linewidths=0.6, zorder=3)
        ax.scatter(x[sig], y[sig], s=sz(T.k.values[sig]), color=TEAL, lw=0, zorder=3)
        t = -np.log10(T.thr.iloc[0])
        ax.axvline(t, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=0)
        ax.text(t + 0.08, len(T) - 0.55, "q = 0.05", ha="left", va="bottom", fontsize=5, color="#4D4D4D")
        ax.set_yticks(y)
        ax.set_yticklabels([cap(s_) for s_ in T.name], fontsize=6)
        ax.set_ylim(len(T) - 0.5, -0.6)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, xmax)
        ax.set_xlabel("−log10 p")
        ax.xaxis.grid(True, color="#EEEEEE", lw=0.5)
        ax.set_axisbelow(True)
        ax.set_title(f"{ttl}\n({'1–2 TFs' if 'specific' in ttl else '≥3 TFs'}, n = {int(T.n_list.iloc[0]):,})", fontsize=6.5, pad=9)
    hs = [plt.scatter([], [], s=sz(n), color="#4D4D4D", lw=0) for n in (20, 50, 100)]
    hs += [plt.scatter([], [], s=sz(40), color=TEAL, lw=0), plt.scatter([], [], s=sz(40), facecolors="white", edgecolors=GREY, linewidths=0.6)]
    fig.legend(hs, ["20", "50", "100 genes", "q < 0.05", "q ≥ 0.05"], loc="lower center", ncol=5, fontsize=5.5, bbox_to_anchor=(0.6, 0.0),
               handletextpad=0.2, columnspacing=1.2)
    pd.concat([SP.assign(group="TF-specific (1-2 TFs)"), CO.assign(group="conserved (>=3 TFs)")])[
        ["group", "name", "k", "n_list", "K", "fold", "p", "q", "n_tfs_sig", "n_tfs_tested"]].to_csv(out / "S_j_gene_set_dotplot.tsv", sep="\t")
    style.save(fig, out, "S_j_gene_set_dotplot")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
