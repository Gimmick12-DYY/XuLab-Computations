"""Figure 3b (option 2): RBBP4 coordinated binding pairs and module edges with Hi-C loop support, and pair classes.

Top: % of RBBP4 first-step Cicero pairs (p <= 0.05, ends > 25 kb apart) and of module edges (pairs of regions inside one
higher-order coordinated binding module, > 25 kb; by module size) that sit on a Peakachu 0.7 loop (_genomics.LoopArrays);
black tick = median of the 10,000-shuffle null for pairs / for all module edges (rbbp4_shuffle_null.npz).
Bottom: one stacked strand of all pairs by the element classes of their two ends (_genomics.ElementClass).

Inputs (source_data/): rbbp4_peak_edges.tsv.gz, rbbp4_modules.tsv, peakachu_{5000,10000}.loops.0.7.bedpe.gz,
  refseq_select_tss.tsv.gz, hek293t_chromhmm18.bed.gz, rbbp4_shuffle_null.npz
Output: b_rbbp4_characterization.{pdf,svg,png}, 120 x 70 mm (print size).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import style  # noqa: E402
import _rbbp4  # noqa: E402

SHOW = [("promoter-promoter", "P–P"), ("promoter-enhancer", "P–E"), ("enhancer-enhancer", "E–E"), ("promoter-repressive", "P–R"),
        ("enhancer-repressive", "E–R"), ("repressive-repressive", "R–R")]
SEGC = {"P–P": "#2B5C9E", "P–E": "#7FA6CE", "E–E": "#E8821E", "P–R": "#8E7CC3", "E–R": "#F6CB8F", "R–R": "#55606E", "Rest": "#E3E3E3"}
LOOP = "#B83A2E"


def main(src, out):
    P = style.use(base_pt=6, small_pt=6)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.labelsize": 6.5, "axes.linewidth": 0.6})
    N = P.neutral
    D = _rbbp4.load(src)
    pairs, ploop, M = D["pairs"], D["pair_on_loop"], D["modules"]
    null = np.load(src / "rbbp4_shuffle_null.npz")
    med = np.median(null["null"], axis=0)                 # columns: pairs, module edges (> 25 kb)
    NULL = {"Pairs": med[0], "3-region": med[1], "4-region": med[1]}
    SZ = ["3", "4", "≥5"]
    rows = [("Pairs", len(pairs), int(ploop.sum()))] + [(f"{s}-region\nmodule edges", int(M.edges[M.sz == s].sum()), int(M.loop[M.sz == s].sum())) for s in SZ]
    rows = [r for r in rows if r[1] > 0]
    pc = D["pair_class"].value_counts()
    seg = [(lab, int(pc.get(k, 0))) for k, lab in SHOW]
    seg.append(("Rest", len(pairs) - sum(v for _, v in seg)))

    fig = style.figure_mm(120, 70)
    X0, XW = 0.25, 0.53
    a1 = fig.add_axes([X0, 0.53, 0.42, 0.36])
    a3 = fig.add_axes([X0, 0.27, XW, 0.065])
    for i, (lab, e, l) in enumerate(rows):
        f = 100 * l / e
        a1.bar(i, f, 0.6, color=LOOP, lw=0)
        a1.text(i, f + 1, f"{f:.0f}%", ha="center", va="bottom", fontsize=7)
        k = lab.split("\n")[0]
        if k in NULL:
            a1.plot([i - 0.38, i + 0.38], [100 * NULL[k]] * 2, color=N.ink, lw=1.0, zorder=3)
    a1.set_xticks(range(len(rows)))
    a1.set_xticklabels([f"{lab}\nn = {e:,}" for lab, e, _ in rows], fontsize=6.5, linespacing=1.05)
    a1.tick_params(axis="x", length=0)
    a1.set_ylim(0, 40)
    a1.set_yticks([0, 10, 20, 30, 40])
    a1.set_ylabel("% on a Hi-C loop", fontsize=7)
    a1.set_xlim(-0.6, len(rows) - 0.4)
    a1.yaxis.grid(True, color=N.faint, lw=0.5)
    a1.set_axisbelow(True)
    a1.legend([plt.Line2D([], [], color=N.ink, lw=1.0)], ["Shuffled expectation\n(median, 10,000 shuffles)"], loc="upper left",
              bbox_to_anchor=(1.03, 1.0), fontsize=6.5, handlelength=1.2, borderaxespad=0)
    fig.text(0.02, 0.965, "RBBP4 coordinated binding", fontsize=7.5, va="top")
    left, tot, pct = 0, len(pairs), []
    for lab, v in seg:
        f = v / tot
        a3.barh(0, f, 1.0, left=left, color=SEGC[lab], lw=0)
        if f >= 0.08:
            a3.text(left + f / 2, 0, f"{100 * f:.0f}%", ha="center", va="center", fontsize=7, color="white" if lab in ("P–P", "R–R") else N.ink)
        left += f
        pct.append(100 * f)
    a3.set_xlim(0, 1)
    a3.set_ylim(-0.5, 0.5)
    a3.set_yticks([0])
    a3.set_yticklabels(["Pair classes"], fontsize=7)
    a3.tick_params(axis="y", length=0)
    a3.set_xticks([0, .25, .5, .75, 1])
    a3.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=6.5)
    a3.set_xlabel(f"% of RBBP4 pairs >25 kb (n = {tot:,})", fontsize=7, labelpad=2)
    a3.spines["left"].set_visible(False)
    fmt = lambda x: f"{x:.0f}%" if x >= 8 else f"{x:.1f}%"
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=SEGC[lab]) for lab, _ in seg], [f"{lab} {fmt(x)}" for (lab, _), x in zip(seg, pct)],
               loc="lower left", bbox_to_anchor=(0.02, 0.07), ncol=7, fontsize=6.5, handlelength=0.8, handleheight=0.8, handletextpad=0.3,
               columnspacing=0.8, borderaxespad=0)
    fig.text(0.02, 0.012, "P promoter, E enhancer, R repressive (ChromHMM); Rest: all other combinations", fontsize=6, color=N.dark)
    print("b_char:", [(r[0].replace(chr(10), " "), r[1], round(100 * r[2] / r[1], 1)) for r in rows])
    style.save(fig, out, "b_rbbp4_characterization")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
