"""Figure 3j: GO biological-process terms of TF-specific vs conserved module elements (GREAT-style region test, one test per TF).

TF-specific: per TF, distal elements in modules of 1-2 TFs vs the TF's other distal peaks; terms with BH q < 0.05, >= 5 regions,
  >= 3 distinct genes and no paralog-cluster artefact that also beat the TF's random-peak null (go_surviving_per_tf.tsv).
  Left: the 10 best terms by q (ties: higher fold enrichment first), at most one per TF, label "term (TF)".
Conserved: per TF, elements in modules of >= 3 TFs vs all the TF's peaks, same filters (go_per_tf_controls.tsv.gz, comparison
  "all", passes). Right: the 10 terms significant in the most TFs (ties by median fold enrichment); a term is skipped when a
  better-ranked term has the same TF set or the same core name. x = median q over those TFs; label = number of TFs.
Dot size = number of distinct genes; dashed line = q 0.05.
Upstream: source_scripts/go_great_regions.py, go_great_controls.py, go_great_null.py.

Inputs (source_data/): go_surviving_per_tf.tsv.gz, go_per_tf_controls.tsv.gz
Output: go_specific_vs_conserved_summary.{pdf,svg,png}, 200 x 82 mm (print size); go_specific_vs_conserved_summary_terms.tsv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from common import style  # noqa: E402

TEAL, LT, GREY = "#2C7A6B", "#C9E3DC", "#4D4D4D"


def short(s, n=40):
    for a, b in (("positive regulation of", "pos. reg."), ("negative regulation of", "neg. reg."), ("regulation of", "reg."), (" process", ""),
                 ("biosynthetic", "biosynth."), ("signaling pathway", "signaling"), ("type II interferon-mediated", "IFN-γ"),
                 ("type I interferon", "type I IFN"), ("long-term synaptic depression", "synaptic LTD"),
                 ("double-strand break repair via homologous recombination", "HR repair"), ("protein localization to nucleus", "nuclear protein import")):
        s = s.replace(a, b)
    s = s[0].upper() + s[1:]
    return s if len(s) <= n else s[:n - 1] + "…"


def core(n):
    n = n.lower().replace("torc1", "tor")
    for w in ("positive regulation of ", "negative regulation of ", "regulation of ", "cellular "):
        n = n.replace(w, "")
    return n.replace(" process", "").strip()


def load(src):
    SP = pd.read_csv(src / "go_surviving_per_tf.tsv.gz", sep="\t")
    SP = SP[SP.group == "specific"].copy()
    C = pd.read_csv(src / "go_per_tf_controls.tsv.gz", sep="\t")
    CO = C[(C.group == "conserved") & (C.comparison == "all") & C.passes].copy()
    return SP, CO


def main(src, out):
    style.use(base_pt=6, small_pt=6)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter
    plt.rcParams.update({"axes.labelsize": 6.5, "axes.linewidth": 0.6})
    SP, CO = load(src)
    L = SP.sort_values(["q", "fold"], ascending=[True, False], kind="stable").drop_duplicates("tf").head(10).reset_index(drop=True)
    R = CO.groupby(["term", "name"]).agg(n_tfs=("tf", "nunique"), tfset=("tf", lambda x: frozenset(x)), fold=("fold", "median"), q=("q", "median"),
                                          fg_regions=("fg_regions", "median"), genes=("distinct_genes", "median")).reset_index()
    R = R.sort_values(["n_tfs", "fold"], ascending=[False, False])
    keep, seen_sets, seen_core = [], set(), set()
    for _, r in R.iterrows():
        if r.tfset in seen_sets or core(r["name"]) in seen_core:
            continue
        keep.append(r)
        seen_sets.add(r.tfset)
        seen_core.add(core(r["name"]))
        if len(keep) == 10:
            break
    R = pd.DataFrame(keep).reset_index(drop=True)
    pd.concat([L.assign(side="TF-specific")[["side", "tf", "term", "name", "q", "fold", "distinct_genes"]],
               R.assign(side="conserved", tf=[",".join(sorted(s)) for s in R.tfset])[["side", "tf", "term", "name", "q", "fold", "genes", "n_tfs"]]]
              ).to_csv(Path(out) / "go_specific_vs_conserved_summary_terms.tsv", sep="\t", index=False)

    fig = style.figure_mm(200, 82)
    a1 = fig.add_axes([0.255, 0.28, 0.2, 0.57])
    a2 = fig.add_axes([0.745, 0.28, 0.2, 0.57], sharex=a1)
    sz = lambda n: 6 + 1.6 * np.asarray(n, float)
    y = np.arange(10)
    a1.scatter(-np.log10(L.q), y[:len(L)], s=sz(L.distinct_genes), color=LT, edgecolors=TEAL, linewidths=0.7, zorder=3)
    a1.set_yticks(y[:len(L)])
    a1.set_yticklabels([f"{short(n, 60)}  ({t})" for n, t in zip(L.name, L.tf)], fontsize=6.5)
    a1.set_title("TF-specific module elements (1–2 TFs)", fontsize=7, pad=5)
    a2.scatter(-np.log10(R.q), y[:len(R)], s=sz(R.genes), color=TEAL, lw=0, zorder=3)
    for i, (f, n, g) in enumerate(zip(-np.log10(R.q), R.n_tfs, R.genes)):
        a2.annotate(f"{n} TF" + ("s" if n > 1 else ""), (f, i), xytext=(np.sqrt(sz(g)) / 2 + 2.5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=6, color=GREY)
    a2.set_yticks(y[:len(R)])
    a2.set_yticklabels([short(n, 60) for n in R.name], fontsize=6.5)
    a2.set_title("Conserved module elements (≥3 TFs)", fontsize=7, pad=5)
    a1.set_xlim(0, max(-np.log10(L.q.min()), -np.log10(R.q.min())) * 1.3)
    tk = [t for t in (2, 3, 4, 5, 6, 8, 10, 12, 14, 16) if t <= a1.get_xlim()[1]]
    for ax in (a1, a2):
        ax.set_ylim(9.5, -0.6)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color="#EEEEEE", lw=0.5)
        ax.set_axisbelow(True)
        ax.axvline(-np.log10(0.05), color="#9E9E9E", lw=0.6, ls=(0, (3, 2)), zorder=0)
        ax.set_xlabel("q-value (BH)")
        ax.xaxis.set_major_locator(FixedLocator(tk))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"1e-{int(v)}"))
        ax.text(-np.log10(0.05) + 0.08, -0.55, "q = 0.05", ha="left", va="top", fontsize=5.5, color="#7F7F7F")
    hs = [plt.scatter([], [], s=sz(n), color="#BDBDBD", lw=0) for n in (5, 20, 50)]
    fig.legend(hs, ["5", "20", "50 genes"], title="Genes", loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=6,
               title_fontsize=6, handletextpad=0.3, columnspacing=1.5)
    style.save(fig, out, "go_specific_vs_conserved_summary")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
