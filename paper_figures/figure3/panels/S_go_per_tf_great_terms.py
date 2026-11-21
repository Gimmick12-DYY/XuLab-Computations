"""Supplementary to Figure 3j: GO term x TF matrix from the GREAT-style region test (term sets as in panel j).

Rows = each TF's top 4 TF-specific and top 4 conserved terms (by q; ties in table order), union, grouped by GO-slim theme (light bands; go_term_themes.tsv);
a term is dropped when another kept term is significant in exactly the same TFs on both sides. Columns = TFs with >= 1 term,
ordered by status (TF-specific, then conserved) and number of terms; TFs without any term are listed in the note.
Status strip per group: terms / tested, none / too few elements (< 20) / no module elements (incl. TFs without Cicero pairs).
Dots: light = TF-specific (left half of the cell), teal = conserved (right half); size = -log10 q (capped at 10).

Inputs (source_data/): go_surviving_per_tf.tsv.gz, go_per_tf_controls.tsv.gz, go_term_themes.tsv, go_element_counts.tsv,
  go_null_calibration.tsv, tf_panel.tsv
Output: supplementary/S_go_per_tf_great_terms.{pdf,svg,png} (print size) and supplementary/S_go_tf_status.tsv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import style  # noqa: E402
from j_go_specific_vs_conserved_summary import load  # noqa: E402

TEAL, LT = "#2C7A6B", "#C9E3DC"
THEMES = ["Protein folding / stress", "Cell cycle / DNA / chromatin", "Gene expression / RNA", "Immune / inflammation", "Signalling", "Metabolism",
          "Development / differentiation", "Organ physiology (muscle, heart, kidney, gut)", "Neural", "Cell organization / transport / adhesion", "Other"]
STL = {3: "terms", 2: "tested, none", 1: "too few elements (< 20)", 0: "no module elements"}


def main(src, out):
    style.use(base_pt=7, small_pt=7)
    import matplotlib.pyplot as plt
    out = Path(out) / "supplementary"
    out.mkdir(parents=True, exist_ok=True)
    SP, CO = load(src)
    TH = pd.read_csv(src / "go_term_themes.tsv", sep="\t").set_index("term").theme
    for d in (SP, CO):
        d["theme"] = d.term.map(TH).fillna("Other")
    top = pd.concat([d.sort_values("q", kind="stable").groupby("tf").head(4) for d in (SP, CO)])
    terms = top.drop_duplicates("term")[["term", "name", "theme"]].copy()
    terms["o"] = terms.theme.map(THEMES.index)
    nsp = SP[SP.term.isin(terms.term)].groupby("term").tf.nunique()
    nco = CO[CO.term.isin(terms.term)].groupby("term").tf.nunique()
    terms["n"] = terms.term.map(nsp).fillna(0) + terms.term.map(nco).fillna(0)
    terms = terms.sort_values(["o", "n"], ascending=[True, False], kind="stable").reset_index(drop=True)
    seen, keep = set(), []
    for t in terms.term:
        k = (frozenset(SP.tf[SP.term == t]), frozenset(CO.tf[CO.term == t]))
        if k not in seen:
            seen.add(k)
            keep.append(t)
    terms = terms[terms.term.isin(keep)].reset_index(drop=True)

    PANEL = list(pd.read_csv(src / "tf_panel.tsv", sep="\t").tf)
    ALL = pd.read_csv(src / "go_element_counts.tsv", sep="\t").set_index("tf")
    NC_ = pd.read_csv(src / "go_null_calibration.tsv", sep="\t")
    tested = {g: set(NC_.tf[NC_.group == g]) for g in ("specific", "conserved")}
    ELC = {"specific": "specific_elements", "conserved": "conserved_elements"}

    def status(tf, g):
        d = SP if g == "specific" else CO
        if tf in set(d.tf):
            return 3
        if tf in tested[g]:
            return 2
        if tf in ALL.index and ALL[ELC[g]][tf] > 0:
            return 1
        return 0

    nterm = lambda t: SP[SP.tf == t].term.nunique() + CO[CO.tf == t].term.nunique()
    tfs = sorted(PANEL, key=lambda t: (-status(t, "specific"), -status(t, "conserved"), -nterm(t), t))
    pd.DataFrame([dict(tf=t, cicero_peaks=int(ALL.peaks[t]) if t in ALL.index else 0,
                       specific_elements=int(ALL.specific_elements[t]) if t in ALL.index else 0,
                       conserved_elements=int(ALL.conserved_elements[t]) if t in ALL.index else 0,
                       specific_status=STL[status(t, "specific")], conserved_status=STL[status(t, "conserved")],
                       specific_terms=int(SP[SP.tf == t].term.nunique()), conserved_terms=int(CO[CO.tf == t].term.nunique())) for t in tfs]
                 ).to_csv(out / "S_go_tf_status.tsv", sep="\t", index=False)
    DROP = [t for t in tfs if t not in set(SP.tf) | set(CO.tf)]
    tfs = [t for t in tfs if t not in DROP]

    def why(t):
        st_ = {status(t, "specific"), status(t, "conserved")}
        return "tested, no term" if 2 in st_ else ("< 20 elements" if 1 in st_ else "no module elements")
    grp = {}
    for t in DROP:
        grp.setdefault(why(t), []).append(t)
    note = f"Not shown (no term on either side; status of all {len(PANEL)} TFs in S_go_tf_status.tsv):\n" + "\n".join(
        {"tested, no term": "Tested, no significant term", "< 20 elements": "Fewer than 20 elements, not tested", "no module elements": "No module elements"}[k]
        + f" ({len(grp[k])}): " + ", ".join(sorted(grp[k])) for k in ("tested, no term", "< 20 elements", "no module elements") if k in grp)

    ti = {t: i for i, t in enumerate(terms.term)}
    fi = {t: i for i, t in enumerate(tfs)}
    nr, nc = len(terms), len(tfs)
    RH = 2.7
    CW = RH * nr / nc
    LM, GW, RM, BM, TM = 105, CW * nc, 45, 34, 32           # mm: term labels | grid | theme labels; footnote | TF labels
    WD, HT = LM + GW + RM, BM + RH * nr + RH * 2.6 + TM
    fig = style.figure_mm(WD, HT)
    ax = fig.add_axes([LM / WD, BM / HT, GW / WD, nr * RH / HT])
    st = fig.add_axes([LM / WD, (BM + RH * (nr + 0.6)) / HT, GW / WD, 2 * RH / HT], sharex=ax)
    sz = lambda q: 3 + 3.2 * np.minimum(-np.log10(q), 10)
    for d, kw, dx in ((CO, dict(color=TEAL, lw=0), 0.18), (SP, dict(color=LT, edgecolors=TEAL, linewidths=0.5), -0.18)):
        d = d[d.term.isin(ti)]
        ax.scatter([fi[t] + dx for t in d.tf], [ti[t] for t in d.term], s=sz(d.q), zorder=3, **kw)
    ax.set_yticks(range(nr))
    ax.set_yticklabels([n[0].upper() + n[1:] for n in terms.name], fontsize=7)
    ax.set_ylim(nr - 0.5, -0.5)
    ax.set_xticks(range(nc))
    ax.tick_params(labelbottom=False, labeltop=False, length=0)
    ax.set_xlim(-0.6, nc - 0.4)
    ax.grid(color="#EEEEEE", lw=0.3)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axvline(len(set(SP.tf)) - 0.5, color="#9E9E9E", lw=0.5)
    st.xaxis.tick_top()
    st.set_xticks(range(nc))
    st.set_xticklabels(tfs, rotation=90, fontsize=7)
    SCOL = {3: TEAL, 2: "#D9D9D9", 1: "white", 0: "white"}
    for j, t in enumerate(tfs):
        for i, g in enumerate(("specific", "conserved")):
            v = status(t, g)
            st.add_patch(plt.Rectangle((j - 0.45, i - 0.4), 0.9, 0.8, facecolor=LT if (g == "specific" and v == 3) else SCOL[v],
                                       edgecolor="#BDBDBD" if v else "none", lw=0.3))
            if v == 0:
                st.text(j, i, "–", ha="center", va="center", fontsize=5, color="#9E9E9E")
    st.set_ylim(1.6, -0.6)
    st.set_yticks([0, 1])
    st.set_yticklabels(["TF-specific elements", "Conserved elements"], fontsize=7)
    st.tick_params(length=0, labelbottom=False)
    for sp_ in st.spines.values():
        sp_.set_visible(False)
    prev = None
    for i, th in enumerate(terms.theme):
        if th != prev:
            k = (terms.theme == th).sum()
            if THEMES.index(th) % 2 == 0:
                ax.axhspan(i - 0.5, i + k - 0.5, color="#F4F4F4", zorder=0, lw=0)
            ax.text(nc - 0.1, i + k / 2 - 0.5, (th.replace(" (", "\n(").replace(" / ", " /\n", 1) if len(th) > 18 and k > 2 else th), fontsize=7,
                    va="center", ha="left", color="#4D4D4D", clip_on=False, linespacing=0.95)
            prev = th
    hs = [plt.scatter([], [], s=sz(1e-3), color=LT, edgecolors=TEAL, linewidths=0.5), plt.scatter([], [], s=sz(1e-3), color=TEAL, lw=0)]
    hs += [plt.scatter([], [], s=sz(q), color="#BDBDBD", lw=0) for q in (1e-2, 1e-5, 1e-10)]
    hs += [plt.Rectangle((0, 0), 1, 1, color="#D9D9D9", ec="#BDBDBD", lw=0.3), plt.Rectangle((0, 0), 1, 1, color="white", ec="#BDBDBD", lw=0.3)]
    fig.text(6 / WD, 2 / HT, note, fontsize=7, color="#4D4D4D", va="bottom", linespacing=1.3)
    fig.legend(hs, ["TF-specific elements (1–2 TFs)", "Conserved elements (≥3 TFs)", "q = 0.01", "1e-5", "≤1e-10", "tested, no significant term",
                    "fewer than 20 elements, not tested"], loc="lower left", bbox_to_anchor=(6 / WD, 17 / HT), ncol=4, fontsize=7, handlelength=1.0)
    print(f"S_go: {nr} terms x {nc} TFs")
    style.save(fig, out, "S_go_per_tf_great_terms", bbox_inches="tight", pad_inches=0.08)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
