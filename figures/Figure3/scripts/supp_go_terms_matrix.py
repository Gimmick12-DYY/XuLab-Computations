"""Supplementary: GO terms x TFs matrix from the GREAT-style region test (same term sets as j_go_specific_vs_conserved_summary.py:
TF-specific = distal-vs-distal passing terms, conserved = all-element passing terms; all TFs beat their random-peak null).
Rows = each TF's top 4 TF-specific and top 4 conserved terms (by q), union, grouped by GO-slim theme (light bands); near-duplicates
merged: a term is dropped when another kept term is significant in exactly the same TFs (same group). Columns = the TFs with >= 1 term (TFs without any term are
listed in a note and in the status sheet, not drawn), ordered by status
(TF-specific, then conserved) and number of terms; a status strip on top marks, per group: terms / tested, none / too few elements
(< 20, minimum for a test) / no module elements (incl. the 11 TFs without Cicero pairs). Status table: checks/go_great_regions/tf_status_79.tsv.
Cell: light hollow dot = TF-specific term, teal dot = conserved term, size = -log10 q (capped at 10). Full lists: go_per_tf_great_terms.xlsx.
  python Figure3/scripts/supp_go_terms_matrix.py   (Mac)"""
import re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = Path(__file__).resolve().parents[1]; G = H / "checks/go_great_regions"; MM = 1 / 25.4
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 7, "svg.fonttype": "none", "axes.linewidth": 0.5, "legend.frameon": False})
TEAL, LT = "#2C7A6B", "#C9E3DC"
TH = pd.concat([pd.read_csv(G / f, sep="\t")[["term", "theme"]] for f in ("surviving_per_TF_themes.tsv", "per_TF_q005_themes.tsv")]).drop_duplicates("term").set_index("term").theme
SP = pd.read_csv(G / "surviving_per_TF.tsv", sep="\t"); SP = SP[SP.group == "specific"].copy()
C = pd.read_csv(G / "per_TF_controls.tsv.gz", sep="\t"); CO = C[(C.group == "conserved") & (C.comparison == "all") & C.passes].copy()
for d in (SP, CO): d["theme"] = d.term.map(TH).fillna("Other")
THEMES = ["Protein folding / stress", "Cell cycle / DNA / chromatin", "Gene expression / RNA", "Immune / inflammation", "Signalling", "Metabolism",
          "Development / differentiation", "Organ physiology (muscle, heart, kidney, gut)", "Neural", "Cell organization / transport / adhesion", "Other"]
top = pd.concat([d.sort_values("q").groupby("tf").head(4) for d in (SP, CO)])
terms = top.drop_duplicates("term")[["term", "name", "theme"]]; terms["o"] = terms.theme.map(THEMES.index)
nsp = SP[SP.term.isin(terms.term)].groupby("term").tf.nunique(); nco = CO[CO.term.isin(terms.term)].groupby("term").tf.nunique()
terms["n"] = terms.term.map(nsp).fillna(0) + terms.term.map(nco).fillna(0); terms = terms.sort_values(["o", "n"], ascending=[True, False]).reset_index(drop=True)
pat = lambda t: (frozenset(SP.tf[SP.term == t]), frozenset(CO.tf[CO.term == t])); seen, keep = set(), []
for t in terms.term:
    k = pat(t)
    if k not in seen: seen.add(k); keep.append(t)
ndup = len(terms) - len(keep); terms = terms[terms.term.isin(keep)].reset_index(drop=True)
PANEL = ["ADNP", "AKAP8L", "CHAMP1", "CTCF", "DDIT3", "DNMT1", "DR1", "E2F3", "E4F1", "EPAS1", "ETS2", "EZH2", "FLYWCH1", "FOXC1", "FOXJ3", "FOXK1", "GLYR1",
         "HES1", "HOXA10", "HOXA5", "HOXB6", "HOXB8", "HOXB9", "HOXC4", "HOXC9", "HOXD10", "HP1", "MAZ", "MBD3", "MEF2A", "MEF2D", "MLX", "MSX2", "MTF2", "NEUROD4",
         "NFYA", "NME2", "NR1D2", "NRF1", "PA2G4", "PCGF6", "POGK", "RBBP4", "RBBP7", "RBPJ", "RELA", "SAFB", "SETDB1", "SMARCA1", "SOX12", "SOX4", "SRCAP", "SREBF2",
         "STAG2", "TBX3", "THAP6", "TRAFD1", "WIZ", "XPA", "ZBTB40", "ZBTB43", "ZBTB7A", "ZIC2", "ZNF143", "ZNF282", "ZNF367", "ZNF503", "ZNF507", "ZNF512",
         "ZNF512B", "ZNF581", "ZNF606", "ZNF644", "ZNF703", "ZNF746", "ZNF768", "ZNF777", "ZNF79", "ZUFSP"]   # 79 panel TFs (cobinding/work/ourrun_k50 + RBBP4)
ALL = pd.read_csv(G / "distal_fractions.tsv", sep="\t").set_index("tf")              # 68 TFs with Cicero peaks; element counts
NC_ = pd.read_csv(G / "null_calibration.tsv", sep="\t"); tested = {g: set(NC_.tf[NC_.group == g]) for g in ("specific", "conserved")}
ELC = {"specific": "specific_elements", "conserved": "conserved_elements"}
def status(tf, g):
    d = SP if g == "specific" else CO
    if tf in set(d.tf): return 3                                                    # >= 1 term passing the controls
    if tf in tested[g]: return 2                                                    # tested, no term
    if tf in ALL.index and ALL[ELC[g]][tf] > 0: return 1                            # 1-19 elements: too few to test
    return 0                                                                        # no module elements of this class (or no Cicero pairs)
nterm = lambda t: SP[SP.tf == t].term.nunique() + CO[CO.tf == t].term.nunique()
tfs = sorted(PANEL, key=lambda t: (-status(t, "specific"), -status(t, "conserved"), -nterm(t), t))
STL = {3: "terms", 2: "tested, none", 1: "too few elements (< 20)", 0: "no module elements"}
ST = pd.DataFrame([dict(tf=t, cicero_peaks=int(ALL.peaks[t]) if t in ALL.index else 0, specific_elements=int(ALL.specific_elements[t]) if t in ALL.index else 0,
                        conserved_elements=int(ALL.conserved_elements[t]) if t in ALL.index else 0, specific_status=STL[status(t, "specific")],
                        conserved_status=STL[status(t, "conserved")], specific_terms=int(SP[SP.tf == t].term.nunique()), conserved_terms=int(CO[CO.tf == t].term.nunique())) for t in tfs])
ST.to_csv(G / "tf_status_79.tsv", sep="\t", index=False)
DROP = [t for t in tfs if t not in set(SP.tf) | set(CO.tf)]                       # no term on either side: listed in the status table, not drawn (user 2026-11-18)
tfs = [t for t in tfs if t not in DROP]
def why(t):
    st_ = {status(t, "specific"), status(t, "conserved")}
    return "tested, no term" if 2 in st_ else ("< 20 elements" if 1 in st_ else "no module elements")
grp = {}
for t in DROP: grp.setdefault(why(t), []).append(t)
note = "Not shown (no term on either side; status of all 79 TFs in go_per_tf_great_terms.xlsx, sheet TF_status_79):\n" + "\n".join(
    {"tested, no term": "Tested, no significant term", "< 20 elements": "Fewer than 20 elements, not tested", "no module elements": "No module elements"}[k] + f" ({len(grp[k])}): " + ", ".join(sorted(grp[k])) for k in ("tested, no term", "< 20 elements", "no module elements") if k in grp)
print({k: len(v) for k, v in grp.items()})
print(pd.crosstab(ST.specific_status, "TF-specific").join(pd.crosstab(ST.conserved_status, "conserved"), how="outer").fillna(0).astype(int))
ti = {t: i for i, t in enumerate(terms.term)}; fi = {t: i for i, t in enumerate(tfs)}
nr, nc = len(terms), len(tfs)
RH = 2.7; CW = RH * nr / nc                                                       # square grid: width = height
LM, GW, RM, BM, TM = 105, CW * nc, 45, 34, 32                                       # mm: term labels | grid | theme labels; footnote | TF labels
WD, HT = LM + GW + RM, BM + RH * nr + RH * 2.6 + TM
fig = plt.figure(figsize=(WD * MM, HT * MM)); ax = fig.add_axes([LM / WD, BM / HT, GW / WD, nr * RH / HT])
st = fig.add_axes([LM / WD, (BM + RH * (nr + 0.6)) / HT, GW / WD, 2 * RH / HT], sharex=ax)
sz = lambda q: 3 + 3.2 * np.minimum(-np.log10(q), 10)
for d, kw, dx in ((CO, dict(color=TEAL, lw=0), 0.18), (SP, dict(color=LT, edgecolors=TEAL, linewidths=0.5), -0.18)):
    d = d[d.term.isin(ti)]; ax.scatter([fi[t] + dx for t in d.tf], [ti[t] for t in d.term], s=sz(d.q), zorder=3, **kw)
def short(s, n=None):
    return s[0].upper() + s[1:]                                                      # full names (user 2026-11-18)
ax.set_yticks(range(nr)); ax.set_yticklabels([short(n) for n in terms.name], fontsize=7); ax.set_ylim(nr - 0.5, -0.5)
ax.set_xticks(range(nc)); ax.tick_params(labelbottom=False, labeltop=False); ax.set_xlim(-0.6, nc - 0.4)
ax.tick_params(length=0); ax.grid(color="#EEEEEE", lw=0.3); ax.set_axisbelow(True)
for sp in ax.spines.values(): sp.set_visible(False)
ax.axvline(len(set(SP.tf)) - 0.5, color="#9E9E9E", lw=0.5)
st.xaxis.tick_top(); st.set_xticks(range(nc)); st.set_xticklabels(tfs, rotation=90, fontsize=7)
SCOL = {3: TEAL, 2: "#D9D9D9", 1: "white", 0: "white"}
for j, t in enumerate(tfs):
    for i, (g, d) in enumerate((("specific", SP), ("conserved", CO))):
        v = status(t, g); st.add_patch(plt.Rectangle((j - 0.45, i - 0.4), 0.9, 0.8, facecolor=LT if (g == "specific" and v == 3) else SCOL[v], edgecolor="#BDBDBD" if v else "none", lw=0.3))
        if v == 0: st.text(j, i, "–", ha="center", va="center", fontsize=5, color="#9E9E9E")
st.set_ylim(1.6, -0.6); st.set_yticks([0, 1]); st.set_yticklabels(["TF-specific elements", "Conserved elements"], fontsize=7); st.tick_params(length=0, labelbottom=False)
for sp_ in st.spines.values(): sp_.set_visible(False)
prev = None
for i, th in enumerate(terms.theme):
    if th != prev:
        k = (terms.theme == th).sum()
        if THEMES.index(th) % 2 == 0: ax.axhspan(i - 0.5, i + k - 0.5, color="#F4F4F4", zorder=0, lw=0)
        ax.text(nc - 0.1, i + k / 2 - 0.5, (th.replace(" (", "\n(").replace(" / ", " /\n", 1) if len(th) > 18 and k > 2 else th), fontsize=7, va="center", ha="left", color="#4D4D4D", clip_on=False, linespacing=0.95)
        prev = th
hs = [plt.scatter([], [], s=sz(1e-3), color=LT, edgecolors=TEAL, linewidths=0.5), plt.scatter([], [], s=sz(1e-3), color=TEAL, lw=0)] + [plt.scatter([], [], s=sz(q), color="#BDBDBD", lw=0) for q in (1e-2, 1e-5, 1e-10)]
hs += [plt.Rectangle((0, 0), 1, 1, color="#D9D9D9", ec="#BDBDBD", lw=0.3), plt.Rectangle((0, 0), 1, 1, color="white", ec="#BDBDBD", lw=0.3)]
fig.text(6 / WD, 2 / HT, note, fontsize=7, color="#4D4D4D", va="bottom", linespacing=1.3)
fig.legend(hs, ["TF-specific elements (1–2 TFs)", "Conserved elements (≥3 TFs)", "q = 0.01", "1e-5", "≤1e-10", "tested, no significant term", "fewer than 20 elements, not tested"],
           loc="lower left", bbox_to_anchor=(6 / WD, 17 / HT), ncol=4, fontsize=7, handlelength=1.0)
def illustrator_text(svg):
    def one(m):
        tag = m.group(0); sm = re.search(r'style="([^"]*)"', tag)
        if not sm: return tag
        keep, size = [], None
        for dd in [x.strip() for x in sm.group(1).split(";") if x.strip()]:
            k, _, v = dd.partition(":"); k, v = k.strip(), v.strip()
            if k == "font": mm_ = re.match(r"(?:\S+\s+)?([\d.]+)px", v); size = mm_.group(1) if mm_ else None
            else: keep.append(f"{k}: {v}")
        if size is None: return tag
        tag = tag.replace(sm.group(0), f'style="{"; ".join(keep)}"' if keep else "")
        return tag[:-1].rstrip() + f' font-family="Arial, ArialMT, Helvetica, sans-serif" font-size="{float(size):.3f}">'
    return re.sub(r"<(?:text|tspan)\b[^>]*>", one, svg)
stem = H / "supplementary/go_per_tf_great_terms"; fig.savefig(f"{stem}.svg", bbox_inches="tight", pad_inches=0.08); fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
p = Path(f"{stem}.svg"); p.write_text(illustrator_text(p.read_text())); print("all TFs", len(tfs), "; near-duplicates merged", ndup, "; terms", nr, "TFs", nc, "(TF-specific TFs", len(set(SP.tf)), ")")
