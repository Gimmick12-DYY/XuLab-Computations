"""Supplementary: GO terms x TFs matrix from the GREAT-style region test (same term sets as j_go_specific_vs_conserved_summary.py:
TF-specific = distal-vs-distal passing terms, conserved = all-element passing terms; all TFs beat their random-peak null).
Rows = each TF's top 4 TF-specific and top 4 conserved terms (by q), union, grouped by GO-slim theme (light bands); near-duplicates
merged: a term is dropped when another kept term is significant in exactly the same TFs (same group). Columns = all TFs that passed.
Cell: light hollow dot = TF-specific term, teal dot = conserved term, size = -log10 q (capped at 10). Full lists: go_per_tf_great_terms.xlsx.
  python Figure3/scripts/supp_go_terms_matrix.py   (Mac)"""
import re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = Path(__file__).resolve().parents[1]; G = H / "checks/go_great_regions"; MM = 1 / 25.4
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 6, "svg.fonttype": "none", "axes.linewidth": 0.5, "legend.frameon": False})
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
tfs = sorted(set(SP.tf)) + sorted(set(CO.tf) - set(SP.tf))
ti = {t: i for i, t in enumerate(terms.term)}; fi = {t: i for i, t in enumerate(tfs)}
nr, nc = len(terms), len(tfs)
RH = 2.4; HT = 34 + RH * nr; fig = plt.figure(figsize=(183 * MM, HT * MM)); ax = fig.add_axes([0.36, 14 / HT, 0.46, nr * RH / HT])
sz = lambda q: 1.5 + 2.2 * np.minimum(-np.log10(q), 10)
for d, kw, dx in ((CO, dict(color=TEAL, lw=0), 0.18), (SP, dict(color=LT, edgecolors=TEAL, linewidths=0.5), -0.18)):
    d = d[d.term.isin(ti)]; ax.scatter([fi[t] + dx for t in d.tf], [ti[t] for t in d.term], s=sz(d.q), zorder=3, **kw)
def short(s, n=58):
    s = s[0].upper() + s[1:]; return s if len(s) <= n else s[:n - 1] + "…"
ax.set_yticks(range(nr)); ax.set_yticklabels([short(n) for n in terms.name], fontsize=6); ax.set_ylim(nr - 0.5, -0.5)
ax.set_xticks(range(nc)); ax.set_xticklabels(tfs, rotation=90, fontsize=6); ax.xaxis.tick_top(); ax.set_xlim(-0.6, nc - 0.4)
ax.tick_params(length=0); ax.grid(color="#EEEEEE", lw=0.3); ax.set_axisbelow(True)
for sp in ax.spines.values(): sp.set_visible(False)
ax.axvline(len(set(SP.tf)) - 0.5, color="#9E9E9E", lw=0.5)
prev = None
for i, th in enumerate(terms.theme):
    if th != prev:
        k = (terms.theme == th).sum()
        if THEMES.index(th) % 2 == 0: ax.axhspan(i - 0.5, i + k - 0.5, color="#F4F4F4", zorder=0, lw=0)
        ax.text(nc - 0.1, i + k / 2 - 0.5, th.replace(" (muscle, heart, kidney, gut)", "").replace(" / ", "/\n", 1), fontsize=6, va="center", ha="left", color="#4D4D4D", clip_on=False, linespacing=0.95)
        prev = th
hs = [plt.scatter([], [], s=sz(1e-3), color=LT, edgecolors=TEAL, linewidths=0.5), plt.scatter([], [], s=sz(1e-3), color=TEAL, lw=0)] + [plt.scatter([], [], s=sz(q), color="#BDBDBD", lw=0) for q in (1e-2, 1e-5, 1e-10)]
fig.legend(hs, ["TF-specific elements (1–2 TFs)", "Conserved elements (≥3 TFs)", "q = 0.01", "1e-5", "≤1e-10"], loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=5, fontsize=6)
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
stem = H / "supplementary/go_per_tf_great_terms"; fig.savefig(f"{stem}.svg"); fig.savefig(f"{stem}.png", dpi=300)
p = Path(f"{stem}.svg"); p.write_text(illustrator_text(p.read_text())); print("near-duplicates merged", ndup, "; terms", nr, "TFs", nc, "(TF-specific TFs", len(set(SP.tf)), ")")
