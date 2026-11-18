"""DRAFT panel j, option A (user 2026-11-18): 'one programme vs many'. GREAT-style region enrichment per TF (checks/go_great_regions,
before the random-peak / open-chromatin controls), significant terms (q < 0.05, no paralog-cluster artifact) grouped into themes through
GO slim generic (scripts/go_theme_map.py; table slim_to_theme.tsv). Matrix: rows = TFs with >= 1 term in either group, columns = themes
(Other omitted); per cell, hollow dot = best TF-specific term (modules of 1-2 TFs), filled teal = best conserved term (>= 3 TFs); size =
-log10 q of that best term (capped at 10). Inset: pairwise Jaccard similarity of the significant term sets between TFs, per group.
  python Figure3/scripts/j_great_theme_matrix_draft.py   (Mac)"""
import re
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = Path(__file__).resolve().parents[1]; G = H / "checks/go_great_regions"; MM = 1 / 25.4
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 6, "axes.labelsize": 6.5, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "legend.frameon": False})
TEAL, GREY = "#2C7A6B", "#4D4D4D"
S = pd.read_csv(G / "per_TF_q005_themes.tsv", sep="\t"); S = S[~S.cluster_artifact.fillna(False).astype(bool)]
THEMES = ["Protein folding / stress", "Cell cycle / DNA / chromatin", "Gene expression / RNA", "Immune / inflammation", "Signalling", "Metabolism",
          "Development / differentiation", "Organ physiology (muscle, heart, kidney, gut)", "Neural", "Cell organization / transport / adhesion"]
LAB = {"Organ physiology (muscle, heart, kidney, gut)": "Organ physiology", "Cell organization / transport / adhesion": "Cell organization / transport"}
B = S[S.theme.isin(THEMES)].groupby(["tf", "group", "theme"]).q.min().reset_index()
tfs = B.groupby("tf").size().index   # TFs with >= 1 themed term
# row order: TFs with conserved stress first? -> order by number of themes hit by TF-specific terms, then name
nsp = B[B.group == "specific"].groupby("tf").theme.nunique().reindex(tfs, fill_value=0); ncs = B[B.group == "conserved"].groupby("tf").theme.nunique().reindex(tfs, fill_value=0)
rows = sorted(tfs, key=lambda t: (-int(nsp[t] > 0), -int(nsp[t]), -int(ncs[t]), t))
fig = plt.figure(figsize=(120 * MM, (34 + 2.1 * len(rows)) * MM)); h = len(rows)
ax = fig.add_axes([0.13, 0.23, 0.5, 0.71]); ins = fig.add_axes([0.8, 0.6, 0.17, 0.28])
sz = lambda q: 2 + 3.2 * np.minimum(-np.log10(q), 10)
for grp, dx in (("specific", -0.17), ("conserved", 0.17)):
    d = B[B.group == grp]; x = [THEMES.index(t) + dx for t in d.theme]; y = [rows.index(t) for t in d.tf]
    if grp == "specific": ax.scatter(x, y, s=sz(d.q), facecolors="white", edgecolors=GREY, linewidths=0.6, zorder=3)
    else: ax.scatter(x, y, s=sz(d.q), color=TEAL, lw=0, zorder=3)
ax.set_xticks(range(len(THEMES))); ax.set_xticklabels([LAB.get(t, t) for t in THEMES], rotation=45, ha="right", fontsize=5.5)
ax.set_yticks(range(h)); ax.set_yticklabels(rows, fontsize=5); ax.set_ylim(h - 0.5, -0.5); ax.set_xlim(-0.6, len(THEMES) - 0.4)
ax.tick_params(length=0); ax.grid(color="#EEEEEE", lw=0.4); ax.set_axisbelow(True)
for sp in ("left", "bottom"): ax.spines[sp].set_visible(False)
# inset: pairwise Jaccard of significant term sets
J = {}
for grp in ("specific", "conserved"):
    sets = {t: set(d.term) for t, d in S[S.group == grp].groupby("tf")}
    J[grp] = np.array([len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for a, b in combinations(sorted(sets), 2)])
bp = ins.boxplot([J["specific"], J["conserved"]], widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=0.8),
                 boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
bp["boxes"][0].set_facecolor("white"); bp["boxes"][1].set_facecolor(TEAL); bp["boxes"][1].set_alpha(0.85)
ins.set_xticks([1, 2]); ins.set_xticklabels(["TF-\nspecific", "Con-\nserved"], fontsize=5.5); ins.set_ylabel("Term overlap between TFs\n(Jaccard)", fontsize=5.5)
ins.tick_params(labelsize=5); ins.set_ylim(0, max(0.05, np.percentile(J["conserved"], 95) * 1.15))
hs = [plt.scatter([], [], s=sz(q), facecolors="white", edgecolors=GREY, linewidths=0.6) for q in (1e-2,)] + [plt.scatter([], [], s=sz(1e-2), color=TEAL, lw=0)] + \
     [plt.scatter([], [], s=sz(q), color="#BDBDBD", lw=0) for q in (1e-2, 1e-5, 1e-10)]
fig.legend(hs, ["TF-specific elements (1–2 TFs)", "Conserved elements (≥3 TFs)", "q = 0.01", "1e-5", "≤1e-10"], loc="upper left", bbox_to_anchor=(0.66, 0.47),
           fontsize=5.5, handletextpad=0.3, labelspacing=0.6)
fig.text(0.99, 0.995, "draft, pending random-peak and open-chromatin controls", ha="right", va="top", fontsize=5, color="#9E9E9E")
def illustrator_text(svg):
    def one(m):
        tag = m.group(0); sm = re.search(r'style="([^"]*)"', tag)
        if not sm: return tag
        keep, size = [], None
        for d in [d.strip() for d in sm.group(1).split(";") if d.strip()]:
            k, _, v = d.partition(":"); k, v = k.strip(), v.strip()
            if k == "font": mm_ = re.match(r"(?:\S+\s+)?([\d.]+)px", v); size = mm_.group(1) if mm_ else None
            else: keep.append(f"{k}: {v}")
        if size is None: return tag
        tag = tag.replace(sm.group(0), f'style="{"; ".join(keep)}"' if keep else "")
        return tag[:-1].rstrip() + f' font-family="Arial, ArialMT, Helvetica, sans-serif" font-size="{float(size):.3f}">'
    return re.sub(r"<(?:text|tspan)\b[^>]*>", one, svg)
stem = H / "j_great_theme_matrix_draft"; fig.savefig(f"{stem}.svg"); fig.savefig(f"{stem}.png", dpi=300)
p = Path(f"{stem}.svg"); p.write_text(illustrator_text(p.read_text()))
print("rows", h, "| Jaccard median specific", round(float(np.median(J["specific"])), 4), "conserved", round(float(np.median(J["conserved"])), 4),
      "| pairs", len(J["specific"]), len(J["conserved"]))
print("TF-specific themes per TF:", nsp[nsp > 0].sort_values(ascending=False).to_dict())
print(B[B.group == "conserved"].theme.value_counts().to_dict()); print(B[B.group == "specific"].theme.value_counts().to_dict())
