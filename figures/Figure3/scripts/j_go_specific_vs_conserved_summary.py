"""Fig 3 panel j draft (user 2026-11-18): simple comparison of TF-specific vs conserved GO results from the GREAT-style region test
(scripts/go_great_regions.py / go_great_controls.py / go_great_null.py, checks/go_great_regions).
TF-specific terms: per TF, module elements of 1-2 TFs, distal elements vs the TF's other distal peaks; terms with q < 0.05, >= 5 regions,
  >= 3 distinct genes, no paralog-cluster artifact (surviving_per_TF.tsv); all 22 such TFs also beat their random-peak null.
Conserved terms: per TF, module elements of >= 3 TFs vs all the TF's peaks, same term filters (per_TF_controls.tsv.gz, comparison 'all');
  all TFs with terms beat their random-peak null.
Themes: GO slim generic (+ response to stress), scripts/go_theme_map.py, table slim_to_theme.tsv. Bars = number of TFs with >= 1 term in
the theme. Inset = pairwise Jaccard similarity of significant term sets between TFs. Also writes the supplementary per-TF matrix
(supplementary/go_per_tf_great_terms.*) and the full term list (supplementary/go_per_tf_great_terms.xlsx).
  python Figure3/scripts/j_go_specific_vs_conserved_summary.py   (Mac)"""
import re
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = Path(__file__).resolve().parents[1]; G = H / "checks/go_great_regions"; MM = 1 / 25.4
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 6, "axes.labelsize": 6.5, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "legend.frameon": False})
TEAL, LT, GREY = "#2C7A6B", "#C9E3DC", "#4D4D4D"
TH = pd.concat([pd.read_csv(G / f, sep="\t")[["term", "theme"]] for f in ("surviving_per_TF_themes.tsv", "per_TF_q005_themes.tsv")]).drop_duplicates("term").set_index("term").theme
SP = pd.read_csv(G / "surviving_per_TF.tsv", sep="\t"); SP = SP[SP.group == "specific"]
C = pd.read_csv(G / "per_TF_controls.tsv.gz", sep="\t"); CO = C[(C.group == "conserved") & (C.comparison == "all") & C.passes]
SP, CO = SP.copy(), CO.copy()
for d in (SP, CO): d["theme"] = d.term.map(TH).fillna("Other")
THEMES = ["Protein folding / stress", "Cell cycle / DNA / chromatin", "Gene expression / RNA", "Immune / inflammation", "Signalling", "Metabolism",
          "Development / differentiation", "Organ physiology (muscle, heart, kidney, gut)", "Neural", "Cell organization / transport / adhesion"]
LAB = {"Organ physiology (muscle, heart, kidney, gut)": "Organ physiology", "Cell organization / transport / adhesion": "Cell organization / transport"}
ns = SP[SP.theme.isin(THEMES)].groupby("theme").tf.nunique().reindex(THEMES, fill_value=0); nc = CO[CO.theme.isin(THEMES)].groupby("theme").tf.nunique().reindex(THEMES, fill_value=0)
order = list((nc - ns).sort_values(ascending=False).index)
def short(s, n=30):
    s = s.replace("positive regulation of", "pos. reg.").replace("negative regulation of", "neg. reg.").replace("regulation of", "reg.").replace(" process", "")
    return s if len(s) <= n else s[:n - 1] + "…"
ex = {}
for t in order:
    d = SP[SP.theme == t].sort_values("q").drop_duplicates("tf")
    ex[t] = "; ".join(f"{r.tf} · {short(r['name'])}" for _, r in d.head(2).iterrows())
NS, NC = SP.tf.nunique(), CO.tf.nunique()
fig = plt.figure(figsize=(100 * MM, 62 * MM)); ax = fig.add_axes([0.3, 0.14, 0.4, 0.72]); ins = fig.add_axes([0.84, 0.2, 0.14, 0.45])
y = np.arange(len(order)); h = 0.38
ax.barh(y - h / 2, [ns[t] for t in order], h, color=LT, edgecolor=TEAL, lw=0.6, label=f"TF-specific elements (1–2 TFs; {NS} TFs)")
ax.barh(y + h / 2, [nc[t] for t in order], h, color=TEAL, lw=0, label=f"Conserved elements (≥3 TFs; {NC} TFs)")
xm = max(ns.max(), nc.max()) * 1.12
for i, t in enumerate(order):
    ax.text(ns[t] + 0.3, i - h / 2, str(ns[t]), va="center", fontsize=5); ax.text(nc[t] + 0.3, i + h / 2, str(nc[t]), va="center", fontsize=5, color=TEAL)
ax.set_yticks(y); ax.set_yticklabels([LAB.get(t, t) for t in order]); ax.set_ylim(len(order) - 0.4, -0.6); ax.tick_params(axis="y", length=0)
ax.set_xlim(0, xm); ax.set_xlabel("TFs with ≥1 significant GO term in the theme")
ax.legend(loc="lower left", bbox_to_anchor=(-0.7, 1.01), ncol=1, fontsize=5.5, handlelength=1.0, columnspacing=1.0, borderaxespad=0)
J = {}
for nm, d in (("specific", SP), ("conserved", CO)):
    sets = {t: set(g.term) for t, g in d.groupby("tf")}; J[nm] = np.array([len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for a, b in combinations(sorted(sets), 2)])
bp = ins.boxplot([J["specific"], J["conserved"]], widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=0.8),
                 boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
bp["boxes"][0].set_facecolor(LT); bp["boxes"][0].set_edgecolor(TEAL); bp["boxes"][1].set_facecolor(TEAL)
ins.set_xticks([1, 2]); ins.set_xticklabels(["TF-\nspecific", "Con-\nserved"], fontsize=5.5); ins.tick_params(labelsize=5)
ins.set_title("Term overlap\nbetween TFs", fontsize=5.5, pad=3); ins.set_ylabel("Jaccard", fontsize=5.5); ins.set_ylim(0, max(0.05, np.percentile(J["conserved"], 90) * 1.2))
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
def save(fig, stem):
    fig.savefig(f"{stem}.svg"); fig.savefig(f"{stem}.png", dpi=300); p = Path(f"{stem}.svg"); p.write_text(illustrator_text(p.read_text()))
save(fig, H / "j_go_specific_vs_conserved_summary")
print(pd.DataFrame({"TF_specific_TFs": ns, "conserved_TFs": nc}).reindex(order).to_string()); print("Jaccard median specific", np.median(J["specific"]), "conserved", round(float(np.median(J["conserved"])), 4), "| TFs", NS, NC)
# ---- supplementary: per-TF theme matrix + full term list ----
B = pd.concat([SP.assign(group="specific"), CO.assign(group="conserved")]); B = B[B.theme.isin(THEMES)].groupby(["tf", "group", "theme"]).q.min().reset_index()
rows = sorted(B.tf.unique(), key=lambda t: (-int((B[(B.tf == t) & (B.group == "specific")]).shape[0] > 0), -B[(B.tf == t) & (B.group == "specific")].shape[0], t))
f2 = plt.figure(figsize=(120 * MM, (34 + 2.1 * len(rows)) * MM)); a2 = f2.add_axes([0.14, 0.23, 0.55, 0.72])
sz = lambda q: 2 + 3.2 * np.minimum(-np.log10(q), 10)
for grp, dx in (("specific", -0.17), ("conserved", 0.17)):
    d = B[B.group == grp]; xx = [THEMES.index(t) + dx for t in d.theme]; yy = [rows.index(t) for t in d.tf]
    if grp == "specific": a2.scatter(xx, yy, s=sz(d.q), color=LT, edgecolors=TEAL, linewidths=0.6, zorder=3)
    else: a2.scatter(xx, yy, s=sz(d.q), color=TEAL, lw=0, zorder=3)
a2.set_xticks(range(len(THEMES))); a2.set_xticklabels([LAB.get(t, t) for t in THEMES], rotation=45, ha="right", fontsize=5.5)
a2.set_yticks(range(len(rows))); a2.set_yticklabels(rows, fontsize=5); a2.set_ylim(len(rows) - 0.5, -0.5); a2.tick_params(length=0); a2.grid(color="#EEEEEE", lw=0.4); a2.set_axisbelow(True)
for sp in ("left", "bottom"): a2.spines[sp].set_visible(False)
hs = [plt.scatter([], [], s=sz(1e-2), color=LT, edgecolors=TEAL, linewidths=0.6), plt.scatter([], [], s=sz(1e-2), color=TEAL, lw=0)] + [plt.scatter([], [], s=sz(q), color="#BDBDBD", lw=0) for q in (1e-2, 1e-5, 1e-10)]
f2.legend(hs, ["TF-specific (1–2 TFs)", "Conserved (≥3 TFs)", "q = 0.01", "1e-5", "≤1e-10"], loc="upper left", bbox_to_anchor=(0.72, 0.9), fontsize=5.5, labelspacing=0.6)
(H / "supplementary").mkdir(exist_ok=True); save(f2, H / "supplementary/go_per_tf_great_terms")
cols = ["tf", "theme", "name", "term", "q", "fold", "fg_regions", "distinct_genes", "top_genes(regions)"]
with pd.ExcelWriter(H / "supplementary/go_per_tf_great_terms.xlsx") as w:
    SP.sort_values(["tf", "q"])[cols].to_excel(w, sheet_name="TF_specific_terms", index=False); CO.sort_values(["tf", "q"])[cols].to_excel(w, sheet_name="conserved_terms", index=False)
    pd.DataFrame({"TF_specific_TFs": ns, "conserved_TFs": nc}).reindex(order).to_excel(w, sheet_name="theme_counts")
    pd.read_csv(G / "slim_to_theme.tsv", sep="\t").to_excel(w, sheet_name="slim_to_theme", index=False)
print("supp rows", len(rows))
