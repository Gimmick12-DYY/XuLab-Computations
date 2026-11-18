"""Fig 3 panel j draft (user 2026-11-18): simple comparison of TF-specific vs conserved GO results from the GREAT-style region test
(scripts/go_great_regions.py / go_great_controls.py / go_great_null.py, checks/go_great_regions).
TF-specific terms: per TF, module elements of 1-2 TFs, distal elements vs the TF's other distal peaks; terms with q < 0.05, >= 5 regions,
  >= 3 distinct genes, no paralog-cluster artifact (surviving_per_TF.tsv); all 22 such TFs also beat their random-peak null.
Conserved terms: per TF, module elements of >= 3 TFs vs all the TF's peaks, same term filters (per_TF_controls.tsv.gz, comparison 'all');
  all TFs with terms beat their random-peak null.
Left: top 10 TF-specific terms by q, at most one per TF (label 'term (TF)'); x = -log10 q, size = foreground regions. Right: the same
construction for conserved terms (best term of each TF, 10 TFs with the best q). Both sides share one x axis (-log10 q). Also writes the full term list (supplementary/go_per_tf_great_terms.xlsx);
the supplementary matrix figure is scripts/supp_go_terms_matrix.py.
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
def short(s, n=40):
    s = s.replace("positive regulation of", "pos. reg.").replace("negative regulation of", "neg. reg.").replace("regulation of", "reg.").replace(" process", "")
    s = s.replace("biosynthetic", "biosynth.").replace("signaling pathway", "signaling").replace("type II interferon-mediated", "IFN-γ").replace("type I interferon", "type I IFN").replace("long-term synaptic depression", "synaptic LTD")
    s = s.replace("double-strand break repair via homologous recombination", "HR repair").replace("intracellular transport", "intracell. transport"); s = s[0].upper() + s[1:]
    return s if len(s) <= n else s[:n - 1] + "…"
def core(n):
    n = n.lower().replace("torc1", "tor")
    for w in ("positive regulation of ", "negative regulation of ", "regulation of ", "cellular "): n = n.replace(w, "")
    return n.replace(" process", "").strip()
# left: TF-specific, top 10 by q, one term per TF
L = SP.sort_values("q").drop_duplicates("tf").head(10).reset_index(drop=True)
# right: conserved, same construction as the left (top term per TF, 10 TFs with the best q) so both sides read the same way
R = CO.sort_values("q").drop_duplicates("tf").head(10).reset_index(drop=True)
NS, NC = SP.tf.nunique(), CO.tf.nunique()
fig = plt.figure(figsize=(130 * MM, 58 * MM)); a1 = fig.add_axes([0.31, 0.23, 0.16, 0.61]); a2 = fig.add_axes([0.78, 0.23, 0.16, 0.61], sharex=a1)
sz = lambda n: 4 + 0.12 * np.asarray(n, float)
y = np.arange(10)
for ax, D, kw, ttl, n_ in ((a1, L, dict(color=LT, edgecolors=TEAL, linewidths=0.6), "TF-specific elements (1–2 TFs)", NS),
                           (a2, R, dict(color=TEAL, lw=0), "Conserved elements (≥3 TFs)", NC)):
    ax.scatter(-np.log10(D.q), y[:len(D)], s=sz(D.fg_regions), zorder=3, **kw)
    ax.set_yticks(y[:len(D)]); ax.set_yticklabels([f"{short(n, 34)} ({t})" for n, t in zip(D.name, D.tf)], fontsize=5.5)
    ax.set_ylim(9.5, -0.6); ax.tick_params(axis="y", length=0); ax.xaxis.grid(True, color="#EEEEEE", lw=0.5); ax.set_axisbelow(True)
    ax.set_xlabel("−log10 q"); ax.set_title(f"{ttl}\nbest term per TF (10 of {n_} TFs)", fontsize=6.5, pad=4)
a1.set_xlim(0, max(-np.log10(L.q.min()), -np.log10(R.q.min())) * 1.08)
hs = [plt.scatter([], [], s=sz(n), color="#BDBDBD", lw=0) for n in (20, 100, 400)]
fig.legend(hs, ["20", "100", "400 regions"], loc="lower center", bbox_to_anchor=(0.6, -0.01), ncol=3, fontsize=5.5, handletextpad=0.2, columnspacing=1.0)
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
print(L[["tf", "name", "q", "fg_regions", "distinct_genes"]].to_string(index=False)); print(R[["tf", "name", "q", "fg_regions", "distinct_genes"]].to_string(index=False))
(H / "supplementary").mkdir(exist_ok=True)
cols = ["tf", "theme", "name", "term", "q", "fold", "fg_regions", "distinct_genes", "top_genes(regions)"]
with pd.ExcelWriter(H / "supplementary/go_per_tf_great_terms.xlsx") as w:
    SP.sort_values(["tf", "q"])[cols].to_excel(w, sheet_name="TF_specific_terms", index=False); CO.sort_values(["tf", "q"])[cols].to_excel(w, sheet_name="conserved_terms", index=False)
    L.to_excel(w, sheet_name="panel_left_top10", index=False); R.to_excel(w, sheet_name="panel_right_top10", index=False)
    pd.read_csv(G / "slim_to_theme.tsv", sep="\t").to_excel(w, sheet_name="slim_to_theme", index=False)

