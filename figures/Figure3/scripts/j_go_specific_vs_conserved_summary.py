"""Fig 3 panel j draft (user 2026-11-18): simple comparison of TF-specific vs conserved GO results from the GREAT-style region test
(scripts/go_great_regions.py / go_great_controls.py / go_great_null.py, checks/go_great_regions).
TF-specific terms: per TF, module elements of 1-2 TFs, distal elements vs the TF's other distal peaks; terms with q < 0.05, >= 5 regions,
  >= 3 distinct genes, no paralog-cluster artifact (surviving_per_TF.tsv); all 22 such TFs also beat their random-peak null.
Conserved terms: per TF, module elements of >= 3 TFs vs all the TF's peaks, same term filters (per_TF_controls.tsv.gz, comparison 'all');
  all TFs with terms beat their random-peak null.
Left: top 10 TF-specific terms by q, at most one per TF (label 'term (TF)'); x = -log10 q, size = foreground regions. Right: the 10 conserved
terms significant in the most TFs (no TF names; near-duplicates dropped), x = median fold over those TFs. Both sides: x = BH q-value
(-log10 axis, ticks shown as q; right = median over the TFs with the term), dot size = number of genes behind the term (right = median), dashed line q = 0.05; 'n TFs' printed next to each dot. Also writes the full term list (supplementary/go_per_tf_great_terms.xlsx);
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
    s = s.replace("double-strand break repair via homologous recombination", "HR repair").replace("protein localization to nucleus", "nuclear protein import"); s = s[0].upper() + s[1:]
    return s if len(s) <= n else s[:n - 1] + "…"
def core(n):
    n = n.lower().replace("torc1", "tor")
    for w in ("positive regulation of ", "negative regulation of ", "regulation of ", "cellular "): n = n.replace(w, "")
    return n.replace(" process", "").strip()
# left: TF-specific, top 10 by q, one term per TF
L = SP.sort_values("q").drop_duplicates("tf").head(10).reset_index(drop=True)
# right: conserved terms (no TF names): ranked by the number of TFs in which the term is significant; near-duplicates = same TF set or
# same core name as a better-ranked term; x = median fold enrichment over those TFs
R = CO.groupby(["term", "name"]).agg(n_tfs=("tf", "nunique"), tfset=("tf", lambda x: frozenset(x)), fold=("fold", "median"), q=("q", "median"),
                                      fg_regions=("fg_regions", "median"), genes=("distinct_genes", "median")).reset_index().sort_values(["n_tfs", "fold"], ascending=[False, False])
keep, seen_sets, seen_core = [], set(), set()
for _, r in R.iterrows():
    if r.tfset in seen_sets or core(r["name"]) in seen_core: continue
    keep.append(r); seen_sets.add(r.tfset); seen_core.add(core(r["name"]))
    if len(keep) == 10: break
R = pd.DataFrame(keep).reset_index(drop=True)
L["n_tfs"] = [SP[SP.term == t].tf.nunique() for t in L.term]
NS, NC = SP.tf.nunique(), CO.tf.nunique()
fig = plt.figure(figsize=(200 * MM, 82 * MM)); a1 = fig.add_axes([0.255, 0.28, 0.2, 0.57]); a2 = fig.add_axes([0.745, 0.28, 0.2, 0.57], sharex=a1)
sz = lambda n: 6 + 1.6 * np.asarray(n, float)                                  # dot size = number of genes behind the term
y = np.arange(10)
a1.scatter(-np.log10(L.q), y[:len(L)], s=sz(L.distinct_genes), color=LT, edgecolors=TEAL, linewidths=0.7, zorder=3)
a1.set_yticks(y[:len(L)]); a1.set_yticklabels([f"{short(n, 60)}  ({t})" for n, t in zip(L.name, L.tf)], fontsize=6.5)
a1.set_title("TF-specific module elements (1–2 TFs)", fontsize=7, pad=5)
a2.scatter(-np.log10(R.q), y[:len(R)], s=sz(R.genes), color=TEAL, lw=0, zorder=3)
for ax, D, gc in ((a2, R, "genes"),):                                       # "n TFs" only on the conserved side (user)
    for i, (f, n, g) in enumerate(zip(-np.log10(D.q), D.n_tfs, D[gc])):
        ax.annotate(f"{n} TF" + ("s" if n > 1 else ""), (f, i), xytext=(np.sqrt(sz(g)) / 2 + 2.5, 0), textcoords="offset points", va="center", ha="left", fontsize=6, color=GREY)
a2.set_yticks(y[:len(R)]); a2.set_yticklabels([short(n, 60) for n in R.name], fontsize=6.5)
a2.set_title("Conserved module elements (≥3 TFs)", fontsize=7, pad=5)
for ax in (a1, a2):
    ax.set_ylim(9.5, -0.6); ax.tick_params(axis="y", length=0); ax.xaxis.grid(True, color="#EEEEEE", lw=0.5); ax.set_axisbelow(True)
    ax.axvline(-np.log10(0.05), color="#9E9E9E", lw=0.6, ls=(0, (3, 2)), zorder=0); ax.set_xlabel("q-value (BH)")
a1.set_xlim(0, max(-np.log10(L.q.min()), -np.log10(R.q.min())) * 1.3)
from matplotlib.ticker import FixedLocator, FuncFormatter
tk = [t for t in (2, 3, 4, 5, 6, 8, 10, 12, 14, 16) if t <= a1.get_xlim()[1]]
for ax in (a1, a2):
    ax.xaxis.set_major_locator(FixedLocator(tk)); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"1e-{int(v)}"))
    ax.text(-np.log10(0.05) + 0.08, -0.55, "q = 0.05", ha="left", va="top", fontsize=5.5, color="#7F7F7F")
a1.set_xlabel("q-value (BH)"); a2.set_xlabel("q-value (BH)")
hs = [plt.scatter([], [], s=sz(n), color="#BDBDBD", lw=0) for n in (5, 20, 50)]
fig.legend(hs, ["5", "20", "50 genes"], title="Genes", loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=6,
           title_fontsize=6, handletextpad=0.3, columnspacing=1.5)
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
print(L[["tf", "name", "q", "fg_regions", "distinct_genes"]].to_string(index=False)); print(L[["tf", "name", "q", "distinct_genes"]].to_string(index=False)); print(R[["name", "n_tfs", "q", "genes"]].to_string(index=False))
(H / "supplementary").mkdir(exist_ok=True)
cols = ["tf", "theme", "name", "term", "q", "fold", "fg_regions", "distinct_genes", "top_genes(regions)"]
with pd.ExcelWriter(H / "supplementary/go_per_tf_great_terms.xlsx") as w:
    SP.sort_values(["tf", "q"])[cols].to_excel(w, sheet_name="TF_specific_terms", index=False); CO.sort_values(["tf", "q"])[cols].to_excel(w, sheet_name="conserved_terms", index=False)
    L.to_excel(w, sheet_name="panel_left_top10", index=False); R.drop(columns=["tfset"]).to_excel(w, sheet_name="panel_right_top10", index=False)
    pd.read_csv(G / "slim_to_theme.tsv", sep="\t").to_excel(w, sheet_name="slim_to_theme", index=False)

