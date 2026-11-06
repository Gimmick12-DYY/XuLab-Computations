"""Hybrid step 12 (Mac): v2 (TF only) vs hybrid (TF + Hi-C loop restraints) validation figure (Arial 7 pt, live SVG text).
A: P(s) of Hi-C, v2, hybrid.  B: Spearman vs Hi-C with restrained-loop pixels (+-1 bin) excluded.
C: Cicero contact enrichment vs distance-matched and vs bound/TF-matched nulls.  D: hub-flower metrics.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model"); O = R / "tf_polymer_hic/figure_compare"; O.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 7, "pdf.fonttype": 42,
                     "svg.fonttype": "none", "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False})
C2, CH, CHIC = "#7F7F7F", "#D55E00", "#B2182B"
V2 = pd.read_csv(R / "tf_polymer_v2/analysis/validation.tsv", sep="\t"); V2 = V2[V2.eps_top == 16]
VH = pd.read_csv(R / "tf_polymer_hic/analysis/validation.tsv", sep="\t")
P2 = pd.read_csv(R / "tf_polymer_v2/analysis/ps.tsv", sep="\t"); P2 = P2[P2.eps_top == 16].reset_index(drop=True)
PH = pd.read_csv(R / "tf_polymer_hic/analysis/ps.tsv", sep="\t")
SH = pd.read_csv(R / "tf_polymer_hic/analysis/sweep_summary.tsv", sep="\t").iloc[0]
S2 = pd.read_csv(R / "tf_polymer_hic/v2_spearman_excl_loops.tsv", sep="\t", header=None, index_col=0)[1]
fig = plt.figure(figsize=(180 / 25.4, 55 / 25.4))
a = fig.add_axes([0.075, 0.2, 0.15, 0.68])
ok = (P2.s_kb <= 1500).values
for df, c, l in ((P2, CHIC, "Hi-C"), (P2, C2, "TF only (v2)"), (PH, CH, "TF + Hi-C loops")):
    col = "p_hic" if l == "Hi-C" else "p_sim"; v = df[col].values; v = v / v[1]
    a.loglog(df.s_kb.values[ok], v[ok], color=c, lw=0.8, label=l)
a.set_xlabel("s (kb)"); a.set_ylabel("P(s) / P(20 kb)"); a.legend(frameon=False, fontsize=6, loc="lower left"); a.set_title("A  P(s)", loc="left", fontsize=7)
b = fig.add_axes([0.29, 0.2, 0.15, 0.68])
vals = [(S2["raw_excl"], SH.spearman_raw_excl_loops), (S2["oe_excl"], SH.spearman_oe_excl_loops)]
for k, (v2, vh) in enumerate(vals):
    b.bar(k - 0.18, v2, 0.34, color=C2); b.bar(k + 0.18, vh, 0.34, color=CH)
    b.text(k - 0.18, v2 + 0.01, f"{v2:.2f}", ha="center", fontsize=6); b.text(k + 0.18, vh + 0.01, f"{vh:.2f}", ha="center", fontsize=6)
b.set_xticks([0, 1]); b.set_xticklabels(["raw", "O/E"]); b.set_ylabel("Spearman vs Hi-C"); b.set_ylim(0, 0.6)
b.set_title("B  loop pixels excluded", loc="left", fontsize=7)
c = fig.add_axes([0.50, 0.2, 0.27, 0.68])
sets = [("hub_edges", "hub\nedges"), ("shared_edges", "shared\nedges"), ("cicero_step1", "Cicero\nstep-1")]
for k, (st, lab) in enumerate(sets):
    for q, (nl, mk) in enumerate((("random_same_distance", "o"), ("bound_matched_class_distance", "s"))):
        for dv, (V, col) in enumerate(((V2, C2), (VH, CH))):
            r = V[(V.set == st) & (V.null == nl)].iloc[0]
            x = k + (q - 0.5) * 0.42 + (dv - 0.5) * 0.14
            c.scatter(x, r.agg_ratio, marker=mk, s=14, color=col, zorder=3)
c.axhline(1, color="#999999", lw=0.5, ls=(0, (2, 2))); c.set_yscale("log"); c.set_yticks([0.8, 1, 2, 4, 8]); c.set_yticklabels(["0.8", "1", "2", "4", "8"])
c.set_xticks(range(3)); c.set_xticklabels([s[1] for s in sets]); c.set_ylabel("contact / matched expectation")
c.set_title("C  Cicero contacts  (● vs random same distance, ■ vs TF-bound matched)", loc="left", fontsize=7)
d = fig.add_axes([0.80, 0.2, 0.19, 0.68]); d.set_axis_off()
h2 = V2[V2.set == "hub_cores_max_in_one_cluster"].iloc[0]; hh = VH[VH.set == "hub_cores_max_in_one_cluster"].iloc[0]
p2 = V2[V2.set == "hub_core_pairs_cocluster"].iloc[0]; ph = VH[VH.set == "hub_core_pairs_cocluster"].iloc[0]
l2 = V2[V2.set == "loops_peakachu07"].iloc[0]; lh = VH[VH.set == "loops_peakachu07"].iloc[0]
txt = (f"D  hub flower (9 cores)\n\n≥3 cores in one flower\n  TF only: {h2.frac_conf_ge3*100:.0f}% (null {h2.null_frac_conf_ge3*100:.0f}%)\n"
       f"  + loops: {hh.frac_conf_ge3*100:.0f}% (null {hh.null_frac_conf_ge3*100:.0f}%)\n\ncore pairs co-clustered\n"
       f"  TF only: {p2.agg_ratio:.2f}× (p={p2.p:.2f})\n  + loops: {ph.agg_ratio:.2f}× (p={ph.p:.2f})\n\n"
       f"Peakachu loops (hybrid input)\n  TF only: {l2.agg_ratio:.1f}×; + loops: {lh.agg_ratio:.1f}×")
d.text(0, 1, txt, va="top", fontsize=6)
for e in ("svg", "pdf"): fig.savefig(O / f"v2_vs_hybrid.{e}", transparent=True)
fig.savefig(O / "v2_vs_hybrid_preview.png", dpi=250, facecolor="white")
print("ok")
