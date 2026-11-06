"""Figure for the direct Hi-C hub test (palette C: red = Hi-C-supported loop-edge, blue = Cicero-only, ink = observed, grey = null).
a) 10 kb O/E among the 22 hub promoter bins (genomic order), Cicero hub edges outlined; b) hub mean O/E vs nulls;
c) edge-level O/E vs TF-matched null; d) APA pileups (hub all pairs, its control, loop-edges, Cicero-only edges)."""
from pathlib import Path
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
O = Path("/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model/hic_hub_test")
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "axes.spines.top": False, "axes.spines.right": False})
RED, BLUE, INK, GREY = "#C0392B", "#4A6FA5", "#222222", "#9E9E9E"
R = json.load(open(O / "results_10kb.json")); R5 = json.load(open(O / "results_5kb.json"))
HB = pd.read_csv(O / "hub_bins_10kb.tsv", sep="\t"); S = np.load(O / "hub_oe_submatrix_10kb.npy")
EL = pd.read_csv(O.parent / "tf_polymer_hicfirst/hicfirst/figure_hub_flow/hub_elements.tsv", sep="\t"); EL["bin"] = (29730000 + 2000 * EL.bead) // 10000
lab = [", ".join(EL[EL.bin == b].label) for b in HB.bin]
HE = pd.read_csv(O / "edges_oe_10kb.tsv", sep="\t")
fig = plt.figure(figsize=(180 / 25.4, 62 / 25.4))
# a
ax = fig.add_axes([0.085, 0.16, 0.24, 0.72]); n = len(HB)
im = ax.imshow(np.log2(S), cmap="RdBu_r", vmin=-2, vmax=2, interpolation="nearest")
pos = {b: k for k, b in enumerate(HB.bin)}
for st, c in (("loop_edges_18", RED), ("cicero_only_6", BLUE)):
    for r in HE[HE.set == st].itertuples():
        if r.i in pos and r.j in pos:
            for x, y in ((pos[r.j], pos[r.i]), (pos[r.i], pos[r.j])): ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, ec=c, lw=0.9))
ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_yticklabels(lab, fontsize=4.2, fontstyle="italic"); ax.set_xticklabels(lab, fontsize=4.2, fontstyle="italic", rotation=90)
ax.tick_params(length=0); ax.set_title("a  Hi-C O/E among hub promoters (10 kb)", loc="left", fontsize=7, x=-0.33)
cb = fig.add_axes([0.332, 0.5, 0.006, 0.3]); fig.colorbar(im, cax=cb); cb.set_ylabel("log2 O/E", fontsize=6); cb.tick_params(labelsize=5)
fig.text(0.332, 0.40, "\u25a1 loop-edge", color=RED, fontsize=5.5, va="top"); fig.text(0.332, 0.33, "\u25a1 Cicero-only", color=BLUE, fontsize=5.5, va="top")
# b
ax = fig.add_axes([0.47, 0.2, 0.16, 0.66])
L = np.load(O / "null_hub_all_pairs.npy")
nulls = [("distance-\nmatched,\nchr16", np.load(O / "hub_null_dist_all_10kb.npy")), ("TF-matched,\nchr16", np.load(O / "hub_null_tf_all_10kb.npy")),
         ("TF-matched,\nsame\n1.8 Mb", L)]
for k, (nm, v) in enumerate(nulls):
    vp = ax.violinplot(v, positions=[k], widths=0.75, showextrema=False)
    for b in vp["bodies"]: b.set_facecolor(GREY); b.set_alpha(0.5); b.set_edgecolor("none")
obs = R["hub_all_tf"]["obs"]; obs_loc = 1.533
ax.hlines(obs, -0.45, 1.45, color=INK, lw=1.2); ax.hlines(obs_loc, 1.55, 2.45, color=INK, lw=1.2)
ps = [R["hub_all_dist"]["p"], R["hub_all_tf"]["p"], 0.216]
for k, p in enumerate(ps): ax.text(k, 2.0, "p<0.001" if p <= 0.001 else f"p={p:.2f}", ha="center", fontsize=5.5)
ax.set_xticks(range(3)); ax.set_xticklabels([x[0] for x in nulls], fontsize=5); ax.set_ylabel("mean O/E, all hub pairs"); ax.set_ylim(0.6, 2.1)
ax.set_title("b  The hub as a module", loc="left", fontsize=7); ax.text(2.45, obs + 0.03, "hub", ha="right", fontsize=5.5, color=INK)
# c
ax = fig.add_axes([0.69, 0.2, 0.12, 0.66])
for k, (st, c) in enumerate((("loop_edges_18", RED), ("cicero_only_6", BLUE), ("shared_edges_locus", GREY))):
    v = np.log2(HE[HE.set == st].oe.clip(lower=0.05)); x = k + np.random.default_rng(k).uniform(-0.15, 0.15, len(v))
    ax.scatter(x, v, s=5, color=c, lw=0); ax.hlines(np.median(v), k - 0.25, k + 0.25, color=INK, lw=1)
    r = R[f"edges_{st}"]; ax.hlines(np.log2(r["null_tf"]), k - 0.3, k + 0.3, color=GREY, lw=1, ls=(0, (2, 1)))
    ax.text(k, 2.3, f"{r['ratio_tf']:.2f}×\np={r['p_tf']:.2f}", ha="center", fontsize=5.2)
ax.set_xticks(range(3)); ax.set_xticklabels(["loop-\nedges", "Cicero-\nonly", "all shared\nedges"], fontsize=5.5); ax.set_ylabel("log2 O/E per edge"); ax.set_ylim(-2.5, 3.0)
ax.axhline(0, color="#CCCCCC", lw=0.4); ax.set_title("c  Single edges", loc="left", fontsize=7)
fig.text(0.66, 0.03, "bar = median; dashed = TF-matched null", fontsize=5)
# d
A = R["apa_10kb"]
for k, (nm, ttl) in enumerate((("hub_all_pairs", "hub pairs"), ("hub_all_pairs_control", "control"), ("loop_edges_18", "loop-edges"), ("cicero_only_6", "Cicero-only"))):
    ax = fig.add_axes([0.835 + (k % 2) * 0.08, 0.53 - (k // 2) * 0.4, 0.07, 0.3]); P = np.load(O / f"apa_{nm}.npy")
    ax.imshow(np.log2(P), cmap="RdBu_r", vmin=-1, vmax=1); ax.set_xticks([]); ax.set_yticks([])
    key = nm.replace("_control", ""); sc = A[key]["control_apa"] if nm.endswith("control") else A[key]["apa"]
    ax.set_title(f"{ttl}\nAPA {sc:.2f}", fontsize=5.5, pad=2)
fig.text(0.835, 0.92, "d  Pileups (O/E, ±50 kb)", fontsize=7)
for e in ("svg", "pdf"): fig.savefig(O / f"hub_hic_test.{e}", transparent=True)
fig.savefig(O / "hub_hic_test_preview.png", dpi=300, facecolor="white"); print("ok")
