"""Fig 3 v7 panel a ("edges percentage"): the approved S1D module loop-edge coverage bars (figD2, deck slide 15) with, rows aligned at
its right, the observed-vs-own-null panel for module edges >25 kb drawn with the approved D1 null code (box = IQR, tick = median,
whisker = 95%, dot = observed). Re-run of hub_figures/basic/descriptive/plot_s1d_split_paletteC.py at the final panel size (palette C,
Arial, 7 pt labels, 6 pt ticks/legends, 5 pt TF names, 5 pt bar values); no panel titles or footnotes (text in figure_legend_notes.md);
frameless compact legends in free plot space; stacked bars with a 0.3 pt dark edge (figures4papers convention).
NULL_MODE=edge (default, Fig 3): per-edge shuffle, each module edge >25 kb shifted on its chromosome keeping its spacing, as for pairs
  (figures/fig3_v6/checks/module_edge_null_check.py, 10 x 1,000 shuffles per TF);
NULL_MODE=block (supplement): the stricter S1D shuffle, each whole module shifted as one block (perm_tf_long.py, column 1).
  python a_module_edges_v7.py <outdir> [w_mm h_mm]"""
import glob, os, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests
ROOT = Path("/Users/dyy/Desktop/HEK293T-ATGTAG"); sys.path.insert(0, str(ROOT / "palette"))
import regulome_palette as rp
P = rp.use(); N = P.neutral; MM = 1 / 25.4
plt.rcParams.update({"legend.frameon": False, "axes.linewidth": 0.6, "svg.fonttype": "none", "pdf.fonttype": 42})
HERE = ROOT / "hub_figures/basic/descriptive"; NULL = HERE / "null_s1d_perm10000/null"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True); WMM, HMM = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (112, 90)
MODE = os.environ.get("NULL_MODE", "edge"); MIN_E = 100
ND = Path(os.environ.get("EDGE_NULL_DIR", str(ROOT / "fig3_v6/checks/null_per_edge")))
T = pd.read_csv(HERE / "s1d_per_tf_null.tsv", sep="\t").set_index("tf")
# ---- D2 data, exactly as the approved script ----
C = pd.read_csv(HERE / "clique_loop_coverage.tsv", sep="\t")
labs = ["no loop edge", ">0–25%", "25–50%", "50–75%", "75–<100%", "100% (all edges)"]; cols = P.loop_strata; cv = C.coverage
C["stratum"] = np.select([C.loop_edges == 0, cv <= .25, cv <= .5, cv <= .75, cv < 1], labs[:5], labs[5])
Pt = C.groupby(["tf", "stratum"]).size().unstack(fill_value=0).reindex(columns=labs, fill_value=0)
Pt = Pt[Pt.sum(1) >= 20]; Pt = Pt.loc[Pt.sum(1).sort_values().index]
# ---- nulls ----
E = T.loc[[t for t in Pt.index if t in T.index and T.edges[t] >= MIN_E]].copy()
if MODE == "edge":
    nul = {t: np.concatenate([np.load(f)["null"] for f in sorted(ND.glob(f"{t}_*.npz"))]) for t in E.index}
    for t in E.index:
        assert abs(float(np.load(next(ND.glob(f"{t}_*.npz")))["obs"]) - E.edge_obs[t]) < 1e-12 and len(nul[t]) == 10000, t
    E["edge_p"] = [(1 + (nul[t] >= E.edge_obs[t]).sum()) / (1 + len(nul[t])) for t in E.index]
else:
    nul = {t: np.vstack([np.load(f)["null"] for f in sorted(glob.glob(str(NULL / f"{t}_*.npz")))])[:, 1] for t in E.index}
E["q"] = multipletests(E.edge_p, method="fdr_bh")[1]; nsig = int((E.q < 0.05).sum())
# ---- figure ----
fig = plt.figure(figsize=(WMM * MM, HMM * MM))
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 0.001], wspace=0.05, left=0.12, right=0.99, top=0.995, bottom=0.075)
ax = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1], sharey=ax)
cap = 2500; y = np.arange(len(Pt)); left = np.zeros(len(Pt)); scale = np.minimum(1.0, cap / Pt.sum(1).values)
for l, c in zip(labs, cols):
    v = Pt[l].values.astype(float) * scale; ax.barh(y, v, left=left, color=c, height=0.72, label=("No loop edge" if l == labs[0] else l), edgecolor="#4D4D4D", linewidth=0.3); left += v
tot = Pt.sum(1).values; sup = tot - Pt[labs[0]].values
for i, (t, s_) in enumerate(zip(tot, sup)):
    ax.text(cap * 1.01 if t > cap else t + 25, i, (f"// {t:,} ({100 * s_ / t:.0f}%)" if t > cap else f"{100 * s_ / t:.0f}%"), va="center", fontsize=5, color=N.dark)
ax.set_xlim(0, cap * 1.38); ax.set_yticks(y); ax.set_yticklabels(Pt.index, fontsize=5); ax.set_ylim(-0.7, len(Pt) - 0.3)
ax.set_xlabel("Modules with ≥1 edge >25 kb"); ax.tick_params(axis="y", length=0, pad=1.5)
ax.legend(title="Module edges on a loop", loc="lower right", ncol=1, fontsize=6, title_fontsize=6, handlelength=1.0, handleheight=0.8,
          labelspacing=0.25, borderaxespad=0.2, alignment="left")
for i, tf in enumerate(Pt.index):
    if tf not in E.index:
        b.text(13.0, i, "n/a", va="center", fontsize=5, color=N.mid); continue
    v = nul[tf] * 100; lo, q1, md, q3, hi = np.percentile(v, [2.5, 25, 50, 75, 97.5])
    b.plot([lo, hi], [i, i], color=N.mid, lw=0.6, solid_capstyle="butt")
    b.add_patch(plt.Rectangle((q1, i - 0.28), q3 - q1, 0.56, color=N.light, lw=0))
    b.plot([md, md], [i - 0.3, i + 0.3], color=N.dark, lw=0.6)
    b.plot([md, E.edge_obs[tf] * 100], [i, i], color=N.light, lw=0.5, zorder=1)
    sig = E.q[tf] < 0.05
    b.scatter([E.edge_obs[tf] * 100], [i], s=5, color=N.ink if sig else "white", edgecolors=N.ink, linewidths=0 if sig else 0.5, zorder=3)
hi_x = max(E.edge_obs.max() * 100, max(np.percentile(nul[t] * 100, 97.5) for t in E.index)) + 1
b.set_xlim(12, hi_x); b.set_xlabel("Module edges >25 kb on a loop (%)")
b.xaxis.grid(True, color=N.faint, lw=0.4); b.set_axisbelow(True)
b.tick_params(axis="y", left=False, labelleft=False); b.spines["left"].set_visible(False)
h = [plt.Line2D([], [], ls="", marker="o", ms=2.6, color=N.ink), plt.Line2D([], [], ls="", marker="o", ms=2.6, mfc="white", mec=N.ink, mew=0.5),
     plt.Rectangle((0, 0), 1, 1, color=N.light)]
b.legend(h, ["Observed, q < 0.05", "Observed, n.s.", "Shuffle null"], loc="lower right", fontsize=6, handlelength=1.0, labelspacing=0.25,
         borderaxespad=0.2, facecolor="white", frameon=True, framealpha=1, edgecolor="none")
stem = "a_module_edges" if MODE == "edge" else "S_module_edge_block_null"
for ext in ("svg", "pdf"): fig.savefig(OUT / f"{stem}.{ext}")
fig.savefig(OUT / f"{stem}.png", dpi=300)
E.assign(null_med=[np.median(nul[t]) for t in E.index], null_lo=[np.percentile(nul[t], 2.5) for t in E.index], null_hi=[np.percentile(nul[t], 97.5) for t in E.index])[
    ["edges", "edge_obs", "null_med", "null_lo", "null_hi", "edge_p", "q"]].to_csv(OUT / f"{stem}.tsv", sep="\t")
print(f"{MODE} null: {nsig} of {len(E)} TFs q < 0.05; median obs {100 * E.edge_obs.median():.1f}%, median null {100 * np.median([np.median(nul[t]) for t in E.index]):.1f}%")
