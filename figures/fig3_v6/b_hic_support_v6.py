"""Fig 3 v6 panel b: the approved S1D palette C figures D1 (pairs with Hi-C loop support + observed vs own 10,000-shuffle null) and
D2 (per-TF loop-edge coverage), deck slide 15; re-run from hub_figures/basic/descriptive/plot_s1d_split_paletteC.py, same data and
style. v6 changes: wording only ("clique" -> "higher-order co-binding module" / "module edge") and the figure size set to the panel
size on the Fig 3 page (fonts stay 5-7 pt); PDF written next to the SVG.
  python b_hic_support_v6.py <outdir> [D1 w h mm] [D2 w h mm]"""
import glob, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
ROOT = Path("/Users/dyy/Desktop/HEK293T-ATGTAG")
sys.path.insert(0, str(ROOT / "palette"))
import regulome_palette as rp
P = rp.use()
HERE = ROOT / "hub_figures/basic/descriptive"
NULL = HERE / "null_s1d_perm10000/null"   # copy of hic/work_wt293t/loops/ourrun_tf_loops_0.7_pad30kb_perm10000/null
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
SZ = [float(v) for v in sys.argv[2:6]] if len(sys.argv) >= 6 else [150, None, 110, None]
MM = 1 / 25.4
N = P.neutral; RED = P.evidence.hic_loop
T = pd.read_csv(HERE / "s1d_per_tf_null.tsv", sep="\t").set_index("tf")
null = {tf: np.vstack([np.load(f)["null"] for f in sorted(glob.glob(str(NULL / f"{tf}_*.npz")))]) for tf in T.index}
# ---------- (1) pairs ----------
W = T[T.pairs >= 500].sort_values("pairs")
y = np.arange(len(W))
fig = plt.figure(figsize=(SZ[0] * MM, (SZ[1] or 2.6 * len(W) + 18) * MM))
gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.05)
a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1], sharey=a)
sup = (W.pair_obs * W.pairs).round()
a.barh(y, W.pairs, color=N.pale, height=0.75, label="pairs >25 kb")
a.barh(y, sup, color=RED, height=0.75, label="… on a 0.7 Hi-C loop (±30 kb)")
a.set_xscale("log")
for i, (n, o) in enumerate(zip(W.pairs, W.pair_obs)):
    a.text(n * 1.08, i, f"{100 * o:.1f}%", va="center", fontsize=5, color=N.dark)
a.set_yticks(y); a.set_yticklabels(W.index, fontsize=5.5); a.set_xlabel("co-binding pairs >25 kb (log scale)")
a.set_xlim(right=W.pairs.max() * 3)
a.legend(loc="lower right")
a.set_title("Pairs with Hi-C loop support", loc="left")
x0, x1 = 12, max(W.pair_obs.max() * 100 + 1.5, 28)
for i, tf in enumerate(W.index):
    v = null[tf][:, 0] * 100
    lo, q1, md, q3, hi = np.percentile(v, [2.5, 25, 50, 75, 97.5])
    b.plot([lo, hi], [i, i], color=N.mid, lw=0.7, solid_capstyle="butt")
    b.add_patch(plt.Rectangle((q1, i - 0.28), q3 - q1, 0.56, color=N.light, lw=0))
    b.plot([md, md], [i - 0.28, i + 0.28], color=N.dark, lw=0.7)
    b.plot([md, W.pair_obs[tf] * 100], [i, i], color=N.light, lw=0.6, zorder=1)
    b.scatter([W.pair_obs[tf] * 100], [i], s=7, color=N.ink, lw=0, zorder=3)
b.scatter([], [], s=7, color=N.ink, label="observed")
b.add_patch(plt.Rectangle((0, 0), 0, 0, color=N.light, label="own shuffle null: box = IQR, line = median, whiskers = 95%"))
b.legend(loc="upper center", bbox_to_anchor=(0.1, -0.075), ncol=2)
b.set_xlim(x0, x1)
b.set_xlabel("% of pairs on a loop")
b.set_title("Observed vs own null (10,000 shuffles)", loc="left")
b.xaxis.grid(True, color=N.faint, lw=0.5); b.set_axisbelow(True)
b.tick_params(axis="y", left=False, labelleft=False)
b.spines["left"].set_visible(False)
a.set_ylim(-0.7, len(W) - 0.3)
fig.text(0.99, -0.03, f"TFs with ≥500 pairs; same-chromosome shuffles keep each pair's spacing; all {len(W)} TFs BH q < 0.05", ha="right", fontsize=5.5, color=N.dark)
for ext in ("svg", "pdf", "png"): fig.savefig(OUT / f"b1_pairs_loop_support.{ext}", bbox_inches="tight")
plt.close(fig)
# ---------- (2) cliques ----------
C = pd.read_csv(HERE / "clique_loop_coverage.tsv", sep="\t")
labs = ["no loop edge", ">0–25%", "25–50%", "50–75%", "75–<100%", "100% (all edges)"]
cols = P.loop_strata
cv = C.coverage
C["stratum"] = np.select([C.loop_edges == 0, cv <= .25, cv <= .5, cv <= .75, cv < 1], labs[:5], labs[5])
Pt = C.groupby(["tf", "stratum"]).size().unstack(fill_value=0).reindex(columns=labs, fill_value=0)
Pt = Pt[Pt.sum(1) >= 20]; Pt = Pt.loc[Pt.sum(1).sort_values().index]
cap = 2500
fig, ax = plt.subplots(figsize=(SZ[2] * MM, (SZ[3] or 2.6 * len(Pt) + 18) * MM))
y = np.arange(len(Pt)); left = np.zeros(len(Pt))
scale = np.minimum(1.0, cap / Pt.sum(1).values)      # bars longer than the axis are shrunk to it (marked //), keeping proportions
for l, c in zip(labs, cols):
    v = Pt[l].values.astype(float) * scale; ax.barh(y, v, left=left, color=c, height=0.75, label=l); left += v
tot = Pt.sum(1).values; sup = tot - Pt[labs[0]].values
for i, (t, s_) in enumerate(zip(tot, sup)):
    if t > cap:
        ax.text(cap * 1.005, i, f"// {t:,} ({100 * s_ / t:.0f}% with a loop edge)", va="center", fontsize=5)
    else:
        ax.text(t + 20, i, f"{100 * s_ / t:.0f}%", va="center", fontsize=5, color=N.dark)
ax.set_xlim(0, cap * 1.32); ax.set_yticks(y); ax.set_yticklabels(Pt.index, fontsize=5.5)
ax.set_xlabel("higher-order co-binding modules (with ≥1 edge >25 kb)"); ax.legend(title="share of module edges on a 0.7 loop", loc="lower right", title_fontsize=6)
ax.set_title("Modules by Hi-C loop-edge coverage (% = modules with ≥1 loop edge)", loc="left")
for ext in ("svg", "pdf", "png"): fig.savefig(OUT / f"b2_module_loop_coverage.{ext}", bbox_inches="tight")
plt.close(fig)
