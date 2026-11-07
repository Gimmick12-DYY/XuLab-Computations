"""Fig 3b (179 x 50 mm): Hi-C support of higher-order co-binding modules per TF, two rows sharing one TF axis.
b1 (top): modules per TF stacked by the share of their edges > 25 kb that sit on a WT Peakachu 0.7 loop (5 kb U 10 kb,
+-30 kb) = figD2 (hub_figures/basic/descriptive/plot_s1d_split_paletteC.py, table clique_loop_coverage.tsv), linear axis
shown as the share of each TF's modules (100% stacked, linear) with the number of modules printed above each bar
(counts span 21 to 24,337, so count bars would leave most TFs invisible).
b2 (bottom): % of module edges > 25 kb on a loop, observed (ink dot) vs the TF's own null (grey box = IQR, line = median,
whiskers = 2.5-97.5%), same style as the right half of figD1. Null = 10,000 same-chromosome shuffles per TF in which each
module is shifted as one block (keeps every edge's spacing; perm_tf_long.py, hic/work_wt293t/loops/
ourrun_tf_loops_0.7_pad30kb_perm10000/null, column 2 = module-edge rate). TFs with >= 100 module edges > 25 kb;
one-sided empirical p from s1d_per_tf_null.tsv, BH q recomputed across the TFs shown. TF order (by module edges) shared."""
import glob
import numpy as np, pandas as pd
from fig3common import P, INP, XULAB, fig_mm, mm_axes, save, plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

NULLD = XULAB / "hic/work_wt293t/loops/ourrun_tf_loops_0.7_pad30kb_perm10000/null"
MIN_EDGES = 100
N = P.neutral
T = pd.read_csv(INP / "s1d_per_tf_null.tsv", sep="\t").set_index("tf")
T = T[T.edges >= MIN_EDGES].sort_values("edges")
p = T.edge_p.values; m = len(p); o = np.argsort(p); q = np.empty(m); q[o] = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]; T["q"] = np.minimum(q, 1)
null = {tf: np.vstack([np.load(f)["null"] for f in sorted(glob.glob(str(NULLD / f"{tf}_*.npz")))])[:, 1] * 100 for tf in T.index}
assert all(len(v) == 10000 for v in null.values())
assert np.allclose([np.median(null[t]) for t in T.index], T.edge_null_med * 100)
C = pd.read_csv(INP / "clique_loop_coverage.tsv", sep="\t")
labs = ["None", ">0–25%", "25–50%", "50–75%", "75–<100%", "100%"]
cv = C.coverage
C["stratum"] = np.select([C.loop_edges == 0, cv <= .25, cv <= .5, cv <= .75, cv < 1], labs[:5], labs[5])
Pt = C.groupby(["tf", "stratum"]).size().unstack(fill_value=0).reindex(index=T.index, columns=labs, fill_value=0)
chk = C.groupby("tf").edges_gt25kb.sum().reindex(T.index)
print("module edges > 25 kb, coverage table vs null table equal:", bool((chk.values == T.edges.values).all()))

W, H = 179.0, 50.0; X0, PW = 14.0, 126.0
fig = fig_mm(W, H)
a1 = mm_axes(fig, W, H, X0, 29.5, PW, 12.5); a2 = mm_axes(fig, W, H, X0, 12.0, PW, 16.5)
x = np.arange(len(T)); bot = np.zeros(len(T)); tot = Pt.sum(1).values
for l, c in zip(labs, P.loop_strata):
    v = Pt[l].values / tot * 100; a1.bar(x, v, bottom=bot, color=c, width=0.72, lw=0); bot += v
for i in x: a1.text(i, 102, f"{tot[i]:,}", rotation=90, ha="center", va="bottom", fontsize=6, color=N.dark)
a1.text(-0.06, 1.02, "n", transform=a1.transAxes, ha="right", va="bottom", fontsize=6, color=N.dark, fontstyle="italic")
a1.set_xlim(-0.6, len(T) - 0.4); a1.set_ylim(0, 100); a1.set_xticks(x, []); a1.tick_params(axis="x", length=0)
a1.set_yticks([0, 50, 100]); a1.set_ylabel("% modules", labelpad=2)
for i, tf in enumerate(T.index):
    v = null[tf]; lo, q1, md, q3, hi = np.percentile(v, [2.5, 25, 50, 75, 97.5]); ob = T.edge_obs[tf] * 100
    a2.plot([i, i], [lo, hi], color=N.mid, lw=0.6, solid_capstyle="butt")
    a2.add_patch(Rectangle((i - 0.3, q1), 0.6, q3 - q1, color=N.light, lw=0))
    a2.plot([i - 0.3, i + 0.3], [md, md], color=N.dark, lw=0.6)
    sig = T.q[tf] < 0.05
    a2.scatter([i], [ob], s=7, color=N.ink if sig else "white", edgecolors=N.ink, linewidths=0 if sig else 0.5, zorder=3)
a2.set_xlim(-0.6, len(T) - 0.4); a2.set_ylim(0, 35); a2.set_yticks([0, 10, 20, 30])
a2.yaxis.grid(True, color=N.faint, lw=0.5); a2.set_axisbelow(True)
a2.set_ylabel("% on a loop", labelpad=2)
a2.set_xticks(x, T.index, rotation=90, fontsize=6); a2.tick_params(axis="x", length=1.5, pad=1)
lg = fig.add_axes([0, 0, 1, 1]); lg.set_xlim(0, W); lg.set_ylim(0, H); lg.set_axis_off(); lg.patch.set_alpha(0)
lx = X0 + PW + 4.0; y = 46.5
lg.text(lx, y, "Module edges on a 0.7 loop", fontsize=6, va="center"); y -= 3.0
for k, (l, c) in enumerate(zip(labs, P.loop_strata)):
    xx, yy = lx + (k % 2) * 17.0, y - (k // 2) * 2.8
    lg.add_patch(Rectangle((xx, yy - 0.9), 2.2, 1.8, color=c, lw=0)); lg.text(xx + 3.0, yy, l, fontsize=6, va="center")
y = 25.0
lg.scatter([lx + 1.1], [y], s=7, color=N.ink, lw=0); lg.text(lx + 3.0, y, "Observed, q < 0.05", fontsize=6, va="center"); y -= 2.8
lg.scatter([lx + 1.1], [y], s=7, color="white", edgecolors=N.ink, linewidths=0.5); lg.text(lx + 3.0, y, "Observed, n.s.", fontsize=6, va="center"); y -= 2.8
lg.add_patch(Rectangle((lx + 0.5, y - 0.9), 1.2, 1.8, color=N.light, lw=0)); lg.add_line(Line2D([lx + 0.5, lx + 1.7], [y, y], color=N.dark, lw=0.6))
lg.text(lx + 3.0, y, "Own shuffle null", fontsize=6, va="center"); y -= 2.6
lg.text(lx, y, "box = IQR, whiskers = 95%", fontsize=6, va="top", color=N.dark); y -= 2.6
lg.text(lx, y, "10,000 shuffles, BH q", fontsize=6, va="top", color=N.dark)
ns = int((T.q < 0.05).sum())
print(f"TFs shown {len(T)}; above null at q < 0.05: {ns}; median observed {np.median(T.edge_obs) * 100:.1f}%, median null {np.median(T.edge_null_med) * 100:.1f}% "
      f"(range of null medians {T.edge_null_med.min() * 100:.1f}-{T.edge_null_med.max() * 100:.1f}%)")
print("n.s.:", T.index[T.q >= 0.05].tolist())
T.assign(edge_obs_pct=T.edge_obs * 100, edge_null_med_pct=T.edge_null_med * 100)[["edges", "edge_obs_pct", "edge_null_med_pct", "edge_p", "q"]].to_csv(INP.parent / "panels/fig3b_module_edge_null.tsv", sep="\t")
save(fig, "fig3b_hic_support")
