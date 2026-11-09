"""Fig 3b v5 (179 x 34 mm): Hi-C support of higher-order co-binding modules per TF, one plot.
Per TF (>= 100 module edges > 25 kb): observed % of module edges on a WT Peakachu 0.7 loop (5 kb U 10 kb, +-30 kb; ink dot,
open = BH q >= 0.05) vs the TF's own null (grey bar = median, thin line = IQR). Null = 10,000 same-chromosome shuffles
per TF, each module shifted as one block (perm_tf_long.py, hic/work_wt293t/loops/ourrun_tf_loops_0.7_pad30kb_perm10000/null,
column 2 = module-edge rate); one-sided empirical p from s1d_per_tf_null.tsv, BH q across the TFs shown.
TFs sorted by observed minus null median (largest first)."""
import glob
import numpy as np, pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from fig3common import INP, XULAB, INK, GREY, LIGHT, PALE, FS, fig_mm, mm_axes, save, plt

NULLD = XULAB / "hic/work_wt293t/loops/ourrun_tf_loops_0.7_pad30kb_perm10000/null"
T = pd.read_csv(INP / "s1d_per_tf_null.tsv", sep="\t").set_index("tf"); T = T[T.edges >= 100]
p = T.edge_p.values; m = len(p); o = np.argsort(p); q = np.empty(m); q[o] = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]; T["q"] = np.minimum(q, 1)
null = {tf: np.vstack([np.load(f)["null"] for f in sorted(glob.glob(str(NULLD / f"{tf}_*.npz")))])[:, 1] * 100 for tf in T.index}
assert all(len(v) == 10000 for v in null.values())
T["obs"] = T.edge_obs * 100; T["med"] = [np.median(null[t]) for t in T.index]
T["q1"] = [np.percentile(null[t], 25) for t in T.index]; T["q3"] = [np.percentile(null[t], 75) for t in T.index]
assert np.allclose(T.med, T.edge_null_med * 100)
T = T.assign(d=T.obs - T.med).sort_values("d", ascending=False)

W, H = 179.0, 34.0; X0, X1, Y0, Y1 = 11.0, 178.0, 11.5, 31.0
fig = fig_mm(W, H); ax = mm_axes(fig, W, H, X0, Y0, X1 - X0, Y1 - Y0)
x = np.arange(len(T))
ax.bar(x, T.med, width=0.62, color=PALE, lw=0, zorder=1)
ax.vlines(x, T.q1, T.q3, color=GREY, lw=0.6, zorder=2)
sig = (T.q < 0.05).values
ax.scatter(x[sig], T.obs[sig], s=8, color=INK, lw=0, zorder=3)
ax.scatter(x[~sig], T.obs[~sig], s=8, color="white", edgecolors=INK, linewidths=0.5, zorder=3)
ax.set_xlim(-0.7, len(T) - 0.3); ax.set_ylim(0, 38); ax.set_yticks([0, 10, 20, 30])
ax.yaxis.grid(True, color="#F2F2F2", lw=0.5); ax.set_axisbelow(True)
ax.set_ylabel("% module edges\non a Hi-C loop", fontsize=FS, labelpad=2, linespacing=1.0)
ax.set_xticks(x, T.index, rotation=90, fontsize=FS); ax.tick_params(axis="x", length=1.5, pad=1)
ns = int(sig.sum())
ax.text(-0.4, 37.6, f"{ns}/{len(T)} TFs above null (q < 0.05)", fontsize=FS, va="top", ha="left", color=INK)
# compact one-row key, top right inside the axes
lg = fig.add_axes([0, 0, 1, 1]); lg.set_xlim(0, W); lg.set_ylim(0, H); lg.set_axis_off(); lg.patch.set_alpha(0)
ky = Y1 - 1.1; kx = 126.0
lg.scatter([kx], [ky], s=8, color=INK, lw=0); lg.text(kx + 1.3, ky, "Observed", fontsize=FS, va="center")
kx += 13.0; lg.scatter([kx], [ky], s=8, color="white", edgecolors=INK, linewidths=0.5); lg.text(kx + 1.3, ky, "n.s.", fontsize=FS, va="center")
kx += 8.0; lg.add_patch(Rectangle((kx - 0.7, ky - 1.0), 1.4, 2.0, color=PALE, lw=0)); lg.add_line(Line2D([kx, kx], [ky - 0.7, ky + 0.7], color=GREY, lw=0.6))
lg.text(kx + 1.6, ky, "Shuffle null (median, IQR)", fontsize=FS, va="center")
print(f"TFs {len(T)}; q < 0.05: {ns}; n.s.: {T.index[~sig].tolist()}; median obs {T.obs.median():.1f}% vs null {T.med.median():.1f}%")
T[["edges", "obs", "med", "q1", "q3", "edge_p", "q"]].to_csv(INP.parent / "panels/fig3b_module_edge_null.tsv", sep="\t")
save(fig, "fig3b_hic_support")
