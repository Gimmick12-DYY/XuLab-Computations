"""Fig 3f v5 (111 x 98 mm): TFs grouped by shared higher-order co-binding. TF x TF sharing z of module regions against
the curveball null that keeps each TF's depth and each region's TF count, run within library strata (cobinding/tf_groups,
null_z.py / cluster_groups.py; tag region_all_strat). Order = Leiden group (G1-G3, then ungrouped), depth within group;
z clipped at +-10, palette C diverging map; diagonal blank; square cells. All TF labels on the left in row order, groups
as brackets left of the labels. Library strip and depth bars moved to the supplement."""
import numpy as np, pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from fig3common import P, XULAB, INK, FS, fig_mm, mm_axes, save, plt

IN = XULAB / "cobinding/tf_groups/results_cliques"; TAG = "region_all_strat"
G = pd.read_csv(IN / f"groups_{TAG}.tsv", sep="\t"); G["g"] = G.group.replace(0, 99); G = G.sort_values(["g", "depth"], ascending=[True, False])
tf = G.tf.tolist(); n = len(tf)
z = pd.read_csv(IN / f"null_{TAG}_z.tsv.gz", sep="\t", index_col=0).loc[tf, tf].values.copy(); np.fill_diagonal(z, np.nan)
W, H = 111.0, 98.0; S = 90.0; HX, HY = W - S - 0.5, 7.0
fig = fig_mm(W, H); ax = mm_axes(fig, W, H, HX, HY, S, S); V = 10
im = ax.imshow(np.clip(z, -V, V), cmap=P.div_cmap(), vmin=-V, vmax=V, interpolation="nearest", aspect="equal")
ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values(): s_.set_visible(True); s_.set_linewidth(0.4)
gv = G.g.values; br = np.where(np.diff(gv))[0]
for x in br: ax.axhline(x + .5, color=INK, lw=0.3); ax.axvline(x + .5, color=INK, lw=0.3)
starts = np.r_[0, br + 1]; ends = np.r_[br + 1, n]
for s0, e0 in zip(starts, ends):
    if gv[s0] != 99: ax.add_patch(Rectangle((s0 - .5, s0 - .5), e0 - s0, e0 - s0, fill=False, ec=INK, lw=0.8, zorder=3))
for k, t in enumerate(tf): ax.text(-0.9, k, t, ha="right", va="center", fontsize=FS)
# group brackets left of the labels (mm coordinates on an overlay)
lg = fig.add_axes([0, 0, 1, 1]); lg.set_xlim(0, W); lg.set_ylim(0, H); lg.set_axis_off(); lg.patch.set_alpha(0)
row_y = lambda k: HY + S - (k + 0.5) * S / n
BX = HX - 12.6
for s0, e0 in zip(starts, ends):
    if gv[s0] == 99: continue
    y0, y1 = row_y(s0) + 0.8, row_y(e0 - 1) - 0.8
    lg.add_line(Line2D([BX + 0.6, BX, BX, BX + 0.6], [y0, y0, y1, y1], color=INK, lw=0.6))
    lg.text(BX - 0.6, (y0 + y1) / 2, f"G{gv[s0]}", ha="right", va="center", fontsize=FS, fontweight="bold")
cax = mm_axes(fig, W, H, HX + S - 24.0, 3.6, 22.0, 1.2)
cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-10, 0, 10]); cb.outline.set_linewidth(0.4)
cb.ax.set_xticklabels(["≤−10", "0", "≥10"]); cb.ax.tick_params(length=1.0, pad=0.6, labelsize=FS)
fig.text((HX + S - 25.5) / W, 4.2 / H, "Sharing z vs null", ha="right", va="center", fontsize=FS)
print("TFs", n, "groups", G.g.value_counts().sort_index().to_dict(), "cell mm", round(S / n, 2))
save(fig, "fig3f_tf_groups")
