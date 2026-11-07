"""Fig 3f (82 x 76 mm): TFs grouped by shared higher-order co-binding. TF x TF sharing z of module regions against the
curveball null that keeps each TF's depth and each region's TF count, run within library strata (cobinding/tf_groups,
null_z.py / cluster_groups.py; tag region_all_strat; same data as tf_groups_figs/fig_groups_cliques_region_all_strat.png
from scripts/plot_groups.py). Order = Leiden group (G1-G3, then ungrouped), depth within group; z clipped at +-10, palette C
diverging map; diagonal blank. Strips: library stratum and log10 module-region count (the null's depth). Row labels
alternate left/right so all 46 TFs are named at 6 pt; columns follow the same order."""
import numpy as np, pandas as pd
import matplotlib
from matplotlib.patches import Rectangle
from fig3common import P, XULAB, fig_mm, mm_axes, save, plt

IN = XULAB / "cobinding/tf_groups/results_cliques"; GT = NT = "region_all_strat"
N = P.neutral
G = pd.read_csv(IN / f"groups_{GT}.tsv", sep="\t"); G["g"] = G.group.replace(0, 99); G = G.sort_values(["g", "depth"], ascending=[True, False])
tf = G.tf.tolist(); n = len(tf)
z = pd.read_csv(IN / f"null_{NT}_z.tsv.gz", sep="\t", index_col=0).loc[tf, tf].values.copy(); np.fill_diagonal(z, np.nan)
lib = pd.read_csv(IN / "library_strata.tsv", sep="\t", header=None, index_col=0)[1].reindex(tf)
W, H = 82.0, 76.0; HX, S = 12.5, 55.0; HY = 9.5
fig = fig_mm(W, H)
ax = mm_axes(fig, W, H, HX, HY, S, S); V = 10
im = ax.imshow(np.clip(z, -V, V), cmap=P.div_cmap(), vmin=-V, vmax=V, interpolation="nearest", aspect="auto")
ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values(): s_.set_linewidth(0.5)
gv = G.g.values; br = np.where(np.diff(gv))[0]
for x in br:
    ax.axhline(x + .5, color=N.ink, lw=0.3); ax.axvline(x + .5, color=N.ink, lw=0.3)
starts = np.r_[0, br + 1]; ends = np.r_[br + 1, n]
for s0, e0 in zip(starts, ends):
    if gv[s0] != 99: ax.add_patch(Rectangle((s0 - .5, s0 - .5), e0 - s0, e0 - s0, fill=False, ec=N.ink, lw=0.9, zorder=3))
for k, t in enumerate(tf):   # alternate sides
    left = k % 2 == 0
    ax.text(-0.8 if left else n - 0.2, k, t, ha="right" if left else "left", va="center", fontsize=6)
# strips
top = HY + S
axl = mm_axes(fig, W, H, HX, top + 0.6, S, 1.6); col = {"low": N.pale, "mid": N.light, "high": N.dark}
axl.imshow([[matplotlib.colors.to_rgb(col.get(lib[t], "white")) for t in tf]], aspect="auto", interpolation="nearest"); axl.set_axis_off()
axl.text(-0.8, 0, "Library", ha="right", va="center", fontsize=6)
axd = mm_axes(fig, W, H, HX, top + 2.8, S, 5.0); axd.bar(range(n), np.log10(G.depth.values), color=N.mid, width=0.8, lw=0)
axd.set_xlim(-.5, n - .5); axd.set_ylim(0, 5); axd.set_xticks([]); axd.set_yticks([0, 5], ["0", "5"]); axd.tick_params(axis="y", length=1.2, pad=0.8, left=False, labelleft=False, right=True, labelright=True)
axd.spines["bottom"].set_visible(False); axd.spines["left"].set_visible(False); axd.spines["right"].set_visible(True)
axd.text(-0.015, 0.5, "log10\nregions", transform=axd.transAxes, ha="right", va="center", fontsize=6, linespacing=0.9)
for s0, e0 in zip(starts, ends):
    if gv[s0] != 99: axd.text((s0 + e0 - 1) / 2, 5.3, f"G{gv[s0]}", ha="center", va="bottom", fontsize=6, fontweight="bold")
# colour bar + library key
cax = mm_axes(fig, W, H, HX, 5.3, 26.0, 1.4)
cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-10, -5, 0, 5, 10]); cb.outline.set_linewidth(0.4)
cb.ax.set_xticklabels(["≤−10", "−5", "0", "5", "≥10"]); cb.ax.tick_params(length=1.2, pad=0.8, labelsize=6)
fig.text(HX / W, 0.4 / H, "Sharing z vs depth- and popularity-matched null", fontsize=6, va="bottom", ha="left")
kx = HX + 33.0
fig.text(kx / W, 6.0 / H, "Library", fontsize=6, va="center")
for j, (k_, nm) in enumerate((("low", "low"), ("mid", "mid"), ("high", "high"))):
    x_ = kx + 9.0 + j * 8.6
    fig.add_artist(Rectangle((x_ / W, 5.2 / H), 1.8 / W, 1.6 / H, transform=fig.transFigure, color=col[k_], lw=0))
    fig.text((x_ + 2.4) / W, 6.0 / H, nm, fontsize=6, va="center")
print("TFs", n, "groups", G.g.value_counts().sort_index().to_dict())
save(fig, "fig3f_tf_groups")
