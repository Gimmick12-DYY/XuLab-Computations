"""Fig: TF x TF sharing z-score heatmap (curveball null), ordered by Leiden group then depth, with depth and library-stratum
bars; second panel = unstratified vs library-stratified z. args: indir groups_tag null_tag [null_tag_unstrat]"""
import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/figures_paletteC"); import regulome_palette as rp
from pathlib import Path
P = rp.use(); IN = Path(sys.argv[1]); GT, NT = sys.argv[2], sys.argv[3]; NT0 = sys.argv[4] if len(sys.argv) > 4 else None
G = pd.read_csv(IN / f"groups_{GT}.tsv", sep="\t"); G["g"] = G.group.replace(0, 99); G = G.sort_values(["g", "depth"], ascending=[True, False])
tf = G.tf.tolist(); z = pd.read_csv(IN / f"null_{NT}_z.tsv.gz", sep="\t", index_col=0).loc[tf, tf].values.copy(); np.fill_diagonal(z, np.nan)
lib = pd.read_csv(IN / "library_strata.tsv", sep="\t", header=None, index_col=0)[1].reindex(tf)
n = len(tf); fig = plt.figure(figsize=(7.2, 6.4))
ax = fig.add_axes([0.14, 0.1, 0.66, 0.74]); cm = rp.div_cmap(); v = 10
im = ax.imshow(np.clip(z, -v, v), cmap=cm, vmin=-v, vmax=v, interpolation="nearest")
ax.set_xticks(range(n)); ax.set_xticklabels(tf, rotation=90, fontsize=5); ax.set_yticks(range(n)); ax.set_yticklabels(tf, fontsize=5)
b = np.where(np.diff(G.g.values))[0]
for x in b: ax.axhline(x + .5, color=P.neutral.ink, lw=.6); ax.axvline(x + .5, color=P.neutral.ink, lw=.6)
for g, sub in G.groupby("g", sort=False):
    if g == 99: continue
    i = [tf.index(t) for t in sub.tf]; ax.text(n + 0.6, np.mean(i), f"G{g}", va="center", fontsize=6, fontweight="bold")
axd = fig.add_axes([0.14, 0.845, 0.66, 0.04]); axd.bar(range(n), np.log10(G.depth.values), color=P.neutral.mid, width=.8)
axd.set_xlim(-.5, n - .5); axd.set_xticks([]); axd.set_ylabel("log10\npairs", fontsize=5)
axl = fig.add_axes([0.14, 0.895, 0.66, 0.015]); col = {"low": P.neutral.pale, "mid": P.neutral.light, "high": P.neutral.dark}
axl.imshow([[matplotlib.colors.to_rgb(col.get(lib[t], "white")) for t in tf]], aspect="auto"); axl.set_axis_off()
axl.text(-1, 0, "library", ha="right", va="center", fontsize=5)
cax = fig.add_axes([0.9, 0.3, 0.015, 0.3]); fig.colorbar(im, cax=cax).set_label("sharing z (depth- and pair-popularity-matched null)", fontsize=5)
fig.suptitle(f"TF groups by shared Cicero co-binding ({NT})", fontsize=7, y=0.97)
fig.savefig(IN / f"fig_groups_{GT}.png", dpi=300); fig.savefig(IN / f"fig_groups_{GT}.svg")
if NT0:
    z0 = pd.read_csv(IN / f"null_{NT0}_z.tsv.gz", sep="\t", index_col=0).loc[tf, tf].values; iu = np.triu_indices(n, 1)
    same = (lib.values[:, None] == lib.values[None, :])[iu]
    f2, a2 = plt.subplots(figsize=(3, 3))
    a2.scatter(z0[iu][~same], z[iu][~same], s=3, color=P.arms.imputed, label="different library stratum", lw=0)
    a2.scatter(z0[iu][same], z[iu][same], s=3, color=P.arms.raw, label="same library stratum", lw=0)
    a2.axline((0, 0), slope=1, color=P.neutral.mid, lw=.5); a2.set_xlabel(f"z, {NT0}"); a2.set_ylabel(f"z, {NT}"); a2.legend()
    f2.tight_layout(); f2.savefig(IN / f"fig_z_strat_vs_unstrat_{GT}.png", dpi=300)
