"""S1D pairs (palette C), left panel as stacked bars on a linear axis; bars longer than the axis are shrunk to it (marked //), keeping proportions."""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "palette"))
import regulome_palette as rp
P = rp.use()
HERE = Path(__file__).resolve().parent
NULL = HERE / "null_s1d_perm10000/null"
OUT = ROOT / "figures_paletteC"
MM = 1 / 25.4
N = P.neutral; RED = P.evidence.hic_loop
T = pd.read_csv(HERE / "s1d_per_tf_null.tsv", sep="\t").set_index("tf")
W = T[T.pairs >= 500].sort_values("pairs")
null = {tf: np.vstack([np.load(f)["null"] for f in sorted(glob.glob(str(NULL / f"{tf}_*.npz")))]) for tf in W.index}
y = np.arange(len(W))
fig = plt.figure(figsize=(150 * MM, (2.6 * len(W) + 18) * MM))
gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.05)
a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1], sharey=a)
cap = 25000
scale = np.minimum(1.0, cap / W.pairs.values)
sup = (W.pair_obs * W.pairs).round()
a.barh(y, W.pairs * scale, color=N.pale, height=0.75, label="pairs >25 kb")
a.barh(y, sup * scale, color=RED, height=0.75, label="… on a 0.7 Hi-C loop (±30 kb)")
for i, (n, o) in enumerate(zip(W.pairs, W.pair_obs)):
    if n > cap:
        a.text(cap * 1.01, i, f"// {n:,} ({100 * o:.1f}%)", va="center", fontsize=5, color=N.dark)
    else:
        a.text(n + cap * 0.01, i, f"{100 * o:.1f}%", va="center", fontsize=5, color=N.dark)
a.set_yticks(y); a.set_yticklabels(W.index, fontsize=5.5); a.set_xlabel("co-binding pairs >25 kb")
a.set_xlim(0, cap * 1.3)
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
b.set_xlim(x0, x1); b.set_xlabel("% of pairs on a loop")
b.set_title("Observed vs own null (10,000 shuffles)", loc="left")
b.xaxis.grid(True, color=N.faint, lw=0.5); b.set_axisbelow(True)
b.tick_params(axis="y", left=False, labelleft=False)
for ext in ("svg", "png"): fig.savefig(OUT / f"figD1_pairs_loop_support_paletteC_linear.{ext}", bbox_inches="tight")
plt.close(fig)
