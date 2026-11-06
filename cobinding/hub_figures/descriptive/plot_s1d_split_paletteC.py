"""S1D (palette C), three figures: (1) pairs per TF with loop-supported share + per-TF null distribution,
(2) cliques per TF stacked by loop-edge coverage, (3) RBBP4 permutation distributions."""
import glob, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "palette"))
import regulome_palette as rp
P = rp.use()
HERE = Path(__file__).resolve().parent
NULL = HERE / "null_s1d_perm10000/null"   # copy of hic/work_wt293t/loops/ourrun_tf_loops_0.7_pad30kb_perm10000/null
OUT = ROOT / "figures_paletteC"; OUT.mkdir(exist_ok=True)
MM = 1 / 25.4
N = P.neutral; RED = P.evidence.hic_loop
T = pd.read_csv(HERE / "s1d_per_tf_null.tsv", sep="\t").set_index("tf")
null = {tf: np.vstack([np.load(f)["null"] for f in sorted(glob.glob(str(NULL / f"{tf}_*.npz")))]) for tf in T.index}
# ---------- (1) pairs ----------
W = T[T.pairs >= 500].sort_values("pairs")
y = np.arange(len(W))
fig = plt.figure(figsize=(150 * MM, (2.6 * len(W) + 18) * MM))
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
for ext in ("svg", "png"): fig.savefig(OUT / f"figD1_pairs_loop_support_paletteC.{ext}", bbox_inches="tight")
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
fig, ax = plt.subplots(figsize=(110 * MM, (2.6 * len(Pt) + 18) * MM))
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
ax.set_xlabel("cliques (with ≥1 edge >25 kb)"); ax.legend(title="share of clique edges on a 0.7 loop", loc="lower right", title_fontsize=6)
ax.set_title("Cliques by Hi-C loop-edge coverage (% = cliques with ≥1 loop edge)", loc="left")
for ext in ("svg", "png"): fig.savefig(OUT / f"figD2_cliques_loop_coverage_paletteC.{ext}", bbox_inches="tight")
plt.close(fig)
old = pd.read_csv(HERE / "clique_loop_coverage_strata.tsv", sep="\t", index_col=0)
print("D2 strata identical to existing table:", old.equals(Pt.assign(total=Pt.sum(1))))
# ---------- (3) RBBP4 ----------
r = T.loc["RBBP4"]; nr = null["RBBP4"] * 100
fig, axs = plt.subplots(1, 2, figsize=(130 * MM, 48 * MM), gridspec_kw={"wspace": 0.3})
for ax, k, nm, n in ((axs[0], 0, "pair", int(r.pairs)), (axs[1], 1, "edge", int(r.edges))):
    x = nr[:, k]; o = r[f"{nm}_obs"] * 100
    step = 100 / n
    ax.hist(x, bins=np.arange(x.min() - step / 2, x.max() + step, step) if nm == "edge" else 60, color=N.mid)
    ax.axvline(o, color=N.ink, lw=1.2)
    ax.axvline(np.median(x), color=N.dark, ls="--", lw=0.6)
    ax.text(o, ax.get_ylim()[1] * 0.78, f"observed  {o:.1f}%", color=N.ink, fontsize=6, ha="right" if o > x.max() else "left")
    ax.text(np.median(x) + 0.3, ax.get_ylim()[1] * 0.98, f"null median {np.median(x):.1f}%\n(95%: {np.percentile(x, 2.5):.1f}–{np.percentile(x, 97.5):.1f}%)", fontsize=5.5, ha="left", va="top", color=N.dark)
    ax.set_xlabel(f"% of {'pairs' if nm == 'pair' else 'clique edges'} >25 kb on a 0.7 loop")
    ax.set_ylabel("shuffles")
    ax.set_title(f"RBBP4 {'pairs' if nm == 'pair' else 'clique edges'} >25 kb (n = {n:,})\np < 1e-4: 0 of 10,000 shuffles ≥ observed", loc="left")
    ax.set_xlim(min(x.min(), o) - 1, o + 2)
for ext in ("svg", "png"): fig.savefig(OUT / f"figD3_RBBP4_permutation_paletteC.{ext}", bbox_inches="tight")
print("RBBP4 pair obs", round(r.pair_obs * 100, 2), "null med", round(np.median(nr[:, 0]), 2), "| edge obs", round(r.edge_obs * 100, 2), "null med", round(np.median(nr[:, 1]), 2))
print("null medians match table:", np.allclose([np.median(null[t][:, 0]) for t in T.index], T.pair_null_med.values))
