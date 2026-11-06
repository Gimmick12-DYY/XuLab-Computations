"""S1D with per-TF nulls: observed % of pairs / clique edges (>25 kb) on a 0.7 loop vs each TF's own
10,000 same-chromosome shuffles (median tick, 2.5–97.5% range), one-sided p, BH across TFs."""
import glob, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
D = Path(sys.argv[1]); MINP = int(sys.argv[2]) if len(sys.argv) > 2 else 500
rows = []
for tf in sorted({Path(f).name.rsplit("_", 1)[0] for f in glob.glob(str(D / "*.npz"))}):
    fs = sorted(glob.glob(str(D / f"{tf}_*.npz")))
    Z = [np.load(f) for f in fs]; null = np.vstack([z["null"] for z in Z]); obs = Z[0]["obs"]; n = Z[0]["n"]
    r = {"tf": tf, "chunks": len(fs), "n_perm": len(null), "pairs": int(n[0]), "edges": int(n[1])}
    for k, nm in enumerate(("pair", "edge")):
        x = null[:, k]; x = x[np.isfinite(x)]
        r[f"{nm}_obs"] = obs[k]; r[f"{nm}_null_med"] = np.median(x) if len(x) else np.nan
        r[f"{nm}_null_lo"] = np.percentile(x, 2.5) if len(x) else np.nan; r[f"{nm}_null_hi"] = np.percentile(x, 97.5) if len(x) else np.nan
        r[f"{nm}_p"] = (1 + (x >= obs[k] - 1e-15).sum()) / (1 + len(x)) if len(x) else np.nan
    rows.append(r)
T = pd.DataFrame(rows)
def bh(p):
    p = np.asarray(p, float); ok = np.isfinite(p); q = np.full(len(p), np.nan); pv = p[ok]; o = np.argsort(pv); m = len(pv)
    qq = pv[o] * m / np.arange(1, m + 1); qq = np.minimum.accumulate(qq[::-1])[::-1]; q[np.where(ok)[0][o]] = np.minimum(qq, 1); return q
T["pair_q"] = bh(T.pair_p); T["edge_q"] = bh(T.edge_p.where(T.edges > 0))
T.to_csv("s1d_per_tf_null.tsv", sep="\t", index=False)
W = T[T.pairs >= MINP].sort_values("pair_obs").reset_index(drop=True)
y = np.arange(len(W))
fig, axs = plt.subplots(1, 2, figsize=(13, 0.24 * len(W) + 1.6), sharey=True, gridspec_kw={"wspace": 0.1})
for ax, nm, col, ttl in ((axs[0], "pair", "#2E7D6B", "Co-binding pairs >25 kb"), (axs[1], "edge", "#1F4E8C", "Clique edges >25 kb")):
    ax.barh(y, W[f"{nm}_obs"] * 100, color=col, height=0.75)
    ax.hlines(y, W[f"{nm}_null_lo"] * 100, W[f"{nm}_null_hi"] * 100, color="#C0392B", lw=1.2)
    ax.scatter(W[f"{nm}_null_med"] * 100, y, marker="|", s=90, color="#C0392B", zorder=3, label="own shuffle null: median, 95% range")
    for i, (o, q) in enumerate(zip(W[f"{nm}_obs"], W[f"{nm}_q"])):
        if np.isfinite(q) and q < 0.05: ax.text(o * 100 + 0.3, i, "*", va="center", fontsize=11)
    ax.set_title(ttl, loc="left", fontsize=12); ax.set_xlabel(f"% of {'pairs' if nm == 'pair' else 'clique edges'} on a 0.7 loop (±30 kb)")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
axs[0].set_yticks(y); axs[0].set_yticklabels([f"{t} ({n:,})" for t, n in zip(W.tf, W.pairs)], fontsize=9)
axs[1].legend(loc="lower right", frameon=False, fontsize=9.5)
fig.text(0.99, -0.01, "* BH-adjusted p < 0.05 (one-sided, 10,000 same-chromosome shuffles per TF)", ha="right", fontsize=9, color="#555")
fig.savefig("figD_hic_support_per_tf_null.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig("figD_hic_support_per_tf_null.svg", bbox_inches="tight", facecolor="white")
print(len(W), "TFs shown; pairs q<0.05:", (W.pair_q < 0.05).sum(), "; edges q<0.05:", (W.edge_q < 0.05).sum())
print(W[["tf", "pairs", "pair_obs", "pair_null_med", "pair_p", "edge_obs", "edge_null_med", "edge_q"]].round(4).to_string())
