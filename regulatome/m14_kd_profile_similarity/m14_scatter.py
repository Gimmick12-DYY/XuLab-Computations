# m14: B vs K scatter redrawn from out/pairs.tsv and out/mantel.tsv (labels de-overlapped). Diagnostic, benchmark-developed.
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import glob as _g, matplotlib.font_manager as _fm
for _f in _g.glob("/work/users/d/y/dyy12/fonts/arial/*.ttf"): _fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none", "font.size": 6, "axes.linewidth": .4})
O = "/work/users/d/y/dyy12/XuLab/regulatome/m14_kd_profile_similarity/out"; P = pd.read_csv(f"{O}/pairs.tsv", sep="\t"); M = pd.read_csv(f"{O}/mantel.tsv", sep="\t"); T = pd.read_csv(f"{O}/top20_pairs.tsv", sep="\t")
Q = P[P.K_all_genes.notna() & P.B_mean5.notna()]; ntf = len(set(Q.a) | set(Q.b)); m = M[(M.tf_set == "all76") & (M.K == "all_genes") & (M.B == "mean5")].iloc[0]
fig, ax = plt.subplots(figsize=(110 / 25.4, 90 / 25.4))
ax.scatter(Q.B_mean5, Q.K_all_genes, s=3, c="#BDBDBD", lw=0, label="TF pairs")
k = Q[Q.known | Q.named_control]; ax.scatter(k.B_mean5, k.K_all_genes, s=10, facecolors="none", edgecolors="#D55E00", lw=.6, label="known / named complex pairs")
a = Q[Q.af3_5seed == "strong"]; ax.scatter(a.B_mean5, a.K_all_genes, s=10, marker="^", c="#0072B2", lw=0, label="AF3 5-seed strong")
lab = T.sort_values("K_all_genes", ascending=False).head(6); ys = np.linspace(.27, .17, len(lab))
for (r, yt) in zip(lab.itertuples(), ys): ax.annotate(f"{r.a}-{r.b}", (r.B_mean5, r.K_all_genes), xytext=(.68, yt), textcoords="data", fontsize=5, ha="right", va="center", arrowprops=dict(arrowstyle="-", lw=.3, color="#777777"))
ax.axhline(0, color="k", lw=.3); ax.set_xlabel("binding similarity B (mean of five ChromHMM percentile matrices)"); ax.set_ylabel("knockdown-profile similarity K (cross-half Spearman)")
ax.set_title(f"{ntf} TFs, {len(Q):,} pairs; Mantel rho = {m.mantel_rho:.3f} (p = {m.p:.2f}); positive control failed", fontsize=6); ax.legend(frameon=False, fontsize=5, loc="lower left"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); [fig.savefig(f"{O}/scatter_B_vs_K.{e}", dpi=300) for e in ("png", "svg")]
