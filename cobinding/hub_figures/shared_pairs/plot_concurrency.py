import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
R = pd.read_csv("pair_concurrency.tsv", sep="\t")
cells = pd.read_csv("../goodex/tf_cell_counts.tsv", sep="\t", header=None, index_col=0)[1]
fig = plt.figure(figsize=(17, 8.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.1], wspace=0.25)
a = fig.add_subplot(gs[0])
nx = R[~R.exonic]; ex = R[R.exonic]
rng = np.random.default_rng(0)
for d, c, l in ((nx, "#1F4E8C", f"no exonic end (n = {len(nx):,})"), (ex, "#E59866", f"≥1 exonic end (n = {len(ex):,})")):
    a.scatter(d.expected_tfs, d.n_tfs + rng.uniform(-0.2, 0.2, len(d)), s=8, alpha=0.5, color=c, label=l, lw=0)
sig = R.p_bonf < 0.05
a.scatter(R.expected_tfs[sig], R.n_tfs[sig], s=40, facecolor="none", edgecolor="#111", lw=1, label=f"Bonferroni p < 0.05 (n = {sig.sum()})")
a.plot([0, 1.3], [0, 1.3], color="#999", ls="--", lw=0.8); a.set_xlim(0, 1.3)
a.set_xlabel("expected TFs sharing the pair (independent TFs)"); a.set_ylabel("observed TFs sharing the pair")
a.set_title("Observed vs expected concurrency per pair", loc="left", fontsize=12)
a.legend(frameon=False, fontsize=9.5, loc="upper right")
for sp in ("top", "right"): a.spines[sp].set_visible(False)
# top non-exonic pairs by p
T = nx.sort_values("p").head(25).reset_index(drop=True)
tfs = pd.Series([t for x in T.tfs for t in x.split(",")]).value_counts()
rows = list(tfs.index)
b = fig.add_subplot(gs[1])
for j, r in T.iterrows():
    s = set(r.tfs.split(","))
    ys = [i for i, t in enumerate(rows) if t in s]
    b.scatter([j] * len(rows), range(len(rows)), s=22, color="#EEEEEE", zorder=1)
    b.scatter([j] * len(ys), ys, s=34, color="#1F4E8C", zorder=2)
def lab(reg, k, g):
    c, se = reg.split(":"); pos = int(se.split("-")[0])
    return (g if (k == "promoter" and isinstance(g, str) and g) else f"{c}:{pos / 1e6:.2f}M") + ("" if k == "promoter" else f" ({k[:5]})")
b.set_xticks(range(len(T)))
b.set_xticklabels([f"{lab(r.region1, r.region1_class, r.region1_gene)} – {lab(r.region2, r.region2_class, r.region2_gene)} · {r.distance_kb:g} kb · {r.n_tfs}/{r.expected_tfs:.2f}" for r in T.itertuples()],
                  rotation=90, fontsize=8.5)
b.set_yticks(range(len(rows))); b.set_yticklabels([f"{t} ({cells.get(t, 0) / 1e3:.0f}k cells)" for t in rows], fontsize=9)
b.invert_yaxis(); b.set_xlim(-0.6, len(T) - 0.4)
for sp in ("top", "right"): b.spines[sp].set_visible(False)
b.set_title("Top 25 most concurrent pairs without exonic ends (ranked by p; label: observed/expected TFs)", loc="left", fontsize=12)
fig.savefig("fig_pair_concurrency.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig("fig_pair_concurrency.svg", bbox_inches="tight", facecolor="white")
print(T[["region1_gene", "region2_gene", "n_tfs", "expected_tfs", "p_bonf"]].head(10).to_string())
