"""Step 4: figure of the best PASTIS model of chr16p11.2 (3 views + quant panel) and an interactive plotly HTML."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from scipy.interpolate import splprep, splev

ap = argparse.ArgumentParser()
ap.add_argument("--prep", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n-labels", type=int, default=5)
ap.add_argument("--label-min-sep", type=int, default=150000, help="min genomic separation between labelled cores")
ap.add_argument("--views", default="20,-60;20,30;80,-90")
a = ap.parse_args()
P0 = Path(a.prep); P = Path(a.model); OUT = Path(a.out)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42, "svg.fonttype": "none"})
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7"]

bins = pd.read_csv(P0 / "bins.tsv", sep="\t")
B = pd.read_csv(P / "best_model.tsv", sep="\t"); X = B[["x", "y", "z"]].values; interp = B.interpolated.values
E = pd.read_csv(P0 / "shared_edges.tsv", sep="\t")
Lp = pd.read_csv(P0 / "loops.tsv", sep="\t")
qp = pd.read_csv(P / "quant_pairs_best.tsv", sep="\t")
Q = pd.read_csv(P / "quant_check.tsv", sep="\t")
nulls = np.load(P / "quant_null_best.npz")
qc = pd.read_csv(P / "model_qc.tsv", sep="\t", header=None, index_col=0)[1]
ntf = bins.n_tfs.values; nmax = max(int(ntf.max()), 1)
cmap = plt.get_cmap("Purples"); norm = colors.Normalize(vmin=0, vmax=nmax)
chrom = bins.chrom.iloc[0]; L0, L1 = bins.start.iloc[0], bins.end.iloc[len(bins) - 1]

# smooth backbone (spline through all beads), oversampled 8x
# (an end bin interpolated by clamping duplicates its neighbour; skip zero-length steps for the spline fit)
step = np.r_[True, np.linalg.norm(np.diff(X, axis=0), axis=1) > 1e-9]
u = np.arange(len(X)) / (len(X) - 1)
tck, _ = splprep(X[step].T, u=u[step], s=step.sum() * 0.05, k=3)
uu = np.linspace(0, u[step][-1], len(X) * 8); S = np.array(splev(uu, tck)).T
seg_interp = np.interp(uu, u, interp.astype(float)) > 0.5

# labels: hub cores with a promoter gene; keep the top N by TF count, at most one per gene
hub = bins[bins.hub_core.notna() & (bins.hub_core.astype(str) != "") & (bins.hub_core.astype(str) != "nan")]
lab = hub[hub.hub_label.notna() & (hub.hub_label.astype(str) != "nan") & (hub.hub_label.astype(str) != "") & ~hub.hub_label.astype(str).str.contains("?", regex=False)]
lab = lab.sort_values("hub_core_ntfs", ascending=False).drop_duplicates("hub_label")
pick = []
for i, r in lab.iterrows():
    if all(abs(r.start - lab.at[j, "start"]) >= a.label_min_sep for j in pick):
        pick.append(i)
    if len(pick) == a.n_labels: break
lab = lab.loc[pick]
centre = X.mean(0)

def draw(ax, elev, azim, legend=False):
    # backbone: solid where fitted, dashed where interpolated
    for flag, ls in ((False, "-"), (True, (0, (2, 2)))):
        Sm = S.copy(); Sm[seg_interp != flag] = np.nan
        ax.plot(*Sm.T, color="#C8C8C8", lw=3.2 if not flag else 1.6, ls=ls, solid_capstyle="round", zorder=1)
    for r in Lp.itertuples():
        p = X[[r.bin1, r.bin2]]
        ax.plot(*p.T, color="#8C8C8C", lw=0.35, ls=(0, (3, 2)), alpha=0.6, zorder=2)
    for r in E.itertuples():
        p = X[[r.bin1, r.bin2]]
        ax.plot(*p.T, color=OKABE[r.clique_group % 4], lw=0.6 + 0.35 * r.n_tfs, alpha=0.9, zorder=3)
    s = 4 + 60 * (ntf / nmax) ** 1.2
    o = ntf == 0
    ax.scatter(*X[o].T, s=3, c="#BDBDBD", depthshade=False, linewidths=0, zorder=4)
    hb = bins.index.isin(hub.index)
    ax.scatter(*X[~o & ~hb].T, s=s[~o & ~hb], c=cmap(norm(ntf[~o & ~hb])), depthshade=False, linewidths=0.2, edgecolors="white", zorder=5)
    ax.scatter(*X[hb].T, s=s[hb], c=cmap(norm(ntf[hb])), depthshade=False, linewidths=0.8, edgecolors="#111111", zorder=6)
    lim = np.percentile(np.abs(X - centre), 99, axis=0).max() * 1.05
    for r in lab.itertuples():
        p0 = X[r.Index]; v = p0 - centre; v = v / (np.linalg.norm(v) + 1e-9)
        p1 = p0 + v * lim * 0.32
        ax.plot(*np.c_[p0, p1], color="#444444", lw=0.4, zorder=7)
        ax.text(*p1, r.hub_label, fontsize=6, fontstyle="italic", color="#111111", zorder=8, ha="center", va="center")
    ax.view_init(elev=elev, azim=azim)
    c = centre
    ax.set_xlim(c[0] - lim, c[0] + lim); ax.set_ylim(c[1] - lim, c[1] + lim); ax.set_zlim(c[2] - lim, c[2] + lim)
    ax.set_box_aspect((1, 1, 1), zoom=1.5); ax.set_axis_off()
    ax.text2D(0.5, 0.97, f"view: elev {elev}°, azim {azim}°", transform=ax.transAxes, fontsize=6, color="#777777", ha="center")

views = [tuple(int(v) for v in s.split(",")) for s in a.views.split(";")]
fig = plt.figure(figsize=(7.2, 5.6))
axs = [fig.add_axes([0.0 + 0.333 * k, 0.37, 0.333, 0.57], projection="3d") for k in range(3)]
for ax, (el, az) in zip(axs, views):
    draw(ax, el, az)
fig.text(0.01, 0.975, f"chr16p11.2 ({chrom}:{L0/1e6:.2f}–{L1/1e6:.2f} Mb, hg38), 10 kb PASTIS PM2 model, WT HEK293T Hi-C", fontsize=8, weight="bold", va="top")
fig.text(0.01, 0.94, "Consensus model from population Hi-C; not a single-cell structure. Arbitrary 3D units.", fontsize=6.5, color="#555555", va="top")

# colourbar for TF count
cax = fig.add_axes([0.03, 0.325, 0.25, 0.016])
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
cb.set_label("TFs with a MACS2 peak (q<0.05) in bin", fontsize=6.5); cb.outline.set_linewidth(0.4); cax.tick_params(length=2, width=0.4)

# legend
handles = [Line2D([], [], color="#C8C8C8", lw=3, label="chromatin backbone (10 kb beads)"),
           Line2D([], [], color="#C8C8C8", lw=1.5, ls=(0, (2, 2)), label="low-coverage bins (interpolated)"),
           Line2D([], [], marker="o", ls="", mfc=cmap(0.6), mec="#111111", mew=1.1, ms=6, label="promoter-hub core (≥5 TFs' cliques)"),
           Line2D([], [], color="#8C8C8C", lw=0.6, ls=(0, (3, 2)), label="Peakachu loop (score ≥0.7, 5 ∪ 10 kb)")]
ng = E.clique_group.nunique()
for g in range(min(ng, 4)):
    handles.append(Line2D([], [], color=OKABE[g], lw=1.8, label="shared clique edges, groups " + ", ".join(str(k + 1) for k in range(g, min(ng, g + 9), 4)) + (", …" if g + 8 < ng else "")))
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.355, 0.35), frameon=False, handlelength=2.4, borderaxespad=0, labelspacing=0.45)
fig.text(0.03, 0.245, f"Shared edges: Cicero step-1 clique edges\n>25 kb shared by ≥2 TFs (n={len(E)}); line width\n∝ no. of TFs. Clique group = edges linked\nthrough a common TF clique ({ng} groups;\n4 colours cycle). Outlined beads: anchor-hub\ncores in ≥5 TFs' cliques; labels = promoter\ngene of the top {len(lab)} cores by TF count.", fontsize=5.8, color="#555555", va="top", linespacing=1.35)

# quant panel
axq = fig.add_axes([0.75, 0.085, 0.235, 0.255])
order = [("shared_clique_edges", "Shared\nedges"), ("cicero_step1_pairs", "Cicero\npairs"), ("peakachu_0.7_loops", "Loops\n(0.7)")]
data = [np.log2(qp[qp.set == k].ratio.values) for k, _ in order]
null = np.log2(np.concatenate([nulls[k] for k, _ in order]))
null = np.random.RandomState(0).choice(null, 5000, replace=False)
pos = [1, 2, 3, 4]
vp = axq.violinplot(data + [null], positions=pos, widths=0.75, showextrema=False)
fc = ["#6A51A3", "#9E9AC8", "#7F7F7F", "#D9D9D9"]
for b, c in zip(vp["bodies"], fc):
    b.set_facecolor(c); b.set_edgecolor("none"); b.set_alpha(0.85)
axq.boxplot(data + [null], positions=pos, widths=0.18, showfliers=False, medianprops=dict(color="black", lw=0.8),
            boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5), capprops=dict(lw=0))
axq.axhline(0, color="#555555", lw=0.5, ls=":")
qb = Q[Q.model_rank == 1].set_index("set")
for x, (k, _) in zip(pos, order):
    r = qb.loc[k]
    pt = f"p<{1/1001:.3f}".replace("0.001", "0.001") if r.p_one_sided <= 1.5 / 1001 else f"p={r.p_one_sided:.2g}"
    axq.text(x, 1.0, f"n={int(r.n_pairs)}\n{pt}", transform=axq.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.5)
axq.set_xticks(pos); axq.set_xticklabels([l for _, l in order] + ["matched\nrandom"])
axq.set_ylabel("log2 3D distance /\nmatched-pair median", fontsize=6.5)
axq.spines[["top", "right"]].set_visible(False); axq.spines[["left", "bottom"]].set_linewidth(0.5); axq.tick_params(width=0.5, length=2)
axq.set_title("3D proximity vs genomic-distance-\nmatched random bin pairs", fontsize=7, pad=17)
axq.set_xticklabels(axq.get_xticklabels(), fontsize=6)
nruns = len(pd.read_csv(P / "runs.tsv", sep="\t"))
fig.text(0.03, 0.025, f"Best of {nruns} random starts (max. likelihood); top-10 median pairwise RMSD = {float(qc['rmsd_top10_median']):.2f} bead steps\n"
         f"(Procrustes); Spearman(model distance, ICE contacts) = {float(qc['spearman_dist_vs_ICE_all_pairs']):.2f}. Grey beads: no MACS2 peak.", fontsize=5.8, color="#555555", linespacing=1.35)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}", dpi=300)
print("saved", OUT)

# interactive HTML
import plotly.graph_objects as go
tr = []
for flag, dash, w in ((False, "solid", 9), (True, "dot", 4)):
    Sm = S.copy(); Sm[seg_interp != flag] = np.nan
    tr.append(go.Scatter3d(x=Sm[:, 0], y=Sm[:, 1], z=Sm[:, 2], mode="lines", line=dict(color="#C8C8C8", width=w, dash=dash),
                           name="backbone" + (" (interpolated)" if flag else ""), hoverinfo="skip"))
def chords(df, color, width, name, dash="solid", show=True):
    xs, ys, zs, tx = [], [], [], []
    for r in df.itertuples():
        for b in (r.bin1, r.bin2):
            xs.append(X[b, 0]); ys.append(X[b, 1]); zs.append(X[b, 2])
        xs.append(None); ys.append(None); zs.append(None)
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=width, dash=dash), name=name, hoverinfo="skip", showlegend=show)
tr.append(chords(Lp, "#7F7F7F", 2, "Peakachu loop ≥0.7", dash="dash"))
for g in sorted(E.clique_group.unique()):
    tr.append(chords(E[E.clique_group == g], OKABE[g % 4], 5, f"shared clique edges, group {g+1}"))
hb = bins.index.isin(hub.index)
txt = [f"{chrom}:{s:,}-{e:,}<br>TFs: {n}<br>{t if isinstance(t, str) else ''}" + (f"<br><b>hub core: {h}</b>" if isinstance(h, str) and h else "")
       for s, e, n, t, h in zip(bins.start, bins.end, bins.n_tfs, bins.tfs, bins.hub_label)]
txt = np.array(txt)
for sel, nm, lc, lw, cbar in ((~hb, "10 kb bins", "#FFFFFF", 0.5, True), (hb, "promoter-hub cores", "#111111", 6, False)):
    tr.append(go.Scatter3d(x=X[sel, 0], y=X[sel, 1], z=X[sel, 2], mode="markers", name=nm,
                           marker=dict(size=2 + 10 * (ntf[sel] / nmax), color=ntf[sel], colorscale="Purples", cmin=0, cmax=nmax,
                                       line=dict(color=lc, width=lw), showscale=cbar,
                                       colorbar=dict(title="TFs (MACS2)", len=0.5, x=-0.05)),
                           text=txt[sel], hoverinfo="text"))
tr.append(go.Scatter3d(x=X[lab.index, 0], y=X[lab.index, 1], z=X[lab.index, 2], mode="text", text=lab.hub_label.tolist(),
                       textfont=dict(size=12, color="#111111"), name="promoter labels", hoverinfo="skip"))
f = go.Figure(tr)
f.update_layout(title=f"chr16p11.2 3D model (PASTIS PM2, WT HEK293T Hi-C 10 kb). Consensus model from population Hi-C; not a single-cell structure.",
                scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode="data"),
                font=dict(family="Arial, Helvetica, sans-serif", size=12), legend=dict(x=1.02, y=0.5), template="plotly_white")
f.write_html(f"{OUT}.html", include_plotlyjs=True)
print("saved html")
