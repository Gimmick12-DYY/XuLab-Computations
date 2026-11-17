"""Fig 3 panel d: shared Cicero pairs, module loop-edge bars and the per-edge shuffle null in ONE figure with
one shared TF row axis (row order = panel d, by number of modules with an edge >25 kb).
Left: Cicero pairs >25 kb shared by >=3 TFs (shared_pairs.tsv; a pair is shared when both ends overlap across TFs), restricted to pairs
  with >=2 of the row TFs; columns clustered as the original heatmap (Jaccard, average linkage on the original 44 rows; branches rotated so higher rows come first; tree not drawn); rows = panel d order, plus heatmap TFs without module bars at the bottom (n/a), "TFs per pair" on top
  (all 68 TFs counted). Middle/right: panel d unchanged (a_module_edges_v7.py code: stacked bars, per-edge null, BH q).
  python Figure3/scripts/d_module_edges_hic_support.py <outdir> [w_mm h_mm]   (Mac; EDGE_NULL_DIR = folder of the per-edge null .npz)"""
import glob, os, re, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests
ROOT = Path("/Users/dyy/Desktop/HEK293T-ATGTAG"); sys.path.insert(0, str(ROOT / "palette"))
import regulome_palette as rp
P = rp.use(); N = P.neutral; MM = 1 / 25.4
plt.rcParams.update({"legend.frameon": False, "axes.linewidth": 0.6, "svg.fonttype": "none", "pdf.fonttype": 42})
HERE = ROOT / "hub_figures/basic/descriptive"; NULL = HERE / "null_s1d_perm10000/null"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True); WMM, HMM = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (183, 95)
MODE = os.environ.get("NULL_MODE", "edge"); MIN_E = 100
ND = Path(os.environ.get("EDGE_NULL_DIR", str(ROOT / "fig3_v6/checks/null_per_edge")))
T = pd.read_csv(HERE / "s1d_per_tf_null.tsv", sep="\t").set_index("tf")
# ---- D2 data, exactly as the approved script ----
C = pd.read_csv(HERE / "clique_loop_coverage.tsv", sep="\t")
labs = ["no loop edge", ">0–25%", "25–50%", "50–75%", "75–<100%", "100% (all edges)"]; cols = P.loop_strata; cv = C.coverage
C["stratum"] = np.select([C.loop_edges == 0, cv <= .25, cv <= .5, cv <= .75, cv < 1], labs[:5], labs[5])
Pt = C.groupby(["tf", "stratum"]).size().unstack(fill_value=0).reindex(columns=labs, fill_value=0)
Pt = Pt[Pt.sum(1) >= 20]; Pt = Pt.loc[Pt.sum(1).sort_values().index]
# ---- nulls ----
E = T.loc[[t for t in Pt.index if t in T.index and T.edges[t] >= MIN_E]].copy()
if MODE == "edge":
    nul = {t: np.concatenate([np.load(f)["null"] for f in sorted(ND.glob(f"{t}_*.npz"))]) for t in E.index}
    for t in E.index:
        assert abs(float(np.load(next(ND.glob(f"{t}_*.npz")))["obs"]) - E.edge_obs[t]) < 1e-12 and len(nul[t]) == 10000, t
    E["edge_p"] = [(1 + (nul[t] >= E.edge_obs[t]).sum()) / (1 + len(nul[t])) for t in E.index]
else:
    nul = {t: np.vstack([np.load(f)["null"] for f in sorted(glob.glob(str(NULL / f"{t}_*.npz")))])[:, 1] for t in E.index}
E["q"] = multipletests(E.edge_p, method="fdr_bh")[1]; nsig = int((E.q < 0.05).sum())
# ---- figure ----
BINKB = int(os.environ.get("D_BIN_KB", 1000)); ORDER = os.environ.get("D_ORDER", "cluster")
BD = pd.read_csv(Path(__file__).resolve().parents[1] / f"checks/d_module_edge_bins_{BINKB}kb.tsv.gz", sep="\t")   # scripts/d_module_edge_bins.py (Longleaf)
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
rows = list(Pt.index); ri = {t: i for i, t in enumerate(rows)}; BD = BD[BD.tf.isin(ri)]
CHO = {f"chr{i}": i for i in range(1, 23)} | {"chrX": 23, "chrY": 24}
bins = BD[["chrom", "bin"]].drop_duplicates(); bins = bins.assign(o=bins.chrom.map(CHO)).sort_values(["o", "bin"]).reset_index(drop=True)
bi = {(c, x): j for j, (c, x) in enumerate(zip(bins.chrom, bins.bin))}
S = np.full((len(rows), len(bins)), -1, int)                                     # -1 no module edge; else bars' stratum 0..5
fr = BD.n_loop / BD.n_edges
st = np.select([BD.n_loop == 0, fr <= .25, fr <= .5, fr <= .75, fr < 1], [0, 1, 2, 3, 4], 5)
for t, c, x, v in zip(BD.tf, BD.chrom, BD.bin, st): S[ri[t], bi[(c, x)]] = v
if ORDER == "cluster":                                                           # Jaccard on presence, average linkage; tree not drawn
    Z = linkage(pdist((S >= 0).T, "jaccard"), "average"); nl = S.shape[1]; members = {i: [i] for i in range(nl)}
    toprow = -(len(rows) - 1 - ((S >= 0)[::-1].argmax(0))).astype(float)
    for k in range(len(Z)):
        l, r = int(Z[k, 0]), int(Z[k, 1])
        if toprow[members[r]].mean() < toprow[members[l]].mean(): Z[k, 0], Z[k, 1] = r, l
        members[nl + k] = members[int(Z[k, 0])] + members[int(Z[k, 1])]
    co = leaves_list(Z)
else: co = np.arange(len(bins))
S = S[:, co]; bins = bins.iloc[co].reset_index(drop=True); ntfb = (S >= 0).sum(0)
fig = plt.figure(figsize=(WMM * MM, HMM * MM))
gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.1, 0.9], height_ratios=[0.07, 1], wspace=0.05, hspace=0.03, left=0.065, right=0.995, top=0.985, bottom=0.085)
hm = fig.add_subplot(gs[1, 0]); top = fig.add_subplot(gs[0, 0], sharex=hm)
ax = fig.add_subplot(gs[1, 1], sharey=hm); b = fig.add_subplot(gs[1, 2], sharey=hm)
from matplotlib.colors import to_rgb
PAL = np.array([to_rgb(c) for c in cols] + [(1, 1, 1)]); SUB = 25; FILL = 18     # each row = 25 sub-pixels, 18 coloured (= bar height 0.72)
img = np.ones((len(rows) * SUB, S.shape[1], 3))
for i in range(len(rows)):
    y0 = i * SUB + (SUB - FILL) // 2; img[y0:y0 + FILL] = PAL[S[i]][None, :, :]
hm.imshow(img, aspect="auto", origin="lower", interpolation="nearest", extent=(-0.5, S.shape[1] - 0.5, -0.5, len(rows) - 0.5))
hm.set_xlim(-0.5, S.shape[1] - 0.5); hm.set_xticks([])
if ORDER == "genome":
    ch = bins.chrom.values; edges_ = [j for j in range(1, len(ch)) if ch[j] != ch[j - 1]]
    for j in edges_: hm.axvline(j - 0.5, color=N.pale, lw=0.3, zorder=0)
    starts = [0] + edges_; ends = edges_ + [len(ch)]
    hm.set_xticks([(s0 + e0) / 2 - 0.5 for s0, e0 in zip(starts, ends)]); hm.set_xticklabels([ch[s0].replace("chr", "") for s0 in starts], fontsize=4)
    hm.tick_params(axis="x", length=0, pad=1)
ne = int(BD.n_edges.sum()); nl_ = int(BD.n_loop.sum())
unit = f"{BINKB // 1000} Mb" if BINKB >= 1000 else f"{BINKB} kb"
hm.set_xlabel(f"Module edges >25 kb, {unit} bins (n = {S.shape[1]:,}" + (", clustered)" if ORDER == "cluster" else ")") + f"\n{ne:,} edges, {100 * nl_ / ne:.0f}% on a loop")
hm.tick_params(axis="y", length=0, pad=1.5)
for sp in ("top", "right", "bottom"): hm.spines[sp].set_visible(False)
top.bar(range(len(ntfb)), ntfb, width=1.0, color=N.mid, lw=0); top.set_xlim(-0.5, len(ntfb) - 0.5); top.set_ylim(0, ntfb.max())
top.set_yticks([0, int(ntfb.max())]); top.tick_params(labelbottom=False, bottom=False, labelsize=5, length=1.5, pad=1)
top.set_ylabel("TFs per\nbin", fontsize=5, rotation=0, ha="right", va="center", labelpad=2)
for sp in ("top", "right", "bottom"): top.spines[sp].set_visible(False)
extra = []; onl = np.array([0]); M = S
cap = 2500; y = np.arange(len(Pt)); left = np.zeros(len(Pt)); scale = np.minimum(1.0, cap / Pt.sum(1).values)
for l, c in zip(labs, cols):
    v = Pt[l].values.astype(float) * scale; ax.barh(y, v, left=left, color=c, height=0.72, label=("No loop edge" if l == labs[0] else l), edgecolor="#4D4D4D", linewidth=0.3); left += v
tot = Pt.sum(1).values; sup = tot - Pt[labs[0]].values
for i, (t, s_) in enumerate(zip(tot, sup)):
    if t == 0: ax.text(25, i, "n/a", va="center", fontsize=5, color=N.mid); continue
    ax.text(cap * 1.01 if t > cap else t + 25, i, (f"// {t:,} ({100 * s_ / t:.0f}%)" if t > cap else f"{100 * s_ / t:.0f}%"), va="center", fontsize=5, color=N.dark)
ax.set_xlim(0, cap * 1.38); hm.set_yticks(y); hm.set_yticklabels(Pt.index, fontsize=5); hm.set_ylim(-0.7, len(Pt) - 0.3); ax.tick_params(axis="y", left=False, labelleft=False)
ax.set_xlabel("Modules with ≥1 edge >25 kb")
ax.legend(title="Module edges on a loop\n(per module in bars, per bin in heatmap)", loc="lower right", ncol=1, fontsize=6, title_fontsize=6, handlelength=1.0, handleheight=0.8,
          labelspacing=0.25, borderaxespad=0.2, alignment="left")
for i, tf in enumerate(Pt.index):
    if tf not in E.index:
        b.text(13.0, i, "n/a", va="center", fontsize=5, color=N.mid); continue
    v = nul[tf] * 100; lo, q1, md, q3, hi = np.percentile(v, [2.5, 25, 50, 75, 97.5])
    b.plot([lo, hi], [i, i], color=N.mid, lw=0.6, solid_capstyle="butt")
    b.add_patch(plt.Rectangle((q1, i - 0.28), q3 - q1, 0.56, color=N.light, lw=0))
    b.plot([md, md], [i - 0.3, i + 0.3], color=N.dark, lw=0.6)
    b.plot([md, E.edge_obs[tf] * 100], [i, i], color=N.light, lw=0.5, zorder=1)
    sig = E.q[tf] < 0.05
    b.scatter([E.edge_obs[tf] * 100], [i], s=5, color=N.ink if sig else "white", edgecolors=N.ink, linewidths=0 if sig else 0.5, zorder=3)
hi_x = max(E.edge_obs.max() * 100, max(np.percentile(nul[t] * 100, 97.5) for t in E.index)) + 1
b.set_xlim(12, hi_x); b.set_xlabel("Module edges >25 kb on a loop (%)")
b.xaxis.grid(True, color=N.faint, lw=0.4); b.set_axisbelow(True)
b.tick_params(axis="y", left=False, labelleft=False); b.spines["left"].set_visible(False)
h = [plt.Line2D([], [], ls="", marker="o", ms=2.6, color=N.ink), plt.Line2D([], [], ls="", marker="o", ms=2.6, mfc="white", mec=N.ink, mew=0.5),
     plt.Rectangle((0, 0), 1, 1, color=N.light)]
b.legend(h, ["Observed, q < 0.05", "Observed, n.s.", "Shuffle null"], loc="lower right", fontsize=6, handlelength=1.0, labelspacing=0.25,
         borderaxespad=0.2, facecolor="white", frameon=True, framealpha=1, edgecolor="none")
PS = {("normal", "normal"): "ArialMT", ("bold", "normal"): "Arial-BoldMT"}
def illustrator_text(svg):
    def one(m):
        tag = m.group(0); sm = re.search(r'style="([^"]*)"', tag)
        if not sm: return tag
        keep, size, wt = [], None, "normal"
        for d in [d.strip() for d in sm.group(1).split(";") if d.strip()]:
            kk, _, v = d.partition(":"); kk, v = kk.strip(), v.strip()
            if kk == "font":
                mm_ = re.match(r"(?:(?:italic|oblique)\s+)?(?:(bold|[1-9]00)\s+)?([\d.]+)px", v)
                if mm_: wt = "bold" if mm_.group(1) in ("bold", "700") else wt; size = mm_.group(2)
            else: keep.append(f"{kk}: {v}")
        if size is None: return tag
        tag = tag.replace(sm.group(0), f'style="{"; ".join(keep)}"' if keep else "")
        return tag[:-1].rstrip() + f' font-family="Arial, {PS[(wt, "normal")]}, Helvetica, sans-serif" font-size="{float(size):.3f}"' + (' font-weight="bold"' if wt == "bold" else "") + ">"
    return re.sub(r"<text\b[^>]*>", one, svg)
stem = os.environ.get("D_STEM", "d_module_edges_hic_support")
fig.savefig(OUT / f"{stem}.svg"); fig.savefig(OUT / f"{stem}.png", dpi=300)
p = OUT / f"{stem}.svg"; p.write_text(illustrator_text(p.read_text()))
fig.canvas.draw(); tr = hm.transData                                             # alignment check: label y vs row centre in every panel
lab_y = {t.get_text(): t.get_window_extent().y0 / 2 + t.get_window_extent().y1 / 2 for t in hm.get_yticklabels()}
rowpx = abs(tr.transform((0, 1))[1] - tr.transform((0, 0))[1]); dev = []
for i, t in enumerate(rows):
    yc = [a_.transData.transform((0, i))[1] for a_ in (hm, ax, b)]; dev.append(max(abs(lab_y[t] - v) for v in yc) / rowpx)
print(f"alignment: max |label - row centre| = {max(dev):.3f} rows over {len(rows)} rows (hm, bars, null)")
dens = pd.Series((S >= 0).sum(1), index=rows).sort_values(ascending=False); edg = BD.groupby("tf").n_edges.sum().reindex(rows)
print(f"{ORDER}: {len(rows)} rows, {S.shape[1]:,} bins, {ne:,} edges; bins per row top 5:", dens.head(5).to_dict(), "| ETS2", int(dens["ETS2"]), "ZNF282", int(dens["ZNF282"]), "RBBP4", int(dens["RBBP4"]), "| edges ETS2", int(edg["ETS2"]))
