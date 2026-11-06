"""Clique triangle figure: rotated WT Hi-C heatmap (10 kb) with the 3 dots circled, aligned anchors, Cicero arcs, TF counts and genes."""
import gzip, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent)); from hicread import Hic
IN = sys.argv[1]; Bd = Path(__file__).resolve().parents[1]; O = Bd / "figure_a_only"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "axes.spines.top": False, "axes.spines.right": False})
TFS = ["ZNF507", "SOX4", "NFYA", "GLYR1", "NME2", "ZNF282", "ADNP", "DR1", "MTF2", "E4F1"]; COL = dict(zip(TFS, ["#C48311", "#32A190", "#D9B637", "#C681F1", "#739B64", "#8C486E", "#4DC7E5", "#8AC975", "#65278F", "#712F55"]))
RED, BLUE, INK, GREY = "#C0392B", "#4A6FA5", "#222222", "#9E9E9E"
R = pd.read_csv(Bd / "triangles_refined.tsv", sep="\t"); N = pd.read_csv(Bd / "null_refined.tsv", sep="\t")
t = R[(R.A == "chr1:180502204-180502470") & (R.C == "chr1:181172243-181172539")].sort_values("ref_min_don", ascending=False).iloc[0]; c = t.chrom
dA, dB, dC = int(t.mA + t.ref_sA * 1000), int(t.mB + t.ref_sB * 1000), int(t.mC + t.ref_sC * 1000); pad = 150000; X0, X1 = t.mA - pad, t.mC + pad; r = 10000
P = pd.read_csv(IN + "/peaks_cand.tsv", sep="\t", header=None, names="c s e tf sig sm".split()); P = P[(P.c == c) & (P.sm >= X0) & (P.sm < X1)]
G = pd.read_csv(gzip.open(IN + "/genes_exons.tsv.gz", "rt"), sep="\t", header=None, names="c t s e st g ty".split()); G = G[(G.c == c) & (G.t == "gene") & (G.ty.isin(["pc", "lnc"])) & (G.e > X0) & (G.s < X1)].sort_values("s")
fig = plt.figure(figsize=(330 / 25.4, 104 / 25.4)); XL = (X0 / 1e6, X1 / 1e6)
# heatmap
ax = fig.add_axes([0.045, 0.47, 0.94, 0.47]); L0 = X0 // r * r; M, _, w = Hic(IN + "/wt.mcool", r).fetch(c, L0, X1); np.savez_compressed(O / "hic_window_10kb.npz", ice=M, start=L0, res=r); n = M.shape[0]; Z = np.log10(M + np.nanmedian(M[M > 0]) * 0.1)
xs = L0 + np.arange(n + 1) * r; XX, YY = np.meshgrid(xs, xs, indexing="ij"); Zm = np.ma.masked_invalid(np.ma.masked_where(np.tril(np.ones_like(Z), -1) > 0, Z))
pc = ax.pcolormesh((XX + YY) / 2e6, (YY - XX) / 2e3, Zm, cmap="Reds", vmin=np.nanpercentile(Z, 20), vmax=np.nanpercentile(Z, 98.5), shading="flat", rasterized=True, lw=0)
for nm, i, j in (("A–B", dA, dB), ("B–C", dB, dC), ("A–C", dA, dC)):
    ax.plot((i + j) / 2e6, (j - i) / 2e3, "o", mfc="none", mec=INK, ms=13, mew=0.9, zorder=5); ax.text((i + j) / 2e6, (j - i) / 2e3 + 25, nm, ha="center", va="bottom", fontsize=7, fontweight="bold", color=INK, zorder=6)
ax.set_xlim(*XL); ax.set_ylim(0, (t.mC - t.mA) / 2e3 * 1.22); ax.tick_params(bottom=False, labelbottom=False); ax.spines["bottom"].set_visible(False); ax.set_ylabel("distance / 2 (kb)")
ax.set_title("WT HEK293T Hi-C, 10 kb, log10 balanced contacts", loc="left", fontsize=7)
cb = fig.add_axes([0.9, 0.84, 0.07, 0.018]); plt.colorbar(pc, cax=cb, orientation="horizontal"); cb.tick_params(labelsize=5, length=1.5, pad=1); cb.set_title("log10 contact", fontsize=5, pad=1.5)
# anchors + arcs
ax2 = fig.add_axes([0.045, 0.33, 0.94, 0.13], sharex=ax); ax2.set_axis_off(); ax2.set_ylim(0, 1.25); AN = [("A", t.mA, "ACBD6 promoter"), ("B", t.mB, "distal"), ("C", t.mC, "LINC01732 promoter")]
for a, b in ((t.mA, t.mB), (t.mB, t.mC), (t.mA, t.mC)):
    xs_ = np.linspace(a, b, 80) / 1e6; ax2.plot(xs_, 0.28 + (0.85 * (b - a) / (t.mC - t.mA)) * np.sin(np.pi * (xs_ - a / 1e6) / ((b - a) / 1e6)), color=BLUE, lw=1)
for nm, m, g in AN:
    ax2.plot(m / 1e6, 0.28, "v", ms=5, color=INK, zorder=4); ax2.text(m / 1e6, 0.14, f"{nm}", ha="center", va="top", fontweight="bold")
ax2.text(XL[0], 1.0, "Cicero clique edges (ETS2)", color=BLUE, fontsize=5.8, va="top")
# TF bars
ax3 = fig.add_axes([0.045, 0.2, 0.94, 0.11], sharex=ax); ed = np.arange(X0 // 5000 * 5000, X1 + 5000, 5000); cnt = [P[(P.sm >= e) & (P.sm < e + 5000)].tf.nunique() for e in ed]
ax3.bar((ed + 2500) / 1e6, cnt, width=0.0048, color="#CFCFCF", lw=0); ax3.set_ylabel("TFs /\n5 kb", fontsize=5.8); ax3.set_ylim(0, 34); ax3.set_yticks([0, 15, 30]); ax3.tick_params(bottom=False, labelbottom=False); out = []
for (nm, m, g), reg in zip(AN, (t.A, t.B, t.C)):
    s, e = map(int, reg.split(":")[1].split("-")); q = sorted(P[(P.sm >= s - 1000) & (P.sm <= e + 1000)].tf.unique()); out.append(dict(anchor=nm, region=reg, label=g, n_tfs=len(q), tfs=",".join(q)))
    ax3.bar(m / 1e6, max(len(q), 0.5), width=0.0048, color=INK, lw=0); kol = [x for x in TFS if x in q]; x0 = m / 1e6 + 0.009
    ax3.text(x0, 31, f"{nm}: {g}, {len(q)} TF" + "s" * (len(q) != 1), fontsize=5.6, va="center", ha="left" if nm != "C" else "right", color=INK) if nm != "C" else ax3.text(m / 1e6 - 0.009, 31, f"{nm}: {g}, {len(q)} TF" + "s" * (len(q) != 1), fontsize=5.6, va="center", ha="right", color=INK)
    for k, x in enumerate(kol):
        xp = x0 + k * 0.009 if nm != "C" else m / 1e6 - 0.009 - k * 0.009; ax3.plot(xp, 23, "o", ms=2.8, color=COL[x], mec="none")
pd.DataFrame(out).to_csv(O / "anchors.tsv", sep="\t", index=False)
# genes
ax4 = fig.add_axes([0.045, 0.075, 0.94, 0.11], sharex=ax); ax4.set_ylim(2.6, -0.6); ax4.set_yticks([]); ax4.spines["left"].set_visible(False); ends = [-1e12] * 3
for g in G.itertuples():
    k = next((i for i in range(3) if g.s > ends[i] + 45000), 2); ends[k] = max(ends[k], g.e); ax4.plot([max(g.s, X0) / 1e6, min(g.e, X1) / 1e6], [k, k], color=INK if g.ty == "pc" else GREY, lw=1.1, solid_capstyle="butt")
    ax4.text(max(max(g.s, X0) / 1e6, XL[0]), k - 0.2, g.g[:14], fontsize=4.8, fontstyle="italic", va="bottom", color=INK if g.ty == "pc" else "#666666", clip_on=True)
ax4.set_xlim(*XL); ax4.set_xticks(np.arange(180.4, 181.4, 0.2)); ax4.set_xlabel(f"{c} (Mb)", labelpad=1)
fig.text(0.985, 0.955, "Circles sit on the best pixel within \u00b120 kb of the Cicero anchors; \u25bc marks the anchors themselves.", fontsize=5.5, va="center", ha="right", color="#444444")
for e in ("svg", "pdf"): fig.savefig(O / f"clique_triangle_a.{e}", dpi=600, transparent=True)
fig.savefig(O / "clique_triangle_a.png", dpi=600, facecolor="white"); fig.savefig(O / "clique_triangle_a_preview.png", dpi=220, facecolor="white")
print(pd.DataFrame(out).drop(columns="tfs").to_string(index=False)); print("dots", dA, dB, dC, "kb", (dB - dA, dC - dB, dC - dA))
