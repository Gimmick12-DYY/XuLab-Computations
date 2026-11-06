"""Slide-ready clique triangle: drawn at its exact frame size (8.30 x 6.60 in) with true 45-degree Hi-C pixels (raw balanced contacts, log10, 10 kb, no smoothing) and the full triangle up to its apex. Heatmap rasterised at 600 dpi; text and lines vector.
Key genes only. Reads the saved Hi-C window (figure_a_only/hic_window_10kb.npz)."""
import gzip, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
IN = sys.argv[1]; Bd = Path(__file__).resolve().parents[1]; O = Bd / "figure_a_hires_full"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 12, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 1, "xtick.major.width": 1, "ytick.major.width": 1, "xtick.major.size": 4, "ytick.major.size": 4, "axes.spines.top": False, "axes.spines.right": False})
RED, BLUE, INK, GREY = "#C0392B", "#4A6FA5", "#222222", "#9E9E9E"; W, H = 8.30, 6.60
R = pd.read_csv(Bd / "triangles_refined.tsv", sep="\t"); t = R[(R.A == "chr1:180502204-180502470") & (R.C == "chr1:181172243-181172539")].sort_values("ref_min_don", ascending=False).iloc[0]; c = t.chrom
dA, dB, dC = int(t.mA + t.ref_sA * 1000), int(t.mB + t.ref_sB * 1000), int(t.mC + t.ref_sC * 1000); r0 = 10000; X0, X1 = int(t.mA - 80000) // r0 * r0, int(t.mC + 80000) // r0 * r0 + r0; span = (X1 - X0) / 1e3
z = np.load(Bd / "figure_a_only/hic_window_10kb.npz"); M = z["ice"]; L0 = int(z["start"]); r = int(z["res"]); i0 = (X0 - L0) // r; i1 = (X1 - L0) // r; M = M[i0:i1, i0:i1]; L0 = X0; n = M.shape[0]; Z = np.log10(M + np.nanmedian(M[M > 0]) * 0.1)
P = pd.read_csv(IN + "/peaks_cand.tsv", sep="\t", header=None, names="c s e tf sig sm".split()); P = P[(P.c == c) & (P.sm >= X0) & (P.sm < X1)]
G = pd.read_csv(gzip.open(IN + "/genes_exons.tsv.gz", "rt"), sep="\t", header=None, names="c t s e st g ty".split()); G = G[(G.c == c) & (G.t == "gene") & (G.e > X0) & (G.s < X1)]
fig = plt.figure(figsize=(W, H)); L, AW = 1.05 / W, 6.90 / W; XL = (X0 / 1e6, X1 / 1e6); ymax = span / 2; hh = 6.90 * ymax / span   # equal x and y scale, so pixels are true diamonds
top = 6.32; ax = fig.add_axes([L, (top - hh) / H, AW, hh / H])
xs = L0 + np.arange(n + 1) * r; XX, YY = np.meshgrid(xs, xs, indexing="ij"); Zm = np.ma.masked_invalid(np.ma.masked_where(np.tril(np.ones_like(Z), -1) > 0, Z))
pc = ax.pcolormesh((XX + YY) / 2e6, (YY - XX) / 2e3, Zm, cmap="Reds", vmin=np.nanpercentile(Z, 20), vmax=np.nanpercentile(Z, 98.5), shading="flat", rasterized=True, lw=0, edgecolors="none", antialiased=False)
for nm, i, j in (("A–B", dA, dB), ("B–C", dB, dC), ("A–C", dA, dC)):
    ax.plot((i + j) / 2e6, (j - i) / 2e3, "o", mfc="none", mec=INK, ms=25, mew=2, zorder=5); ax.text((i + j) / 2e6, (j - i) / 2e3 + 32, nm, ha="center", va="bottom", fontsize=14, fontweight="bold", color=INK, zorder=6)
ax.set_xlim(*XL); ax.set_ylim(0, ymax); ax.tick_params(bottom=False, labelbottom=False); ax.spines["bottom"].set_visible(False); ax.set_ylabel("Distance / 2 (kb)", fontsize=14); ax.set_yticks([0, 100, 200, 300, 400])
ax.set_title("WT HEK293T Hi-C, 10 kb", loc="left", fontsize=14, fontweight="bold", pad=6)
cb = fig.add_axes([L + AW - 1.45 / W, (top - 0.42) / H, 1.4 / W, 0.14 / H]); plt.colorbar(pc, cax=cb, orientation="horizontal"); cb.set_title("log10 contact", fontsize=12, pad=3); cb.tick_params(labelsize=12); cb.xaxis.set_major_locator(plt.MaxNLocator(3))
# anchors + arcs
y2 = top - hh - 0.86; ax2 = fig.add_axes([L, y2 / H, AW, 0.82 / H], sharex=ax); ax2.set_axis_off(); ax2.set_ylim(0, 1.2); AN = [("A", t.mA), ("B", t.mB), ("C", t.mC)]
for a, b in ((t.mA, t.mB), (t.mB, t.mC), (t.mA, t.mC)):
    q = np.linspace(a, b, 100) / 1e6; ax2.plot(q, 0.5 + (0.65 * (b - a) / (t.mC - t.mA)) * np.sin(np.pi * (q - a / 1e6) / ((b - a) / 1e6)), color=BLUE, lw=2)
for nm, m in AN:
    ax2.plot(m / 1e6, 0.5, "v", ms=10, color=INK, zorder=4); ax2.text(m / 1e6, 0.33, nm, ha="center", va="top", fontsize=14, fontweight="bold")
ax2.text(XL[1], 1.17, "Cicero clique edges (ETS2)", color=BLUE, fontsize=12, va="top", ha="right")
# anchor labels + TF bars
y3 = y2 - 0.84; ax3 = fig.add_axes([L, y3 / H, AW, 0.68 / H], sharex=ax); ed = np.arange(X0 // 5000 * 5000, X1 + 5000, 5000); cnt = np.array([P[(P.sm >= e) & (P.sm < e + 5000)].tf.nunique() for e in ed])
ax3.bar((ed + 2500) / 1e6, cnt, width=0.0048, color="#BFBFBF", lw=0); ax3.set_ylabel("TFs", fontsize=14); ax3.set_ylim(0, 26); ax3.set_yticks([0, 20]); ax3.tick_params(bottom=False, labelbottom=False)
for (nm, m), reg, lab, ha in zip(AN, (t.A, t.B, t.C), ("ACBD6 promoter", "distal", "LINC01732 promoter"), ("left", "center", "right")):
    s, e = map(int, reg.split(":")[1].split("-")); k = P[(P.sm >= s - 1000) & (P.sm <= e + 1000)].tf.nunique(); ax3.bar(m / 1e6, max(k, 0.8), width=0.0048, color=INK, lw=0)
    x_ = m / 1e6 if nm != "C" else XL[1]; ax3.text(x_, 27.5, f"{lab}, {k} TF" + "s" * (k != 1), fontsize=12, ha=ha if nm != "A" else "left", va="bottom", clip_on=False)
# key genes
KEY = ["ACBD6", "XPR1", "KIAA1614", "STX6", "MR1", "IER5", "LINC01732"]; y4 = y3 - 0.62; ax4 = fig.add_axes([L, y4 / H, AW, 0.56 / H], sharex=ax); ax4.set_ylim(1.75, -0.65); ax4.set_yticks([]); ax4.spines["left"].set_visible(False)
ROW = dict(ACBD6=0, XPR1=0, MR1=0, KIAA1614=1, STX6=0, IER5=1, LINC01732=1)
for nm in KEY:
    g = G[G.g == nm].iloc[0]; rw = ROW[nm]; a, b = max(g.s, X0) / 1e6, min(g.e, X1) / 1e6; ax4.plot([a, b], [rw + 0.32] * 2, color=INK, lw=2.2, solid_capstyle="butt"); ax4.text(max((a + b) / 2, XL[0] + 0.03), rw + 0.2, nm, fontsize=12, fontstyle="italic", ha="center", va="bottom")
ax4.set_xlim(*XL); ax4.set_xticks([180.5, 180.7, 180.9, 181.1]); ax4.set_xlabel(f"{c} (Mb)", fontsize=14, labelpad=2); ax4.tick_params(labelsize=12)
for e in ("svg", "pdf"): fig.savefig(O / f"clique_triangle_a_full.{e}", dpi=600, transparent=True)
fig.savefig(O / "clique_triangle_a_full.png", dpi=600, facecolor="white")
from PIL import Image
Image.MAX_IMAGE_PIXELS = None; im = Image.open(O / "clique_triangle_a_full.png"); print("png", im.size, "heat height in", round(hh, 2), "y4", round(y4, 2)); im.convert("RGB").resize((round(1920 * W / 13.333), round(1920 * W / 13.333 * H / W)), Image.LANCZOS).save(O / "check_on_1920px_slide.png")
