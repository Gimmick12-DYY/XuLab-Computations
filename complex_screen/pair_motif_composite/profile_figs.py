# Threshold-free motif profile, step 4: figures for AF3-supported novel pairs (palette C).
# (1) profile/fig_concordance_af3.svg: rows = AF3 strong/weak novel pairs; left block = profile concordance (Spearman, diverging),
#     right block = its percentile among 200 depth- and pct-matched random pairs (sequential), per method. Context only, no cutoff.
# (2) profile/logos/<pair>.svg: logo strip of the top 4 non-generic shared motifs per method (peak500, seqN20) with each TF's score
#     and the depth-matched percentile of the pair's min score.
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath; from matplotlib.patches import PathPatch; from matplotlib.transforms import Affine2D; from matplotlib.font_manager import FontProperties
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; G = f"{O}/profile"
sys.path.insert(0, f"{X}/figures_paletteC"); import regulome_palette as rp; P = rp.use()
T = pd.read_csv(f"{G}/novel_pairs_motif_profile.tsv", sep="\t"); T = T[T.af3_tier.isin(["strong", "weak"])].copy()
METH = ["peak500", "peak2000", "seqN20", "seqN3k"]; LAB = {"peak500": "peak-first\nN=500", "peak2000": "peak-first\nN=2,000", "seqN20": "sequence-first\n5,903 cells", "seqN3k": "sequence-first\n3,000 cells"}
T["_o"] = T.af3_tier.map({"strong": 0, "weak": 1}); T["_m"] = T[[f"conc_{m}_matched_pct" for m in METH]].mean(1); T = T.sort_values(["_o", "_m"], ascending=[True, False])
n = len(T); fig = plt.figure(figsize=(4.6, 1.6 + 0.085 * n))
bot = 0.75 / (0.6 + 0.085 * n + 1.0); top = 1 - 0.95 / (0.6 + 0.085 * n + 1.0); ax0 = fig.add_axes([0.30, bot, 0.025, top - bot]); ax1 = fig.add_axes([0.34, bot, 0.26, top - bot]); ax2 = fig.add_axes([0.63, bot, 0.26, top - bot])
ax0.imshow(T._o.values[:, None], aspect="auto", cmap=matplotlib.colors.ListedColormap([P.neutral.ink, P.neutral.light]), vmin=0, vmax=1)
ax0.set_yticks(range(n)); ax0.set_yticklabels(T.pair, fontsize=4.5); ax0.set_xticks([0]); ax0.set_xticklabels(["AF3"], rotation=90, fontsize=5)
C = T[[f"conc_{m}" for m in METH]].values; v = np.nanmax(np.abs(C))
im1 = ax1.imshow(C, aspect="auto", cmap=P.div_cmap(), vmin=-v, vmax=v); im2 = ax2.imshow(T[[f"conc_{m}_matched_pct" for m in METH]].values, aspect="auto", cmap=P.seq_cmap(), vmin=0, vmax=1)
for a, t in [(ax1, "Profile concordance (Spearman)"), (ax2, "Percentile vs matched random pairs")]:
    a.set_yticks([]); a.set_xticks(range(len(METH))); a.set_xticklabels([LAB[m] for m in METH], rotation=90, fontsize=5); a.xaxis.tick_top(); a.set_title(t, fontsize=6, pad=46)
for a, im, lab in [(ax1, im1, "rho"), (ax2, im2, "percentile")]:
    p_ = a.get_position(); cax = fig.add_axes([p_.x0, bot * 0.55, p_.width, bot * 0.12]); cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_label(lab, fontsize=5); cb.ax.tick_params(labelsize=4.5)
fig.text(0.05, bot * 0.12, "AF3: black = strong, grey = weak. Rows sorted by mean percentile within tier. Context only; no cutoff applied.", fontsize=4.5)
fig.savefig(f"{G}/fig_concordance_af3.svg"); fig.savefig(f"{G}/fig_concordance_af3.png", dpi=200); plt.close(fig)
# ---- logos
pwm = {}
for f in [f"{O}/db/known.meme", f"{O}/per_tf_raw/denovo_cmp/reps.meme"]:
    cur = None; rows = []
    for l in open(f):
        if l.startswith("MOTIF "):
            if cur and rows: pwm[cur] = np.array(rows)
            cur = l.split()[1]; rows = []
        elif cur and l.strip() and l.strip()[0] in "0123456789." and len(l.split()) == 4: rows.append([float(x) for x in l.split()])
    if cur and rows: pwm[cur] = np.array(rows)
COL = {"A": "#2C7A6B", "C": "#2166AC", "G": "#D98C1F", "T": "#B2182B"}; FP = FontProperties(family="DejaVu Sans", weight="bold")
def logo(ax, m):
    m = m / m.sum(1, keepdims=True); ic = 2 + (m * np.log2(np.maximum(m, 1e-9))).sum(1)
    for i, (row, h) in enumerate(zip(m, ic)):
        y = 0
        for k in np.argsort(row):
            hh = row[k] * h
            if hh < 0.01: continue
            tp = TextPath((0, 0), "ACGT"[k], size=1, prop=FP); bb = tp.get_extents()
            tr = Affine2D().translate(-bb.x0, -bb.y0).scale(0.9 / bb.width, hh / bb.height).translate(i + 0.05, y)
            ax.add_patch(PathPatch(tr.transform_path(tp), fc=COL["ACGT"[k]], lw=0)); y += hh
    ax.set_xlim(0, len(m)); ax.set_ylim(0, 2); ax.axis("off")
L = pd.read_csv(f"{G}/pair_top_motifs_long.tsv", sep="\t"); os.makedirs(f"{G}/logos", exist_ok=True)
for _, r in T.iterrows():
    a, b = r.pair.split("-"); fig, axs = plt.subplots(2, 4, figsize=(7.2, 2.4))
    for i, m in enumerate(["peak500", "seqN20"]):
        s = L[(L.pair == r.pair) & (L.method == m) & ~L.denovo].head(4)
        for j in range(4):
            ax = axs[i, j]
            if j >= len(s): ax.axis("off"); continue
            d = s.iloc[j]; mid = d.motif.split("|")[0]
            if mid in pwm: logo(ax, pwm[mid])
            else: ax.axis("off")
            sc = "-log10p" if m.startswith("peak") else "z"
            ax.set_title(f"{d.motif.split('|')[-1][:22]}\n{a} {d.score_A:g} / {b} {d.score_B:g} ({sc})\npanel {d.panel_pct:.2f}, depth-matched {d.depth_matched_pct:.2f}", fontsize=5)
        axs[i, 0].text(-0.25, 0.5, LAB[m].replace("\n", " "), transform=axs[i, 0].transAxes, rotation=90, va="center", ha="center", fontsize=5.5)
    fig.suptitle(f"{r.pair}  (AF3 {r.af3_tier}, ipSAE {r.af3_ipsae:.2f}; correlation pct {r.pct:.2f}; concordance pct peak {r.conc_peak500_matched_pct:.2f}, seq {r.conc_seqN20_matched_pct:.2f})", fontsize=6)
    fig.tight_layout(rect=(0.02, 0, 1, 0.93)); fig.savefig(f"{G}/logos/{r.pair}.svg"); plt.close(fig)
print(n, "AF3 pairs plotted")
