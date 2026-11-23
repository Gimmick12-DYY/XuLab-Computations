# m53 figures: score spread vs cells, pooled target histograms with the matched shuffle band, per-TF histogram pages.
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/figures_paletteC"); import regulome_palette as rp
B = "/work/users/d/y/dyy12/XuLab/regulatome/m53_stage1_percentile"; D = np.load(f"{B}/out/plot_data.npz", allow_pickle=True); PL = f"{B}/plots"; os.makedirs(PL, exist_ok=True)
def style(mode):
    P = rp.use()
    if mode == "slide": plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13, "axes.titlesize": 14, "axes.linewidth": 0.9, "lines.linewidth": 2.0, "xtick.major.width": 0.9, "ytick.major.width": 0.9})
    return P
def save(fig, name, mode):
    for ext in ("png", "pdf", "svg"): fig.savefig(f"{PL}/{name}_{mode}.{ext}", dpi=300 if mode == "slide" else 600)
    plt.close(fig); print("saved", name, mode, flush=True)
def spread(mode):
    P = style(mode); s = mode == "slide"; fig, ax = plt.subplots(figsize=(12.9, 5.9) if s else (3.5, 2.6), gridspec_kw=None); fig.subplots_adjust(left=0.08 if s else 0.17, right=0.98, bottom=0.14 if s else 0.18, top=0.95)
    ax.scatter(D["cells"], D["p95"], s=60 if s else 9, color=P.arms.raw, edgecolor="none"); ax.set_xlabel("top-25% cells per TF"); ax.set_ylabel("95th percentile of |score|"); ax.set_ylim(0, None); ax.set_xlim(0, None)
    save(fig, "P1_score_spread_vs_cells", mode)
def pooled(mode, pad, kind, sn, name):
    P = style(mode); s = mode == "slide"; o, m, lo, hi = D[f"{pad}|{kind}|{sn}"]; x = np.arange(20) * 5 + 2.5
    fig, ax = plt.subplots(figsize=(12.9, 5.9) if s else (7.2, 2.6)); fig.subplots_adjust(left=0.07 if s else 0.08, right=0.97, bottom=0.14 if s else 0.17, top=0.88)
    ax.fill_between(np.r_[0, np.repeat(np.arange(1, 20) * 5, 2), 100], np.repeat(lo, 2), np.repeat(hi, 2), color=P.neutral.light, alpha=0.6, lw=0, label="matched shuffle, 2.5–97.5%")
    ax.plot(np.r_[0, np.repeat(np.arange(1, 20) * 5, 2), 100], np.repeat(m, 2), color=P.neutral.ink, lw=1.6 if s else 0.8, label="matched shuffle, mean")
    ax.plot(x, o, "-o", color=P.arms.raw, ms=8 if s else 3, lw=1.6 if s else 0.8, label="Regulome targets")
    ax.set_xlim(0, 100); ax.set_xticks(range(0, 101, 10)); lo_ = min(o.min(), lo.min()); hi_ = max(o.max(), hi.max()); pad_ = 0.12 * (hi_ - lo_); ax.set_ylim(lo_ - pad_, hi_ + pad_); ax.set_ylabel("share of target genes (%)")
    ax.set_xlabel("within-TF percentile of |score|" if kind.startswith("|") else "within-TF percentile of signed score (0 = most down, 100 = most up)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False); save(fig, name, mode)
def per_tf(mode):
    P = style(mode); s = mode == "slide"; order = np.argsort(D["cells"], kind="stable"); tfs = D["tfs"]; x = np.arange(10) * 10 + 5
    per = 12 if s else 63; pages = [order[k:k + per] for k in range(0, len(order), per)]; pdf = PdfPages(f"{PL}/P2_per_TF_histograms_{mode}.pdf")
    for pg, idx in enumerate(pages):
        nr, nc = (3, 4) if s else (9, 7); fig, axs = plt.subplots(nr, nc, figsize=(12.9, 5.9) if s else (7.2, 9.0), sharex=True); fig.subplots_adjust(left=0.07 if s else 0.07, right=0.98, bottom=0.11 if s else 0.05, top=0.94 if s else 0.97, hspace=0.6 if s else 0.75, wspace=0.3 if s else 0.35)
        for ax in axs.ravel()[len(idx):]: ax.axis("off")
        for ax, i in zip(axs.ravel(), idx):
            tg = D["tg10"][i]; o = 100 * D["obs10"][i] / tg; m = 100 * D["dr10_mean"][i] / tg; lo = 100 * D["dr10_lo"][i] / tg; hi = 100 * D["dr10_hi"][i] / tg
            ax.fill_between(np.r_[0, np.repeat(np.arange(1, 10) * 10, 2), 100], np.repeat(lo, 2), np.repeat(hi, 2), color=P.neutral.light, alpha=0.6, lw=0)
            ax.plot(np.r_[0, np.repeat(np.arange(1, 10) * 10, 2), 100], np.repeat(m, 2), color=P.neutral.ink, lw=1.2 if s else 0.5); ax.plot(x, o, "-o", color=P.arms.raw, ms=4 if s else 1.5, lw=1.2 if s else 0.5); ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100])
            ax.set_title(f"{tfs[i]} ({int(D['cells'][i])} cells)", fontsize=12 if s else 5.5, pad=2)
            if not s: ax.tick_params(labelsize=5)
        fig.supxlabel("within-TF percentile of |score|", fontsize=13 if s else 6, y=0.0 if s else 0.005); fig.supylabel("share of target genes (%)", fontsize=13 if s else 6, x=0.0 if s else 0.005)
        pdf.savefig(fig); fig.savefig(f"{PL}/P2_per_TF_histograms_{mode}_page{pg + 1}.png", dpi=300 if s else 600); plt.close(fig)
    pdf.close(); print("saved per TF", mode, len(pages), "pages", flush=True)
for mode in ("slide", "paper"):
    spread(mode)
    for kind, kn in (("|score| percentile", "abs"), ("signed percentile", "signed")):
        for sn, sl in (("all 63 TFs", "63TFs"), ("55 non-broad TFs", "55nonbroad"), ("without NRF1, E4F1, DNMT1", "without_NRF1_E4F1_DNMT1")): pooled(mode, "10 kb", kind, sn, f"P2_hist_{kn}_percentile_{sl}_10kb")
        for pad, pl in (("0", "0kb"), ("50 kb", "50kb")): pooled(mode, pad, kind, "all 63 TFs", f"P2_hist_{kn}_percentile_63TFs_{pl}")
    per_tf(mode)
print("done")
