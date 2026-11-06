"""Boltz-2 interface scores for the MAZ-ZNF143 worked example: 31 calibration pairs (ipTM vs ipSAE) with MAZ-ZNF143 and
MLX-SREBF2 marked (calibration run and screen run). Values from complex_screen/calib/calib_scores.tsv and screen/scores.tsv."""
import sys
sys.path.insert(0, "/Users/dyy/Desktop/HEK293T-ATGTAG/palette")
import regulome_palette as rp
import matplotlib.pyplot as plt
P = rp.use()
CAL = {  # class: [(ipTM_top, ipSAE_top)]
    "known complex (direct)": [(0.568, 0.217), (0.584, 0.053), (0.772, 0.775), (0.711, 0.765), (0.429, 0.432), (0.764, 0.663),
                               (0.631, 0.770), (0.840, 0.745)],
    "same complex, not touching": [(0.216, 0.0), (0.237, 0.0), (0.228, 0.0)],
    "should not bind": [(0.140, 0.0), (0.168, 0.0), (0.211, 0.0), (0.169, 0.0), (0.167, 0.0), (0.244, 0.0), (0.396, 0.0)],
}
COL = {"known complex (direct)": P.evidence.hic_loop, "same complex, not touching": P.arms.imputed, "should not bind": P.neutral.mid}
fig, ax = plt.subplots(figsize=(3.3, 2.7))
ax.fill_between([0.4, 0.9], 0.3, 0.85, color=P.highlight, lw=0, zorder=0)
ax.axhline(0.3, ls="--", lw=0.6, color=P.neutral.mid); ax.axvline(0.4, ls="--", lw=0.6, color=P.neutral.mid)
for k, v in CAL.items():
    x, y = zip(*v); ax.scatter(x, y, s=14, color=COL[k], label=k, lw=0, alpha=0.9, zorder=2)
for name, pts, c in (("MAZ–ZNF143", [(0.200, 0.0), (0.163, 0.0)], P.arms.raw), ("MLX–SREBF2", [(0.495, 0.402), (0.604, 0.444)], P.arms.atac)):
    ax.scatter(*zip(*pts), s=34, marker="D", facecolor="white", edgecolor=c, lw=1.2, zorder=3, label=name + " (2 runs)")
ax.annotate("MAZ–ZNF143\nipSAE 0 in both runs", (0.18, 0.0), (0.15, 0.17), fontsize=6, color=P.arms.raw,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=P.arms.raw))
ax.text(0.89, 0.31, "hit zone", ha="right", va="bottom", fontsize=6, color=P.neutral.dark)
ax.set_xlim(0.1, 0.9); ax.set_ylim(-0.05, 0.85)
ax.set_xlabel("ipTM (placement confidence)"); ax.set_ylabel("ipSAE (interface confidence)")
ax.legend(loc="upper left", fontsize=5.5, handletextpad=0.3, borderaxespad=0.2)
fig.tight_layout(); fig.savefig("png/MAZ_ZNF143_boltz_scores.png", dpi=400)
