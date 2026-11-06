"""Gate 1 (v2.1) explained: split-half reproducibility + specificity, as a flow schematic (palette C)."""
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, "/Users/dyy/Desktop/HEK293T-ATGTAG/palette")
import regulome_palette as rp
P = rp.use(); N = P.neutral
plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(figsize=(8.4, 4.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 6.6); ax.axis("off")


def box(x, y, w, h, head, body, fc=N.faint, ec=N.dark, hc=N.ink):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.18", fc=fc, ec=ec, lw=1))
    ax.text(x + 0.15, y + h - 0.15, head, ha="left", va="top", fontsize=9.5, fontweight="bold", color=hc)
    ax.text(x + 0.15, y + h - 0.55, body, ha="left", va="top", fontsize=8, color=N.dark, linespacing=1.35)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color=N.mid, lw=1.2))


box(0.1, 5.0, 3.6, 1.5, "1  One TF's cells", "e.g. all 181,396 RBBP4 cells\nonly TFs with ≥3,400 cells are tested\n(40 of 79 TFs)")
box(4.2, 5.0, 3.6, 1.5, "2  Split in two, 20 times", "random halves A and B\ncount reads per gene in each half\n(ABC links elements to genes)")
box(8.3, 5.0, 3.6, 1.5, "3  Remove what isn't the TF", "residual after accessibility (ATAC),\nthe pooled panel of other TFs,\nGC, GC² and mappability")
arrow(3.75, 5.75, 4.15, 5.75); arrow(7.85, 5.75, 8.25, 5.75)
ax.plot([10.1, 10.1, 3.0], [4.95, 4.55, 4.55], color=N.mid, lw=1.2); arrow(3.0, 4.55, 3.0, 3.95); arrow(9.0, 4.55, 9.0, 3.95)
box(0.1, 1.45, 5.7, 2.45, "Test A · Reproducible?",
    "correlate half A with half B across genes (ρ_own)\n\nPASS if ρ_own is above the 90th percentile of\n100 pseudo-TFs: random panel cells of the\nsame number, with the TF itself left out\n(FDR 10% across TFs)",
    fc="white", ec=N.ink)
box(6.2, 1.45, 5.7, 2.45, "Test B · Specific?",
    "correlate half A with the halves of every other\ndeep TF; take the best one (ρ_other)\n\nPASS if margin = ρ_own − ρ_other > 0\nin ≥18 of 20 splits",
    fc="white", ec=N.ink)
ax.text(6.0, 1.05, "TF-level regulator = passes A and B, and fails no count-aware re-check (genes with ≥5 expected reads per half)",
        ha="center", va="center", fontsize=8.5, fontweight="bold", color=N.ink)
ax.text(6.0, 0.55, "Shallow TFs (<3,400 cells) are undetermined: with almost no reads per gene, the residual is driven by GC and\nmappability, so every shallow TF looks reproducible and specific (NEUROD4 'passed' although it is not expressed)",
        ha="center", va="center", fontsize=7.5, color=N.dark)
fig.savefig("findings_slides/png/gate1_schematic.png", dpi=400, bbox_inches="tight")
fig.savefig("findings_slides/gate1_schematic.svg", bbox_inches="tight")
