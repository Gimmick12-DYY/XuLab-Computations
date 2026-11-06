"""Regulome status schematic (10/29): TF-level regulators, module nodes and TF-TF complex edges, palette C."""
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
sys.path.insert(0, "/Users/dyy/Desktop/HEK293T-ATGTAG/palette")
import regulome_palette as rp
P = rp.use()
N = P.neutral
plt.rcParams.update({"font.size": 9})

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.set_xlim(0, 10.6); ax.set_ylim(-0.3, 6.2); ax.axis("off")


def box(x, y, w, h, title, sub, ls="-", fc=N.faint, ec=N.dark):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.25", fc=fc, ec=ec, lw=1, ls=ls))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=9, fontweight="bold", color=N.ink)
    ax.text(x + w / 2, y + h - 0.55, sub, ha="center", va="top", fontsize=7, color=N.dark)


pos = {}


def node(name, x, y, kind):
    pos[name] = (x, y)
    if kind == "tf":          # TF-level regulator
        kw = dict(fc=N.ink, ec=N.ink, tc="white")
    elif kind == "border":    # TF-level, borderline
        kw = dict(fc="white", ec=N.ink, tc=N.ink)
    else:                     # partner / module member, no own gene targets
        kw = dict(fc=N.pale, ec=N.mid, tc=N.ink)
    ax.text(x, y, name, ha="center", va="center", fontsize=8.5, color=kw["tc"], fontweight="bold" if kind != "member" else "normal",
            bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.5", fc=kw["fc"], ec=kw["ec"], lw=1.1), zorder=5)


# module nodes
box(0.25, 3.35, 3.0, 2.45, "PRC2 core", "module node (RBBP4/7 removed)")
box(0.25, 0.25, 3.0, 2.0, "NuRD core", "module node (RBBP4/7 removed)")
box(7.55, 4.15, 2.9, 1.75, "HOX / ZIC group", "provisional: all low-library TFs", ls="--", fc="white", ec=N.light)
node("MTF2", 0.95, 4.35, "member"); node("EZH2", 2.45, 4.35, "member"); node("PCGF6", 1.7, 3.75, "member")
node("MBD3", 1.75, 0.9, "member")
ax.text(9.0, 4.85, "HOXA10 · HOXB6 · HOXC9\nZIC2 · SREBF2 · DR1 …", ha="center", va="center", fontsize=7.5, color=N.mid)

# TF-level regulators
node("RBBP4", 4.6, 3.1, "border"); node("RBBP7", 4.6, 1.6, "member")
node("WIZ", 4.6, 5.2, "member"); node("ZNF644", 6.4, 5.2, "member")
node("MLX", 6.4, 3.1, "tf"); node("FOXK1", 8.1, 3.1, "member")
for k, t in enumerate(["NFYA", "SOX4", "ZNF507", "E4F1", "ZNF777"]):
    node(t, 4.6 + k * 1.3, 0.75, "border" if t == "NFYA" else "tf")
ax.text(7.2, 1.45, "TF-level regulators with own gene targets", ha="center", fontsize=7.5, color=N.dark)


def edge(a, b, style):
    (x1, y1), (x2, y2) = pos[a], pos[b]
    kw = {"complex": dict(color=N.ink, lw=2.2, ls="-"), "dna": dict(color=N.ink, lw=1.6, ls=(0, (4, 2))),
          "weak": dict(color=N.mid, lw=1.2, ls=(0, (1, 1.5)))}[style]
    ax.plot([x1, x2], [y1, y2], zorder=2, solid_capstyle="round", **kw)


edge("EZH2", "MTF2", "complex"); edge("PCGF6", "RBBP4", "complex"); edge("RBBP4", "RBBP7", "complex")
edge("WIZ", "ZNF644", "complex"); edge("EZH2", "WIZ", "weak"); edge("MLX", "FOXK1", "dna")

leg = [Line2D([], [], marker="o", ls="", ms=9, mfc=N.ink, mec=N.ink, label="TF-level regulator"),
       Line2D([], [], marker="o", ls="", ms=9, mfc="white", mec=N.ink, label="TF-level, borderline"),
       Line2D([], [], marker="o", ls="", ms=9, mfc=N.pale, mec=N.mid, label="partner / module member"),
       Line2D([], [], color=N.ink, lw=2.2, label="complex edge (structure + STRING)"),
       Line2D([], [], color=N.ink, lw=1.6, ls=(0, (4, 2)), label="DNA-guided (CAP-SELEX + co-binding)"),
       Line2D([], [], color=N.mid, lw=1.2, ls=(0, (1, 1.5)), label="weak, structure only")]
ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=7.5, handlelength=2.6, columnspacing=1.6)
fig.savefig("png/regulome_schematic.png", dpi=400, bbox_inches="tight")
fig.savefig("regulome_schematic.svg", bbox_inches="tight")
