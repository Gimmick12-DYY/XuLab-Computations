"""Shared figure style for the paper figures.

Every panel script calls `use()` before plotting and `save(fig, outdir, name)` to write it. This guarantees:
  * Arial, with text kept as live, editable text in every vector file
    (PDF: TrueType fonts embedded, pdf.fonttype = 42; SVG: <text> elements, svg.fonttype = "none");
  * palette C colours (PALETTE below);
  * the same three outputs per panel: <name>.pdf, <name>.svg and a 300 dpi <name>.png preview.

The Arial TrueType files are read from $FIG_FONT_DIR (default: the fonts/arial folder next to the repo on Longleaf).
"""
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

MM = 1 / 25.4   # inches per mm; figure sizes are given in mm
FONT_DIR = Path(os.environ.get("FIG_FONT_DIR", "/work/users/d/y/dyy12/fonts/arial"))


class _NS(dict):
    __getattr__ = dict.__getitem__


# Palette C ("Continuity"), shared by all figures
NEUTRAL = _NS(ink="#1A1A1A", dark="#4D4D4D", mid="#8C8C8C", light="#BDBDBD", pale="#E3E3E3", faint="#F2F2F2")
CHROMHMM = _NS(promoter="#B03A2E", active_enh="#E8821E", weak_enh="#F6CB8F", tss_flank="#D46A5E",
               transcribed="#7FA6CE", repressed="#55606E", quiescent="#E3E3E3")
EVIDENCE = _NS(hic_loop="#B83A2E", cicero_only="#7A9CCB")
LOOP_STRATA = ["#E3E3E3", "#FCBBA1", "#FC9272", "#EF3B2C", "#CB181D", "#67000D"]   # no loop edge, >0-25, 25-50, 50-75, 75-<100, 100 %
SEQ = ["#FFF5F0", "#FEE0D2", "#FCBBA1", "#FC9272", "#FB6A4A", "#EF3B2C", "#CB181D", "#A50F15", "#67000D"]
HIGHLIGHT = "#FCE9A8"
PALETTE = _NS(neutral=NEUTRAL, chromhmm=CHROMHMM, evidence=EVIDENCE, loop_strata=LOOP_STRATA, seq=SEQ, highlight=HIGHLIGHT)

# Super-enhancer hub figures (panels g-i): edge colours by the class of the partner element
SE_EDGE = _NS(se_promoter="#0072B2", se_enhancer="#D55E00", se_other="#999999",
              pp_promoter="#56B4E9", pp_enhancer="#F0A35E", pp_other="#D0D0D0")
SE_NODE = _NS(promoter="#0072B2", enhancer="#D55E00", other="#8C8C8C", se_fill="#FFC20A", se_edge="#5A4500")
SE_TF = _NS(ZNF507="#E69F00", SOX4="#009E73", NFYA="#F0E442", GLYR1="#CC79A7", NME2="#117733", ZNF282="#DDCC77",
            ADNP="#44AA99", DR1="#999933", MTF2="#AA4499", E4F1="#A6761D", other="#BDBDBD")


def seq_cmap():
    c = LinearSegmentedColormap.from_list("regulome_seq", SEQ)
    c.set_bad(NEUTRAL.faint)
    return c


def _register_arial():
    if FONT_DIR.is_dir():
        for f in sorted(FONT_DIR.glob("*.ttf")):
            font_manager.fontManager.addfont(str(f))
    names = {f.name for f in font_manager.fontManager.ttflist}
    if "Arial" not in names:
        raise RuntimeError(f"Arial not found (looked in {FONT_DIR}); set FIG_FONT_DIR to a folder with Arial*.ttf")


def use(base_pt=7.0, small_pt=6.0):
    """Apply the paper style. Sizes are final print sizes in pt (panels are drawn at print size)."""
    _register_arial()
    ink = NEUTRAL.ink
    matplotlib.rcdefaults()   # panels run in one process: start every panel from the same state
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial"],
        "font.size": base_pt, "axes.titlesize": base_pt, "axes.labelsize": base_pt, "figure.titlesize": base_pt,
        "xtick.labelsize": small_pt, "ytick.labelsize": small_pt, "legend.fontsize": small_pt,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "lines.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "text.color": ink, "axes.labelcolor": ink, "axes.edgecolor": ink, "axes.titlecolor": ink,
        "xtick.color": ink, "ytick.color": ink, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "svg.hashsalt": "paper-figures", "path.simplify": False,
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    })
    return PALETTE


_PS = {("normal", "normal"): "ArialMT", ("bold", "normal"): "Arial-BoldMT",
       ("normal", "italic"): "Arial-ItalicMT", ("bold", "italic"): "Arial-BoldItalicMT"}


def _illustrator_svg_text(svg):
    """Matplotlib writes SVG text styles as a CSS 'font:' shorthand (e.g. font: 700 6px 'Arial'), which Adobe Illustrator
    misreads (it takes the weight as the size). Rewrite each <text> style as explicit font-family/size/weight/style
    attributes. Geometry and text content are not touched."""
    def one(m):
        tag = m.group(0)
        sm = re.search(r'style="([^"]*)"', tag)
        if not sm:
            return tag
        keep, size, wt, it = [], None, "normal", "normal"
        for d in [d.strip() for d in sm.group(1).split(";") if d.strip()]:
            k, _, v = d.partition(":")
            k, v = k.strip(), v.strip()
            if k == "font":
                mm = re.match(r"(?:(italic|oblique)\s+)?(?:(normal|bold|[1-9]00)\s+)?([\d.]+)px", v)
                if mm:
                    it = "italic" if mm.group(1) else it
                    wt = "bold" if mm.group(2) in ("bold", "600", "700", "800", "900") else wt
                    size = mm.group(3)
            elif k == "font-size":
                size = v.replace("px", "")
            elif k == "font-weight":
                wt = "bold" if v in ("bold", "600", "700", "800", "900") else "normal"
            elif k == "font-style":
                it = "italic" if v in ("italic", "oblique") else "normal"
            elif k == "font-family":
                pass
            else:
                keep.append(f"{k}: {v}")
        if size is None:
            return tag
        tag = tag.replace(sm.group(0), f'style="{"; ".join(keep)}"' if keep else "")
        return (tag[:-1].rstrip() + f' font-family="Arial, {_PS[(wt, it)]}, Helvetica, sans-serif" font-size="{float(size):.3f}"'
                + (' font-weight="bold"' if wt == "bold" else "") + (' font-style="italic"' if it == "italic" else "") + ">")
    return re.sub(r"<(?:text|tspan)\b[^>]*>", one, svg)


def save(fig, outdir, name, png_dpi=300, raster_dpi=600, **kw):
    """Write <name>.pdf, .svg (live text) and a .png preview. raster_dpi only affects artists drawn with rasterized=True
    (dense heatmaps, Hi-C matrices); everything else is vector."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    kw.setdefault("metadata", None)
    fig.savefig(outdir / f"{name}.pdf", dpi=raster_dpi, metadata={"Creator": "matplotlib", "Producer": None, "CreationDate": None}, **{k: v for k, v in kw.items() if k != "metadata"})
    fig.savefig(outdir / f"{name}.svg", dpi=raster_dpi, metadata={"Date": None}, **{k: v for k, v in kw.items() if k != "metadata"})
    p = outdir / f"{name}.svg"
    p.write_text(_illustrator_svg_text(p.read_text()))
    fig.savefig(outdir / f"{name}.png", dpi=png_dpi, **{k: v for k, v in kw.items() if k != "metadata"})
    plt.close(fig)
    return outdir / f"{name}.pdf"


def content_bbox(fig, pad_in=0.02):
    """Bounding box (inches) of everything drawn, ignoring the frame of full-figure axes; pass as bbox_inches to save()."""
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = []
    for ax in fig.axes:
        for a in ax.get_children():
            if a.get_visible() and a is not ax.patch and type(a).__name__ not in ("Spine", "XAxis", "YAxis"):
                try:
                    b = a.get_window_extent(r)
                except Exception:
                    continue
                if b.width > 0 or b.height > 0:
                    boxes.append(b)
    b = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted())
    return Bbox.from_extents(b.x0 - pad_in, b.y0 - pad_in, b.x1 + pad_in, b.y1 + pad_in)


def figure_mm(w_mm, h_mm):
    return plt.figure(figsize=(w_mm * MM, h_mm * MM))
