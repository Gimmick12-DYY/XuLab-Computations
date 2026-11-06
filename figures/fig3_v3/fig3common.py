"""Shared setup for the Fig 3 v3 panel scripts: palette C (regulome_palette), Arial registered from a user font folder
(Longleaf has no system Arial), exact-size saving (SVG + PDF, live text) and a layout check (nothing outside the canvas,
no overlapping text, no text below MIN_PT)."""
import sys, glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
XULAB = HERE.parents[1]
INP = HERE / "inputs"
OUT = HERE / "panels"; OUT.mkdir(exist_ok=True)
FONT_DIR = Path("/work/users/d/y/dyy12/fonts/arial")
for f in glob.glob(str(FONT_DIR / "*.ttf")):
    font_manager.fontManager.addfont(f)
sys.path.insert(0, str(XULAB / "figures_paletteC"))
import regulome_palette as rp

P = rp.use()
plt.rcParams.update({"font.sans-serif": ["Arial"], "svg.fonttype": "none", "pdf.fonttype": 42})
MM = 1 / 25.4
MIN_PT = 6.0


def fig_mm(w, h, **kw):
    return plt.figure(figsize=(w * MM, h * MM), **kw)


def mm_axes(fig, w, h, left, bottom, width, height):
    """Axes placed in mm from the bottom-left corner of a w x h mm figure."""
    return fig.add_axes([left / w, bottom / h, width / w, height / h])


def check(fig, name, allow_overlap=()):
    """Raise on text below MIN_PT; print any artist outside the canvas and any overlapping text pairs."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer(); W, H = fig.bbox.width, fig.bbox.height
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    small = [(t.get_text(), t.get_fontsize()) for t in texts if t.get_fontsize() < MIN_PT - 1e-6]
    if small:
        raise SystemExit(f"{name}: text below {MIN_PT} pt: {small}")
    bbs = [(t, t.get_window_extent(r)) for t in texts]
    tb = fig.get_tightbbox(r).transformed(fig.dpi_scale_trans)
    out = [(t.get_text(), b) for t, b in bbs if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > W + 0.5 or b.y1 > H + 0.5]
    over, tol = [], 0.4 / 25.4 * fig.dpi   # line boxes carry ~0.8 mm of empty ascent/descent at 6 pt; 0.4 mm overlap = no ink contact
    for i in range(len(bbs)):
        for j in range(i + 1, len(bbs)):
            a, b = bbs[i][1], bbs[j][1]
            if a.x0 < b.x1 - tol and b.x0 < a.x1 - tol and a.y0 < b.y1 - tol and b.y0 < a.y1 - tol:
                pair = (bbs[i][0].get_text(), bbs[j][0].get_text())
                if pair not in allow_overlap and pair[::-1] not in allow_overlap:
                    over.append(pair)
    print(f"[{name}] text outside canvas: {[o[0] for o in out] or 'none'}; overlapping text: {over or 'none'}")
    return out, over


def save(fig, name, **ck):
    check(fig, name, **ck)
    for ext in ("svg", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", facecolor="white")   # no bbox_inches: keep the exact mm size
    fig.savefig(OUT / f"{name}_check.png", dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", OUT / f"{name}.svg")
