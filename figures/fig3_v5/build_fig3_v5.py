"""Assemble Fig 3 v5 (coordinated binding network, simplified) on one 183 x 247 mm page from the exact-size panels in panels/.
SVG: each panel's matplotlib SVG is inlined as <g id="panel_x" transform="translate(...)"> (ids prefixed per panel, text
kept live, Arial), with its bold 8 pt panel letter inside the group. PDF: the panel PDFs are placed on one page with pypdf
(TrueType fonts stay embedded, text is not outlined). PNG: 300 dpi preview rendered from the PDF with pdftoppm.
Usage: python build_fig3_v5.py [outdir]   (default: figures/fig3_v5/)
Grid: row 1 a | f, row 2 b | e (genomic class by number of TFs), then c and d full width; 4 mm gutters."""
import re, subprocess, sys
from pathlib import Path
from pypdf import PdfReader, PdfWriter, Transformation

HERE = Path(__file__).resolve().parent; PAN = HERE / "panels"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE; OUT.mkdir(parents=True, exist_ok=True)
PT = 72 / 25.4
PAGE_W, PAGE_H = 183.0, 247.0
# letter, panel file stem, (x, y) of the top-left corner in mm from the page's top-left, (w, h) in mm
LAYOUT = [("a", "fig3a_network", (2, 2), (64, 98)),
          ("f", "fig3f_tf_groups", (70, 2), (111, 98)),
          ("b", "fig3b_hic_support", (2, 104), (91, 45)),
          ("e", "fig3e_class_bars", (97, 104), (84, 45)),
          ("c", "fig3c_se_browser", (2, 153), (179, 41)),
          ("d", "fig3d_se_hub_3d_flower", (2, 198), (179, 48))]
LETTER_DX, LETTER_DY = -1.6, 2.6   # letter baseline offset from the panel corner, mm


def check_layout():
    for l, stem, (x, y), (w, h) in LAYOUT:
        assert x >= 0 and y >= 0 and x + w <= PAGE_W and y + h <= PAGE_H, f"panel {l} leaves the page"
        for l2, _, (x2, y2), (w2, h2) in LAYOUT:
            if l < l2 and x < x2 + w2 and x2 < x + w and y < y2 + h2 and y2 < y + h:
                raise SystemExit(f"panels {l} and {l2} overlap")


def panel_svg(letter, stem, x, y, w, h):
    s = (PAN / f"{stem}.svg").read_text()
    m = re.search(r"<svg\b([^>]*)>", s)
    vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', m.group(1)).group(1).split()]
    if abs(vb[2] - w * PT) > 0.5 or abs(vb[3] - h * PT) > 0.5:
        raise SystemExit(f"panel {letter}: {vb[2] / PT:.1f} x {vb[3] / PT:.1f} mm, expected {w} x {h}")
    body = s[m.end():s.rindex("</svg>")]
    body = re.sub(r"<metadata>.*?</metadata>", "", body, flags=re.S)
    p = f"{letter}_"
    body = re.sub(r'\bid="([^"]+)"', lambda k: f'id="{p}{k.group(1)}"', body)
    body = re.sub(r"url\(#([^)]+)\)", lambda k: f"url(#{p}{k.group(1)})", body)
    body = re.sub(r'(xlink:href|href)="#([^"]+)"', lambda k: f'{k.group(1)}="#{p}{k.group(2)}"', body)
    lx, ly = max(x + LETTER_DX, 0.3), y + LETTER_DY
    letter_el = (f'<text x="{(lx - x) * PT:.2f}" y="{(ly - y) * PT:.2f}" style="font-family: Arial, Helvetica, sans-serif; '
                 f'font-size: 8px; font-weight: bold; fill: #1A1A1A">{letter}</text>')
    return f'<g id="panel_{letter}" transform="translate({x * PT:.3f} {y * PT:.3f})">\n{letter_el}\n{body}\n</g>'


def build_svg(path):
    parts = [panel_svg(l, stem, x, y, w, h) for l, stem, (x, y), (w, h) in LAYOUT]
    svg = ('<?xml version="1.0" encoding="utf-8" standalone="no"?>\n'
           f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" '
           f'width="{PAGE_W}mm" height="{PAGE_H}mm" viewBox="0 0 {PAGE_W * PT:.3f} {PAGE_H * PT:.3f}">\n'
           f'<rect id="page" width="{PAGE_W * PT:.3f}" height="{PAGE_H * PT:.3f}" fill="#FFFFFF"/>\n' + "\n".join(parts) + "\n</svg>\n")
    path.write_text(svg)


def build_pdf(path):
    wr = PdfWriter(); page = wr.add_blank_page(width=PAGE_W * PT, height=PAGE_H * PT)
    for l, stem, (x, y), (w, h) in LAYOUT:
        src = PdfReader(PAN / f"{stem}.pdf").pages[0]
        page.merge_transformed_page(src, Transformation().translate(x * PT, (PAGE_H - y - h) * PT))
    # panel letters: a small content stream with Helvetica-Bold would not be Arial, so letters come from a 1-page matplotlib PDF
    lp = letters_pdf()
    page.merge_page(PdfReader(lp).pages[0]); lp.unlink()
    wr.add_metadata({"/Title": "Figure 3"}); wr.write(path)


def letters_pdf():
    sys.path.insert(0, str(HERE)); from fig3common import fig_mm, plt   # registers Arial
    fig = fig_mm(PAGE_W, PAGE_H); fig.patch.set_alpha(0)
    for l, _, (x, y), _ in LAYOUT:
        fig.text(max(x + LETTER_DX, 0.3) / PAGE_W, 1 - (y + LETTER_DY) / PAGE_H, l, fontsize=8, fontweight="bold", va="baseline", ha="left", color="#1A1A1A")
    p = OUT / "_letters.pdf"; fig.savefig(p, transparent=True); plt.close(fig); return p


check_layout()
build_svg(OUT / "Fig3_v5.svg"); build_pdf(OUT / "Fig3_v5.pdf")
subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", str(OUT / "Fig3_v5.pdf"), str(OUT / "Fig3_v5_preview")], check=True)
fonts = subprocess.run(["pdffonts", str(OUT / "Fig3_v5.pdf")], capture_output=True, text=True).stdout
print(fonts)
print("wrote", OUT / "Fig3_v5.svg", OUT / "Fig3_v5.pdf", OUT / "Fig3_v5_preview.png")
