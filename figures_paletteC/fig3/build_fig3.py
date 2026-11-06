# Assemble the Fig 3 draft (coordinated binding) as journal-width SVG pages from existing panel files.
# Panels are read only; SVG panels are inlined (ids prefixed per panel, fonts set to Arial) so text stays editable in
# Illustrator; PNG panels are embedded as base64 images (href + xlink:href, so SVG 1.1 and SVG 2 viewers both show them).
# Outputs: fig3_draft.svg (main), fig3_extended_data.svg, <name>_panels.txt, <name>_preview.html (for the PDF/PNG preview).
import base64, re
from pathlib import Path

ROOT = Path("/Users/dyy/Desktop/HEK293T-ATGTAG")
H = ROOT / "hub_figures/basic"
OUT = ROOT / "fig3_draft"

PT_PER_MM = 72 / 25.4
WIDTH_MM = 180                      # double-column journal width
MARGIN, HGAP, LETTER_H, VGAP = 8.0, 10.0, 11.0, 6.0
W = WIDTH_MM * PT_PER_MM
CW = W - 2 * MARGIN

# (letter, file, title for the panel list); each row is filled left to right at a common height
ROWS = [
    [("a", H / "descriptive/figC_genomic_annotation_chromhmm.svg", "Genomic class of co-bound regions; promoter and enhancer share per TF")],
    [("b", H / "shared_pairs/fig_tf_x_shared_pairs_sorted.svg", "Cicero pairs >25 kb shared by >=3 TFs"),
     ("c", H / "figC_tf_cooccupancy_stratified_OR.svg", "TF x TF co-occupancy, stratified odds ratio (modules M1-M3)")],
    [("d", H / "clique_edge_overlap/hub_chr16_browser.svg", "chr16p11.2 promoter hub: Hi-C, loops, shared edges, per-TF coverage"),
     ("e", H / "clique_edge_overlap/hub_chr16_browser_zoom.svg", "Zoom-in on two hub windows: shared-edge anchors, per-TF coverage, ChromHMM, genes")],
    [("f", OUT / "panels/f_3d_model_panelB_crop.png", "3D model of the chr16p11.2 hub (PASTIS PM2; panel B of results_3d_model/v2, 600 dpi)"),
     ("g", H / "shared_pairs/fig_pair_concurrency.svg", "Observed vs expected TFs per pair (concurrency test)")],
]
EXTENDED = [
    [("a", H / "descriptive/figD_hic_support_per_tf.png", "Hi-C loop support of co-binding pairs per TF")],
]

def size(path):
    if path.suffix == ".svg":
        vb = re.search(r'viewBox="([^"]+)"', path.read_text()[:4000]).group(1).split()
        return float(vb[2]), float(vb[3])
    from PIL import Image
    return Image.open(path).size

def inline_svg(path, prefix, x, y, w, h):
    s = path.read_text()
    s = s[s.index("<svg"):]
    s = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{prefix}{m.group(1)}"', s)
    s = re.sub(r'url\(#([^)]+)\)', lambda m: f'url(#{prefix}{m.group(1)})', s)
    s = re.sub(r'(xlink:href|href)="#([^"]+)"', lambda m: f'{m.group(1)}="#{prefix}{m.group(2)}"', s)
    s = re.sub(r'<image([^>]*?) xlink:href="(data:[^"]+)"', r'<image\1 href="\2" xlink:href="\2"', s)
    # matplotlib default DejaVu Sans -> Arial (journal font; Illustrator and browsers resolve it)
    s = re.sub(r"(font: (?:\d+ )?[\d.]+px) '?DejaVu Sans'?(, '[^;\"]*?sans-serif)?", r"\1 Arial, Helvetica, sans-serif", s)
    s = re.sub(r"font-family: ?'?DejaVu Sans'?[^;\"]*", "font-family: Arial, Helvetica, sans-serif", s)
    head_end = s.index(">")
    head = re.sub(r'\s(width|height|x|y)="[^"]*"', "", s[:head_end])
    head += f' x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" preserveAspectRatio="xMinYMin meet"'
    return head + s[head_end:]

def embed_png(path, x, y, w, h):
    b = base64.b64encode(path.read_bytes()).decode()
    return (f'<image x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" preserveAspectRatio="xMinYMin meet" '
            f'href="data:image/png;base64,{b}" xlink:href="data:image/png;base64,{b}"/>')

def build(rows, name):
    body, y, legend = [], MARGIN, []
    for row in rows:
        aspects = [size(p)[0] / size(p)[1] for _, p, _ in row]
        h = (CW - HGAP * (len(row) - 1)) / sum(aspects)
        x = MARGIN
        for (letter, p, title), a in zip(row, aspects):
            w = a * h
            body.append(f'<text x="{x:.2f}" y="{y + 8:.2f}" font-family="Arial, Helvetica, sans-serif" '
                        f'font-size="9" font-weight="bold">{letter}</text>')
            body.append(inline_svg(p, f"p{name[-3:]}{letter}_", x, y + LETTER_H, w, h) if p.suffix == ".svg"
                        else embed_png(p, x, y + LETTER_H, w, h))
            legend.append(f"{letter}  {title}  [{p.relative_to(ROOT)}]  {w / PT_PER_MM:.0f} x {h / PT_PER_MM:.0f} mm")
            x += w + HGAP
        y += LETTER_H + h + VGAP
    Htot = y - VGAP + MARGIN
    svg = (f'<?xml version="1.0" encoding="utf-8"?>\n'
           f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" '
           f'width="{W:.2f}pt" height="{Htot:.2f}pt" viewBox="0 0 {W:.2f} {Htot:.2f}">\n'
           f'<rect width="100%" height="100%" fill="white"/>\n' + "\n".join(body) + "\n</svg>\n")
    (OUT / f"{name}.svg").write_text(svg)
    (OUT / f"{name}_panels.txt").write_text(f"{name}: {WIDTH_MM} x {Htot / PT_PER_MM:.0f} mm\n" + "\n".join(legend) + "\n")
    (OUT / f"{name}_preview.html").write_text(
        f'<!doctype html><html><head><meta charset="utf-8"><style>@page {{ size: {W:.2f}pt {Htot:.2f}pt; margin: 0 }}'
        f' html, body {{ margin: 0; background: white }} img {{ display: block; width: {W:.2f}pt; height: {Htot:.2f}pt }}</style></head>'
        f'<body><img src="{name}.svg"></body></html>\n')
    print(name, f"{WIDTH_MM} x {Htot / PT_PER_MM:.0f} mm"); print("\n".join(legend))

build(ROWS, "fig3_draft")
build(EXTENDED, "fig3_extended_data")
