"""Assemble Fig 3 v6 on one 183 mm wide page from the approved-figure SVGs in panels/ (made by prep_panels_v6.py).
Each panel SVG is inlined as <g id="panel_x"> holding a nested <svg> whose viewBox is the panel's content box (found by rendering
the panel once with rsvg-convert and trimming the white margin), so marks keep their original geometry; ids are prefixed per panel;
text stays live (Arial). Scaling is uniform per row (both halves of a row share one scale). Panel letters 8 pt bold Arial.
PDF + 300 dpi PNG are rendered from the page SVG with rsvg-convert (Arial via fonts.conf -> /work/users/d/y/dyy12/fonts/arial).
Runs on Longleaf:  python build_fig3_v6.py [outdir]"""
import os, re, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent; PAN = HERE / "panels"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE; OUT.mkdir(parents=True, exist_ok=True)
PT = 72 / 25.4; W = 183.0; G = 3.0          # page width, gutter (mm)
FC = HERE / "fonts.conf"
FC.write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig><dir>/work/users/d/y/dyy12/fonts/arial</dir>'
              '<include ignore_missing="yes">/etc/fonts/fonts.conf</include><cachedir>' + str(HERE / ".fccache") + '</cachedir></fontconfig>')
ENV = dict(os.environ, FONTCONFIG_FILE=str(FC))
INK = "#1A1A1A"


def content_box(stem):
    """viewBox of the non-white content of panels/<stem>.svg, in the panel's own user units (pt)."""
    s = (PAN / f"{stem}.svg").read_text()
    vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', s).group(1).split()]
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "p.png"
        subprocess.run(["rsvg-convert", "-z", "2", "-b", "white", "-o", str(png), str(PAN / f"{stem}.svg")], check=True, env=ENV)
        a = np.asarray(Image.open(png).convert("L")).astype(int)
    ys, xs = np.where(a < 245); k = a.shape[1] / vb[2]
    pad = 2.0
    x0, x1 = max(xs.min() / k - pad, 0), min(xs.max() / k + pad, vb[2]); y0, y1 = max(ys.min() / k - pad, 0), min(ys.max() / k + pad, vb[3])
    return [vb[0] + x0, vb[1] + y0, x1 - x0, y1 - y0]


def inline(stem, letter, x, y, w, h, box):
    s = (PAN / f"{stem}.svg").read_text()
    m = re.search(r"<svg\b[^>]*>", s); body = s[m.end():s.rindex("</svg>")]
    body = re.sub(r"<metadata>.*?</metadata>", "", body, flags=re.S)
    p = f"{stem}_"
    body = re.sub(r'\bid="([^"]+)"', lambda k: f'id="{p}{k.group(1)}"', body)
    body = re.sub(r"url\(#([^)]+)\)", lambda k: f"url(#{p}{k.group(1)})", body)
    body = re.sub(r'(xlink:href|href)="#([^"]+)"', lambda k: f'{k.group(1)}="#{p}{k.group(2)}"', body)
    vb = " ".join(f"{v:.3f}" for v in box)
    lab = (f'<text x="{(x + 0.1) * PT:.2f}" y="{(y + 2.6) * PT:.2f}" style="font-family: Arial, Helvetica, sans-serif; font-size: 8px; '
           f'font-weight: bold; fill: {INK}">{letter}</text>\n') if letter else ""
    return (f'<g id="panel_{stem}">\n<svg x="{x * PT:.3f}" y="{y * PT:.3f}" width="{w * PT:.3f}" height="{h * PT:.3f}" '
            f'viewBox="{vb}" preserveAspectRatio="xMinYMin meet" overflow="hidden">\n{body}\n</svg>\n{lab}</g>')   # letter on top of the panel background


def network_legend(x, y):
    """legend of panel a, same handles as the approved figure, v6 wording, 5 pt live text"""
    fs = 5.0; dy = 2.45; out = [f'<g id="panel_a_legend" transform="translate({x * PT:.3f} {y * PT:.3f})">']
    items = [("line", "#e8433f", 1.0, "higher-order co-binding module"), ("line", "#c8ccd2", 0.6, "surrounding pair"),
             ("cased", "#FF1493", 1.3, "loop-supported module edge"), ("cased", "#F2B705", 1.3, "module with every edge on a loop (15)"),
             ("ring", "#FF1493", 0, "module peak on that loop")]
    for i, (kind, col, lw, lab) in enumerate(items):
        yy = (i * dy + 1.2) * PT; x0, x1 = 0, 4.2 * PT
        if kind == "line": out.append(f'<line x1="{x0}" y1="{yy:.2f}" x2="{x1:.2f}" y2="{yy:.2f}" style="stroke: {col}; stroke-width: {lw}"/>')
        elif kind == "cased":
            out.append(f'<line x1="{x0}" y1="{yy:.2f}" x2="{x1:.2f}" y2="{yy:.2f}" style="stroke: #111111; stroke-width: 2.6"/>')
            out.append(f'<line x1="{x0}" y1="{yy:.2f}" x2="{x1:.2f}" y2="{yy:.2f}" style="stroke: #ffffff; stroke-width: 1.9"/>')
            out.append(f'<line x1="{x0}" y1="{yy:.2f}" x2="{x1:.2f}" y2="{yy:.2f}" style="stroke: {col}; stroke-width: {lw}"/>')
        else:
            out.append(f'<circle cx="{x1 / 2:.2f}" cy="{yy:.2f}" r="2.2" style="fill: none; stroke: #111111; stroke-width: 0.9"/>')
            out.append(f'<circle cx="{x1 / 2:.2f}" cy="{yy:.2f}" r="2.2" style="fill: none; stroke: {col}; stroke-width: 0.55"/>')
        out.append(f'<text x="{5.4 * PT:.2f}" y="{yy + fs * 0.35:.2f}" style="font-family: Arial, Helvetica, sans-serif; font-size: {fs}px; fill: #222222">{lab}</text>')
    out.append("</g>"); return "\n".join(out)


B = {st: content_box(st) for st in ["a_network", "b_class_bars", "c1_pairs_loop_support", "c2_module_loop_coverage",
                                     "d1_se_browser", "d2_se_edge_table", "e1_se_rosette", "e2_se_flower"]}
mm = lambda st: (B[st][2] / PT, B[st][3] / PT)    # native content size, mm
parts, y, report = [], 0.0, []

# row 1: a network | its legend | b class bars (same height)
H1 = 56.0; LEGW = 36.0
wa = H1 * mm("a_network")[0] / mm("a_network")[1]; sb = H1 / mm("b_class_bars")[1]; wb = mm("b_class_bars")[0] * sb
if wa + LEGW + wb + 2 * G > W: sb = (W - wa - LEGW - 2 * G) / mm("b_class_bars")[0]; wb = W - wa - LEGW - 2 * G
parts.append(inline("a_network", "a", 0, y, wa, H1, B["a_network"])); parts.append(network_legend(wa + G, y + H1 - 5 * 2.45 - 1))
parts.append(inline("b_class_bars", "b", W - wb, y, wb, mm("b_class_bars")[1] * sb, B["b_class_bars"]))
report += [("a", H1 / mm("a_network")[1]), ("b", sb)]; y += H1 + G

# rows 2-4: two panels sharing one scale, filling the width
def pair_row(l, s1, s2, letters, max_h=None):
    global y
    s = (W - G) / (mm(s1)[0] + mm(s2)[0])
    if max_h and s * max(mm(s1)[1], mm(s2)[1]) > max_h: s = max_h / max(mm(s1)[1], mm(s2)[1])
    w1, h1 = mm(s1)[0] * s, mm(s1)[1] * s; w2, h2 = mm(s2)[0] * s, mm(s2)[1] * s
    x2 = W - w2 if not max_h else w1 + G
    parts.append(inline(s1, letters[0], 0, y, w1, h1, B[s1])); parts.append(inline(s2, letters[1], x2, y, w2, h2, B[s2]))
    report.append((l, s)); y += max(h1, h2) + G

pair_row("c", "c1_pairs_loop_support", "c2_module_loop_coverage", ("c", ""))
pair_row("d", "d1_se_browser", "d2_se_edge_table", ("d", ""), max_h=float(os.environ.get("H_D", 55)))
pair_row("e", "e1_se_rosette", "e2_se_flower", ("e", ""))
PAGE_H = y - G
svg = ('<?xml version="1.0" encoding="utf-8" standalone="no"?>\n'
       f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" '
       f'width="{W}mm" height="{PAGE_H:.2f}mm" viewBox="0 0 {W * PT:.3f} {PAGE_H * PT:.3f}">\n'
       f'<rect id="page" width="{W * PT:.3f}" height="{PAGE_H * PT:.3f}" fill="#FFFFFF"/>\n' + "\n".join(parts) + "\n</svg>\n")
(OUT / "Fig3_v6.svg").write_text(svg)
subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(OUT / "Fig3_v6.pdf"), str(OUT / "Fig3_v6.svg")], check=True, env=ENV)
subprocess.run(["rsvg-convert", "-d", "300", "-p", "300", "-b", "white", "-o", str(OUT / "Fig3_v6_preview.png"), str(OUT / "Fig3_v6.svg")], check=True, env=ENV)
print(f"page 183 x {PAGE_H:.1f} mm"); [print(f"  panel {l}: scale {s:.3f} (a 12 pt label prints at {12 * s:.1f} pt, 5.5 pt at {5.5 * s:.1f} pt)") for l, s in report]
print(subprocess.run(["pdffonts", str(OUT / "Fig3_v6.pdf")], capture_output=True, text=True).stdout)
