"""Assemble Fig 3 v7 (183 mm wide, <= 247 mm) from the final-size panels in panels/ (prep_panels_v7.py). Panels a, b, the c-e legend and
e are drawn at their final size (placed at 1x); c (browser) and d (rosette) were drawn on their original canvases with fonts preset for
their placement scale. Each panel is a named group <g id="panel_x"> holding a nested <svg> cropped to its content; text stays live
Arial; letters 8 pt bold. PDF and 300 dpi PNG from rsvg-convert (Arial from /work/users/d/y/dyy12/fonts/arial). Runs on Longleaf.
Layout: row 1 a | b; row 2 c (full width); shared legend for c-e; row 3 d | e.
  python build_fig3_v7.py [outdir]"""
import os, re, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
HERE = Path(__file__).resolve().parent; PAN = HERE / "panels"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE; OUT.mkdir(parents=True, exist_ok=True)
PT = 72 / 25.4; W = 183.0; G = 3.0; INK = "#1A1A1A"
FC = HERE / "fonts.conf"
FC.write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig><dir>/work/users/d/y/dyy12/fonts/arial</dir>'
              '<include ignore_missing="yes">/etc/fonts/fonts.conf</include><cachedir>' + str(HERE / ".fccache") + '</cachedir></fontconfig>')
ENV = dict(os.environ, FONTCONFIG_FILE=str(FC))


def content_box(stem):
    s = (PAN / f"{stem}.svg").read_text(); vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', s).group(1).split()]
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "p.png"; subprocess.run(["rsvg-convert", "-z", "2", "-b", "white", "-o", str(png), str(PAN / f"{stem}.svg")], check=True, env=ENV)
        a = np.asarray(Image.open(png).convert("L")).astype(int)
    ys, xs = np.where(a < 245); k = a.shape[1] / vb[2]; pad = 1.0
    x0, x1 = max(xs.min() / k - pad, 0), min(xs.max() / k + pad, vb[2]); y0, y1 = max(ys.min() / k - pad, 0), min(ys.max() / k + pad, vb[3])
    return [vb[0] + x0, vb[1] + y0, x1 - x0, y1 - y0]


def inline(stem, letter, x, y, w, h, box, lx=0.0):
    s = (PAN / f"{stem}.svg").read_text(); m = re.search(r"<svg\b[^>]*>", s); body = s[m.end():s.rindex("</svg>")]
    body = re.sub(r"<metadata>.*?</metadata>", "", body, flags=re.S); p = f"{stem}_"
    body = re.sub(r'\bid="([^"]+)"', lambda k: f'id="{p}{k.group(1)}"', body)
    body = re.sub(r"url\(#([^)]+)\)", lambda k: f"url(#{p}{k.group(1)})", body)
    body = re.sub(r'(xlink:href|href)="#([^"]+)"', lambda k: f'{k.group(1)}="#{p}{k.group(2)}"', body)
    lab = (f'<text x="{(x + lx) * PT:.2f}" y="{(y + 2.6) * PT:.2f}" style="font-family: Arial, Helvetica, sans-serif; font-size: 8px; font-weight: bold; '
           f'fill: {INK}">{letter}</text>\n') if letter else ""
    vb = " ".join(f"{v:.3f}" for v in box)
    return (f'<g id="panel_{letter or stem}">\n<svg x="{x * PT:.3f}" y="{y * PT:.3f}" width="{w * PT:.3f}" height="{h * PT:.3f}" viewBox="{vb}" '
            f'preserveAspectRatio="xMinYMin meet" overflow="hidden">\n{body}\n</svg>\n{lab}</g>')


ST = ["a_module_edges", "b_class_bars", "c_se_browser", "cde_legend", "d_se_rosette", "e_se_flower"]
B = {st: content_box(st) for st in ST}; mm = lambda st: (B[st][2] / PT, B[st][3] / PT)
parts, rep = [], []
LM = 3.0   # left margin for letters
# row 1: a (1x) | b (1x)
wa, ha = mm("a_module_edges"); wb, hb = mm("b_class_bars"); y = 0.0
parts.append(inline("a_module_edges", "a", LM, y, wa, ha, B["a_module_edges"], lx=-LM)); xb = W - wb
assert LM + wa + G <= xb + 0.01, f"row 1 too wide: {LM + wa + G + wb:.1f} mm"
parts.append(inline("b_class_bars", "b", xb, y, wb, hb, B["b_class_bars"], lx=-LM)); rep += [("a", 1.0), ("b", 1.0)]; y += max(ha, hb) + G
# row 2: c, full width
wc, hc = mm("c_se_browser"); sc = (W - LM) / wc
parts.append(inline("c_se_browser", "c", LM, y, wc * sc, hc * sc, B["c_se_browser"], lx=-LM)); rep.append(("c", sc)); y += hc * sc + 1.0
wl, hl = mm("cde_legend"); parts.append(inline("cde_legend", "", LM, y, wl, hl, B["cde_legend"])); y += hl + G
# row 3: d | e (e at 1x; d scaled to e's height)
se = 89.0 / (9.4 * 25.4)   # flower_v7.py draws a 9.4 in canvas with fonts preset for 89 mm (SF there)
we, he = mm("e_se_flower"); we, he = we * se, he * se; wd, hd = mm("d_se_rosette"); sd = min(he / hd, (W - LM - we - G) / wd)
parts.append(inline("d_se_rosette", "d", LM, y, wd * sd, hd * sd, B["d_se_rosette"], lx=-LM))
parts.append(inline("e_se_flower", "e", W - we, y, we, he, B["e_se_flower"], lx=-LM)); rep += [("d", sd), ("e", se)]; y += max(hd * sd, he)
PAGE_H = y
svg = ('<?xml version="1.0" encoding="utf-8" standalone="no"?>\n'
       f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" width="{W}mm" height="{PAGE_H:.2f}mm" '
       f'viewBox="0 0 {W * PT:.3f} {PAGE_H * PT:.3f}">\n<rect id="page" width="{W * PT:.3f}" height="{PAGE_H * PT:.3f}" fill="#FFFFFF"/>\n' + "\n".join(parts) + "\n</svg>\n")
(OUT / "Fig3_v7.svg").write_text(svg)
subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(OUT / "Fig3_v7.pdf"), str(OUT / "Fig3_v7.svg")], check=True, env=ENV)
subprocess.run(["rsvg-convert", "-d", "300", "-p", "300", "-b", "white", "-o", str(OUT / "Fig3_v7_preview.png"), str(OUT / "Fig3_v7.svg")], check=True, env=ENV)
# smallest printed font: every live text size in each panel x its placement scale (content-box scale relative to the panel's own units)
small = []
for st, (l, s_) in zip(["a_module_edges", "b_class_bars", "c_se_browser", "d_se_rosette", "e_se_flower"], rep):
    t = (PAN / f"{st}.svg").read_text(); sz = [float(v) for v in re.findall(r"font[^;\"]*?([\d.]+)px", t)] + [float(v) for v in re.findall(r"font-size: ?([\d.]+)px", t)]
    small.append((l, round(min(sz) * s_, 2)))
tl = (PAN / "cde_legend.svg").read_text(); small.append(("legend", min(float(v) for v in re.findall(r"font[^;\"]*?([\d.]+)px", tl))))
print(f"page 183 x {PAGE_H:.1f} mm; scales", [(l, round(s, 3)) for l, s in rep]); print("smallest font per panel (pt):", small)
print(subprocess.run(["pdffonts", str(OUT / "Fig3_v7.pdf")], capture_output=True, text=True).stdout)
