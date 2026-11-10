"""Swap the matrix picture on the two correlation slides, fix slide 6's subtitle and the table slide's footnote numbers. Nothing else changes. usage: edit_callouts.py in.pptx out.pptx"""
import json, sys
from pptx import Presentation
from pptx.util import Inches
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
IN, OUT = sys.argv[1], sys.argv[2]; p = Presentation(IN); FR = json.load(open("fig/frames.json")); slides = list(p.slides); ttl = lambda s: next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()), ""); seen = 0
for s in slides:
    t = ttl(s); f = "normmat_EZH2_MTF2" if t.startswith("PRC2 core EZH2–MTF2- layer 2") else ("normmat_MLX_SREBF2" if t.startswith("Novel MLX–SREBF2- layer 2") else None)
    if not f: continue
    pics = [sh for sh in s.shapes if sh.shape_type == 13]; assert len(pics) == 1, (t, len(pics)); o = pics[0]; print(t[:40], "old pos", round(o.left / 914400, 2), round(o.top / 914400, 2), round(o.width / 914400, 2), round(o.height / 914400, 2)); w, h = FR[f]; px = Image.open(f"fig/{f}.png").size; assert abs(px[0] / w - 600) < 2; L, Tp = o.left, o.top; o._element.getparent().remove(o._element); n = s.shapes.add_picture(f"fig/{f}.png", L, Tp, Inches(w), Inches(h)); n.name = "Matrix"; seen += 1
    if f.endswith("SREBF2"):
        for sh in s.shapes:
            if sh.name == "Subtitle":
                r = sh.text_frame.paragraphs[0].runs; r[0].text = "Normalized matrix: percentile 0.84 before correction, 0.44–0.81 after. MLX and SREBF2 join only high in the tree, so local ≈ global"
                for x in r[1:]: x.text = ""
assert seen == 2
hit = 0; m = next(i for i, s in enumerate(slides, 1) if ttl(s).startswith("Cobinding Analysis")); hic = next(i for i, s in enumerate(slides, 1) if ttl(s).startswith("4E-H12")); print("matrix slide", m, "Hi-C slide", hic)
for sh in slides[0].shapes:
    if sh.has_text_frame:
        for par in sh.text_frame.paragraphs:
            for r in par.runs:
                if "(slide 183)" in r.text: r.text = r.text.replace("(slide 183)", f"(slide {m})"); hit += 1
                if "triangle: slide 27" in r.text: r.text = r.text.replace("triangle: slide 27", f"triangle: slide {hic}"); hit += 1
assert hit in (0, 2), hit
assert not any("library block" in sh.text_frame.text.lower() for s in slides[:7] for sh in s.shapes if sh.has_text_frame), "library block wording left"
p.save(OUT); print("ok", len(p.slides))
