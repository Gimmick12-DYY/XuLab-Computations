"""Merge the two MLX-SREBF2 structure slides into one, drop the bar map, replace the PRC2 structure panel with its corrected labels, update the table footnote numbers, remove unused images. usage: edit_merge56.py in.pptx out.pptx"""
import json, sys
from pptx import Presentation
from pptx.util import Inches
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
p = Presentation(sys.argv[1]); slides = list(p.slides); ttl = lambda s: next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()), ""); i5 = next(i for i, s in enumerate(slides) if ttl(s) == "Novel MLX–SREBF2- layer 1: structure"); i6 = next(i for i, s in enumerate(slides) if ttl(s) == "Novel MLX–SREBF2 – layer 1: AF3 interface"); assert i6 == i5 + 1; s = slides[i5]; o = slides[i6]
foot = next((sh.text_frame.text for sh in s.shapes if sh.name == "Footnote"), ""); n5 = s.notes_slide.notes_text_frame.text if s.has_notes_slide else ""; n6 = o.notes_slide.notes_text_frame.text if o.has_notes_slide else ""
for sh in [x for x in s.shapes if x.name not in ("Title", "Subtitle")]: sh._element.getparent().remove(sh._element)
for sh in s.shapes:
    if sh.name == "Title":
        r = sh.text_frame.paragraphs[0].runs; r[0].text = "Novel MLX–SREBF2 – layer 1: AF3 structure"
        for x in r[1:]: x.text = ""
    if sh.name == "Subtitle":
        r = sh.text_frame.paragraphs[0].runs; r[0].text = "Contacts sit on the dimerisation helices; the basic DNA-binding regions stay free"
        for x in r[1:]: x.text = ""
f = "super/af3_MLX_merged.png"; px = Image.open(f).size; assert abs(px[0] / 12.9 - 600) < 2 and abs(px[1] / 5.6 - 600) < 2, px; s.shapes.add_picture(f, Inches(0.2), Inches(1.1), Inches(12.9), Inches(5.6))
extra = "Interface: HLH-ZIP 38 of 38 contact residues on MLX, 31 of 40 on SREBF2; basic region 0 and 1. Boltz-2: strong, trimmed and full length."
s.notes_slide.notes_text_frame.text = "\n".join(x for x in (n6, extra, foot, n5) if x)
s2 = next(x for x in slides if ttl(x).startswith("PRC2 core EZH2–MTF2- layer 1: structure")); pic = next(x for x in s2.shapes if x.shape_type == 13 and abs(x.width / 914400 - 6.6) < .01 and abs(x.height / 914400 - 4.85) < .01); g = "fig/struct_PRC2.png"; px = Image.open(g).size; assert px == (3960, 2910), px; _, rid = s2.part.get_or_add_image_part(g); pic._element.blipFill.find("{http://schemas.openxmlformats.org/drawingml/2006/main}blip").set("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed", rid)
Ls = p.slides._sldIdLst; e = list(Ls)[i6]; Ls.remove(e); p.part.drop_rel(e.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")); new = list(p.slides); m = next(k for k, x in enumerate(new, 1) if ttl(x).startswith("Cobinding Analysis")); h = next(k for k, x in enumerate(new, 1) if ttl(x).startswith("4E-H12")); hit = 0
for sh in new[0].shapes:
    if sh.has_text_frame:
        for par in sh.text_frame.paragraphs:
            for r in par.runs:
                if "(slide 183)" in r.text: r.text = r.text.replace("(slide 183)", f"(slide {m})"); hit += 1
                if "triangle: slide 27" in r.text: r.text = r.text.replace("triangle: slide 27", f"triangle: slide {h}"); hit += 1
assert hit == 2, hit
n_drop = 0
for x in new[:8]:
    xml = x.part._element.xml
    for rid, rel in list(x.part.rels.items()):
        if rel.reltype.endswith("/image") and f'"{rid}"' not in xml: x.part.rels.pop(rid); n_drop += 1
print("unused images dropped", n_drop, "| matrix", m, "Hi-C", h); p.save(sys.argv[2]); print("ok", len(new))
