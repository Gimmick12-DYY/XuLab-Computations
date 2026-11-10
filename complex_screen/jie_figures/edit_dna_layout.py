"""MLX-SREBF2 structure slide: no PAE, main panel in the annotated style, close-ups and bar map to the right. usage: edit_dna_layout.py in.pptx out.pptx"""
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
p = Presentation(sys.argv[1]); ttl = lambda s: next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()), ""); s = next(x for x in p.slides if ttl(x).startswith("Novel MLX–SREBF2 – layer 1"))
for sh in [x for x in s.shapes if x.shape_type == 13]: sh._element.getparent().remove(sh._element)
def pic(f, x, y, w, h):
    px = Image.open(f).size; assert abs(px[0] / w - 600) < 2 and abs(px[1] / h - 600) < 2, (f, px); s.shapes.add_picture(f, Inches(x), Inches(y), Inches(w), Inches(h))
pic("super/af3_MLX_dna_main.png", 0.1, 1.05, 6.3, 5.4); pic("super/af3_MLX_dna_close.png", 6.5, 1.05, 6.6, 2.8); pic("fig/sitemap_MLX_dna.png", 6.5, 3.9, 6.6, 1.95)
sc = next(x for x in s.shapes if x.name == "Scores"); ps = sc.text_frame.paragraphs; txt = ["AF3 3.0, bHLH-ZIP + E-box DNA, 5 seeds: ", "ipTM 0.87, ipSAE 0.74; protein–DNA ipTM 0.77 / 0.74"]; r0 = ps[0].runs; r0[0].text = txt[0] + txt[1]; r0[0].font.bold = False; r0[0].font.size = Pt(13)
for x in r0[1:]: x.text = ""
for q in list(ps)[1:]: q._p.getparent().remove(q._p)
sc.left, sc.top, sc.width, sc.height = Inches(6.55), Inches(5.87), Inches(6.55), Inches(.55); xml = s.part._element.xml; n = 0
for rid, rel in list(s.part.rels.items()):
    if rel.reltype.endswith("/image") and f'"{rid}"' not in xml: s.part.rels.pop(rid); n += 1
print("unused images dropped", n); p.save(sys.argv[2]); print("ok", len(list(p.slides)))
