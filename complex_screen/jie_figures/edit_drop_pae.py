"""Remove the PAE plot from the MLX-SREBF2 structure slide and move the scores into its place. usage: edit_drop_pae.py in.pptx out.pptx"""
import sys
from pptx import Presentation
from pptx.util import Inches
p = Presentation(sys.argv[1]); ttl = lambda s: next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()), ""); s = next(x for x in p.slides if ttl(x).startswith("Novel MLX–SREBF2 – layer 1")); pa = next(x for x in s.shapes if x.shape_type == 13 and abs(x.width / 914400 - 2.6) < .01 and abs(x.height / 914400 - 1.95) < .01); pa._element.getparent().remove(pa._element); sc = next(x for x in s.shapes if x.name == "Scores"); sc.left, sc.width = Inches(0.3), Inches(5.4); xml = s.part._element.xml; n = 0
for rid, rel in list(s.part.rels.items()):
    if rel.reltype.endswith("/image") and f'"{rid}"' not in xml: s.part.rels.pop(rid); n += 1
print("unused images dropped", n); p.save(sys.argv[2]); print("ok", len(list(p.slides)))
