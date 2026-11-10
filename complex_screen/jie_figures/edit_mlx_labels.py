"""Replace the merged MLX-SREBF2 figure with its corrected leader lines and label colours; remove the unused image. usage: edit_mlx_labels.py in.pptx out.pptx"""
import sys
from pptx import Presentation
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
p = Presentation(sys.argv[1]); slides = list(p.slides); ttl = lambda s: next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()), ""); s = next(x for x in slides if ttl(x) == "Novel MLX–SREBF2 – layer 1: AF3 structure"); pic = next(x for x in s.shapes if x.shape_type == 13 and abs(x.width / 914400 - 12.9) < .01); f = "super/af3_MLX_merged.png"; px = Image.open(f).size; assert px == (7740, 3360), px; _, rid = s.part.get_or_add_image_part(f); pic._element.blipFill.find("{http://schemas.openxmlformats.org/drawingml/2006/main}blip").set("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed", rid); xml = s.part._element.xml; n = 0
for r, rel in list(s.part.rels.items()):
    if rel.reltype.endswith("/image") and f'"{r}"' not in xml: s.part.rels.pop(r); n += 1
print("unused images dropped", n); p.save(sys.argv[2]); print("ok", len(slides))
