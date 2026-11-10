"""EZH2-MTF2 correlation slide: matrix picture with the pair boxes on their own cells; nothing else changes. usage: edit_s3_box.py in.pptx out.pptx"""
import hashlib, sys
from pptx import Presentation
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"; R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
p = Presentation(sys.argv[1]); s = p.slides[2]; assert "layer 2" in s.shapes[0].text_frame.text and "EZH2" in s.shapes[0].text_frame.text; pc = next(x for x in s.shapes if x.name == "Matrix"); print("old picture is heatmap_all_EZH2_MTF2.png:", hashlib.md5(pc.image.blob).hexdigest() == hashlib.md5(open("fig/heatmap_all_EZH2_MTF2.png", "rb").read()).hexdigest()); f = "fig/heatmap_all_EZH2_MTF2_boxfix.png"; assert Image.open(f).size == (4200, 3750); b = pc._element.blipFill.find(f"{A}blip"); old = b.get(R); _, rid = s.part.get_or_add_image_part(f); b.set(R, rid)
if f'"{old}"' not in s.part._element.xml: s.part.rels.pop(old)
p.save(sys.argv[2]); print("ok", len(list(p.slides)))
