"""MLX-SREBF2 correlation slide: enhancer matrix with PowerNorm gamma 0.6, two-tone box and crosshair, same frame. usage: edit_s5_g06.py in.pptx out.pptx"""
import sys
from pptx import Presentation
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"; R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
p = Presentation(sys.argv[1]); s = next(x for x in p.slides if any(sh.has_text_frame and "MLX–SREBF2- layer 2" in sh.text_frame.text for sh in x.shapes)); assert p.slides.index(s) == 4; pc = next(x for x in s.shapes if x.name == "Matrix"); f = "fig/heatmap_enh_MLX_SREBF2_g06.png"; assert Image.open(f).size == (4200, 3750); b = pc._element.blipFill.find(f"{A}blip"); old = b.get(R); _, rid = s.part.get_or_add_image_part(f); b.set(R, rid)
if f'"{old}"' not in s.part._element.xml: s.part.rels.pop(old)
p.save(sys.argv[2]); print("ok", len(list(p.slides)))
