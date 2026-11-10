"""Change only the EZH2-MTF2 structure slide's subtitle. usage: edit_sub2.py in.pptx out.pptx"""
import sys
from pptx import Presentation
p = Presentation(sys.argv[1]); s = next(x for x in p.slides if any(sh.has_text_frame and sh.text_frame.text.startswith("PRC2 core EZH2–MTF2- layer 1: structure") for sh in x.shapes)); sub = next(sh for sh in s.shapes if sh.name == "Subtitle"); r = sub.text_frame.paragraphs[0].runs; r[0].text = "AF3 connects EZH2 and MTF2 through SUZ12, not directly; the exact interface is uncertain"
for x in r[1:]: x.text = ""
p.save(sys.argv[2]); print("ok", len(p.slides))
