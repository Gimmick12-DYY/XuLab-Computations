"""Set the final wording of the 6NQ3 comparison in the EZH2-MTF2 structure slide's speaker notes. usage: edit_note2.py in.pptx out.pptx"""
import sys
from pptx import Presentation
p = Presentation(sys.argv[1]); s = p.slides[1]; assert s.shapes[0].text_frame.text.startswith("PRC2 core EZH2–MTF2- layer 1: structure") or any(sh.has_text_frame and sh.text_frame.text.startswith("PRC2 core EZH2–MTF2- layer 1: structure") for sh in s.shapes); tf = s.notes_slide.notes_text_frame; assert "SUZ12 Q95–Y 2.7 Å in crystal vs 38.4 Å in model" in tf.text
tf.text = tf.text.replace("SUZ12 Q95–Y 2.7 Å in crystal vs 38.4 Å in model", "SUZ12 Q95–PHF19 Y542 2.7 Å in crystal vs SUZ12 Q95–MTF2 Y555 38.4 Å in model"); print(tf.text[-200:])
p.save(sys.argv[2]); print("ok", len(p.slides))
