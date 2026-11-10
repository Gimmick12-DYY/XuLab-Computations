"""Slide 5 made consistent with the hand-edited table: no footnote, subtitle range and conclusion match the table. usage: edit_s5_clean.py in.pptx out.pptx"""
import sys
from pptx import Presentation
p = Presentation(sys.argv[1]); slides = list(p.slides); ttl = lambda s: next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()), ""); s = next(x for x in slides if ttl(x).startswith("Novel MLX–SREBF2- layer 2"))
for sh in [x for x in s.shapes if x.name == "Footnote"]: sh._element.getparent().remove(sh._element)
def st(q, v):
    rr = q.runs; rr[0].text = v
    for x in rr[1:]: x.text = ""
sb = next(x for x in s.shapes if x.name == "Subtitle").text_frame.paragraphs[0]; assert "0.44–0.81" in sb.text, sb.text; st(sb, sb.text.replace("0.44–0.81", "0.44–0.78").strip())
c = next(x for x in s.shapes if x.name == "Conclusion").text_frame.paragraphs[0]; rr = c.runs; assert rr[0].text.startswith("Conclusion"), rr[0].text; rr[1].text = "MLX–SREBF2 is moderate overall (0.71 on all elements), higher on enhancer (0.78) and low on repressive (0.44)."
for x in rr[2:]: x.text = ""
s3 = next(x for x in slides if ttl(x).startswith("PRC2 core EZH2–MTF2- layer 2")); tx = " ".join(x.text_frame.text for x in s3.shapes if x.has_text_frame); print("slide 3 leftover references:", [w for w in ("Polycomb", "local", "Local", "orrected") if w in tx])
p.save(sys.argv[2]); print("ok", len(slides))
