# Moves the 10/30 header and Regulome status slide (7-8) below slide 25, so complex analysis and coordinated binding open the deck. usage: edit_move_s7_8.py deck out
import sys
from pptx import Presentation
p = Presentation(sys.argv[1]); L = p.slides._sldIdLst; ids = list(L); assert len(ids) == 323, len(ids)
def title(s): return next((x.text_frame.text for x in s.shapes if x.has_text_frame and x.text_frame.text.strip()), "")
S = list(p.slides); assert title(S[6]) == "10/30/26 TF Analysis" and title(S[7]).startswith("Regulome- where the findings") and title(S[24]).startswith("4E-H12") and title(S[8]).startswith("Coordinated Binding Network"), [title(S[i]) for i in (6, 7, 8, 24)]
a, b = ids[6], ids[7]; L.remove(a); L.remove(b); anchor = ids[24]; anchor.addnext(a); a.addnext(b)
p.save(sys.argv[2]); print("ok", len(p.slides))
