# Copies the "new Hi-C loops" / "--xj" table slides (115-118) to right after 4E-H12 at slide 23; originals stay. usage: edit_dup_hic_tables.py deck out
import copy, sys
from pptx import Presentation
p = Presentation(sys.argv[1]); L = p.slides._sldIdLst; ids = list(L); S = list(p.slides); assert len(S) == 323, len(S)
def title(s): return next((x.text_frame.text for x in s.shapes if x.has_text_frame and x.text_frame.text.strip()), "")
assert title(S[22]).startswith("4E-H12") and title(S[23]) == "10/30/26 TF Analysis", (title(S[22]), title(S[23]))
src = S[114:118]; assert [any(getattr(x, "has_table", False) and x.has_table for x in s.shapes) for s in src] == [True] * 4 and all(len(s.part.rels) == 1 and not s.has_notes_slide for s in src)
n0 = len(ids); new = []
for s in src:
    d = p.slides.add_slide(s.slide_layout); t = d.shapes._spTree
    for x in list(t)[2:]: t.remove(x)
    for x in list(s.shapes._spTree)[2:]: t.append(copy.deepcopy(x))
all_ = list(L); new = all_[n0:]; anchor = ids[22]
for x in new: L.remove(x)
for x in new: anchor.addnext(x); anchor = x
p.save(sys.argv[2]); print("ok", len(p.slides))
