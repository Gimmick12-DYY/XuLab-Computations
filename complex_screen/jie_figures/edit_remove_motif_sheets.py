# Removes the 13 motif copy slides from the top of the deck, identified by title. usage: edit_remove_motif_sheets.py deck out
import sys
from pptx import Presentation
p = Presentation(sys.argv[1]); L = p.slides._sldIdLst; ids = list(L); n0 = len(ids); assert n0 == 336, n0
def title(s): return next((x.text_frame.text for x in s.shapes if x.has_text_frame and x.text_frame.text.strip()), "")
T = [title(s) for s in list(p.slides)[:14]]; want = ["11/3/26 Motif results (MLX, SREBF2)"] + [f"{t} · {r} · {b}" for t in ("SREBF2", "MLX") for r in ("motif_bins", "motif") for b in ("homer_ocr", "homer", "ame_ocr + ame")]
assert T[:13] == want, T[:13]; assert T[13].startswith("Cobinding Complex"), T[13]
for x in ids[:13]: p.part.drop_rel(x.rId); L.remove(x)
p.save(sys.argv[2]); print("ok", len(p.slides))
