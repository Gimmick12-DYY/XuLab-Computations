"""Delete the EZH2-MTF2 model-vs-experiment slide, keep its finding in the speaker notes of the structure slide, update the table footnote's slide numbers. usage: edit_drop3.py in.pptx out.pptx"""
import sys
from pptx import Presentation
p = Presentation(sys.argv[1]); slides = list(p.slides); ttl = lambda s: next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()), "")
ix = [i for i, s in enumerate(slides) if ttl(s) == "PRC2 core EZH2–MTF2- layer 1: model vs experiment"]; assert len(ix) == 1, ix; i = ix[0]; st = next(j for j, s in enumerate(slides) if ttl(s) == "PRC2 core EZH2–MTF2- layer 1: structure")
NOTE = "Compared with PRC2–PHF19 (PDB 6NQ3; PHF19 is the MTF2 paralog), AF3 places the MTF2 C-terminus on a different SUZ12 surface (2/20 contact residues shared; SUZ12 Q95–Y 2.7 Å in crystal vs 38.4 Å in model), so the exact predicted interface should not be trusted."
tf = slides[st].notes_slide.notes_text_frame; tf.text = (tf.text + "\n" + NOTE).strip()
Ls = p.slides._sldIdLst; e = list(Ls)[i]; Ls.remove(e); p.part.drop_rel(e.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")); new = list(p.slides)
m = next(k for k, s in enumerate(new, 1) if ttl(s).startswith("Cobinding Analysis")); h = next(k for k, s in enumerate(new, 1) if ttl(s).startswith("4E-H12")); hit = 0
for sh in new[0].shapes:
    if sh.has_text_frame:
        for par in sh.text_frame.paragraphs:
            for r in par.runs:
                if "(slide 184)" in r.text: r.text = r.text.replace("(slide 184)", f"(slide {m})"); hit += 1
                if "triangle: slide 28" in r.text: r.text = r.text.replace("triangle: slide 28", f"triangle: slide {h}"); hit += 1
assert hit == 2, hit; p.save(sys.argv[2]); print("ok", len(new), "matrix", m, "Hi-C", h)
