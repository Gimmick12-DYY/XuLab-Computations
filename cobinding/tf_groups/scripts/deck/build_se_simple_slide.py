"""Replace the two SE/density slides at the front of the local deck with one simple SE-capture slide (in place).
usage: python build_se_simple_slide.py deck.pptx"""
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
FIG = "/Users/dyy/Desktop/HEK293T-ATGTAG/tf_groups_figs/fig_se_capture_simple.png"
INK, GREY = RGBColor(0x1A, 0x1A, 0x1A), RGBColor(0x8C, 0x8C, 0x8C)
def tb(s, x, y, w, h, paras, size, color=INK, name="Text"):
    b = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); b.name = name; tf = b.text_frame; tf.word_wrap = True
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.space_after = Pt(10)
        for txt, bb in para:
            r = p.add_run(); r.text = txt; r.font.name = "Cambria"; r.font.size = Pt(size); r.font.bold = bb; r.font.color.rgb = color
prs = Presentation(sys.argv[1])
old = [i for i, sl in enumerate(prs.slides) if any(sh.has_text_frame and sh.text_frame.text.startswith(("Coordinated Binding Network- super-enhancer capture",
       "Coordinated Binding Network- why peak density matters")) for sh in sl.shapes)]
s = prs.slides.add_slide(prs.slides[0].slide_layout)
for ph in list(s.placeholders): ph._element.getparent().remove(ph._element)
tb(s, 0.17, 0.12, 12.9, 0.57, [[("Coordinated Binding Network- super-enhancer capture", True)]], 28, name="Title")
tb(s, 0.33, 0.66, 12.69, 0.4, [[("SEdb HEK293T super-enhancers vs random genome windows of the same size", False)]], 15, GREY, name="Subtitle")
s.shapes.add_picture(FIG, Inches(0.3), Inches(1.2), width=Inches(7.6))
tb(s, 8.3, 1.6, 4.8, 4.0, [[("99%", True), (" of SEs have a TF peak (vs 76%)", False)],
                           [("68%", True), (" have a clique of ≥2 TFs (vs 19%)", False)],
                           [("29%", True), (" have a clique of ≥5 TFs (vs 3%)", False)],
                           [("p = 0.001", True), (" for all three (1,000 permutations)", False)]], 18, name="Numbers")
tb(s, 0.3, 6.45, 12.6, 0.5, [[("Conclusion: our TF data captures super-enhancers.", True)]], 16, name="Conclusion")
ids = prs.slides._sldIdLst; new = list(ids)[-1]; ids.remove(new); ids.insert(0, new)
for i in sorted([i + 1 for i in old], reverse=True):      # indices shifted by the inserted slide; delete after adding
    el = list(ids)[i]; prs.part.drop_rel(el.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")); ids.remove(el)
prs.save(sys.argv[1]); print("removed", len(old), "slides; total", len(prs.slides))
