"""Insert 2 slides at the front of the local deck (in place): SE capture by cliques vs nulls, and why peak density matters.
Figures from Longleaf cobinding/tf_groups/scripts/plot_se_slides.py. usage: python build_se_density_slides.py deck.pptx"""
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
FIG = "/Users/dyy/Desktop/HEK293T-ATGTAG/tf_groups_figs/"
INK, GREY = RGBColor(0x1A, 0x1A, 0x1A), RGBColor(0x8C, 0x8C, 0x8C)
def tb(s, x, y, w, h, paras, size, bold=False, color=INK, name="Text"):
    b = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); b.name = name; tf = b.text_frame; tf.word_wrap = True
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.space_after = Pt(6)
        for txt, bb in (para if isinstance(para, list) else [(para, bold)]):
            r = p.add_run(); r.text = txt; r.font.name = "Cambria"; r.font.size = Pt(size); r.font.bold = bb; r.font.color.rgb = color
    return b
def new_slide(prs, layout, title, sub):
    s = prs.slides.add_slide(layout)
    for ph in list(s.placeholders): ph._element.getparent().remove(ph._element)
    tb(s, 0.17, 0.12, 12.9, 0.57, [title], 28, True, name="Title"); tb(s, 0.33, 0.66, 12.69, 0.4, [sub], 15, False, GREY, name="Subtitle")
    return s
prs = Presentation(sys.argv[1]); lay = prs.slides[0].slide_layout; n0 = len(prs.slides)
s1 = new_slide(prs, lay, "Coordinated Binding Network- super-enhancer capture",
               "SEdb HEK293T super-enhancers (SE_02_1951) vs random windows of the same size")
s1.shapes.add_picture(FIG + "fig_se_capture_nulls.png", Inches(0.2), Inches(1.15), width=Inches(8.4))
tb(s1, 8.85, 1.35, 4.3, 4.4, [
    [("Observed: ", True), ("87% of SEs hold a clique region; 28% hold cliques from ≥5 TFs.", False)],
    [("Active-chromatin windows: ", True), ("only 8% reach ≥5 TFs (p < 0.005).", False)],
    [("Peak-density-matched windows: ", True), ("32% reach ≥5 TFs; SEs are not higher (p = 0.97).", False)],
    [("Clique TFs per SE: ", True), ("3.56 vs 3.96 matched (p = 0.001, lower).", False)]], 14, name="Bullets")
tb(s1, 0.3, 6.55, 12.6, 0.6, ["Conclusion: SEs capture many cliques only because they are dense peak clusters; at equal peak density they are not enriched."], 14, True, name="Conclusion")
s2 = new_slide(prs, lay, "Coordinated Binding Network- why peak density matters",
               "A clique needs several co-accessible peaks within 1 Mb; 52 TFs with ≥30 clique and non-clique peaks")
s2.shapes.add_picture(FIG + "fig_clique_density_effect.png", Inches(0.25), Inches(1.15), width=Inches(12.8))
tb(s2, 0.3, 6.15, 12.6, 0.9, [
    [("Matching: ", True), ("each peak compared only with peaks of the same TF in the same local-density decile.", False)],
    [("Conclusion: ", True), ("clique location follows peak density; GO of clique genes vs each TF's own network genes is empty (1 of 39 TFs).", True)]], 14, name="Conclusion")
ids = prs.slides._sldIdLst; new = list(ids)[n0:]
for el in new: ids.remove(el)
for i, el in enumerate(new): ids.insert(i, el)
prs.save(sys.argv[1]); print("slides now", len(prs.slides))
