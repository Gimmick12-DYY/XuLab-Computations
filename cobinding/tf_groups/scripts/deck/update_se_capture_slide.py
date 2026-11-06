"""Update the 'super-enhancer capture' slide in place: capture claim vs random genome windows (primary), active-chromatin
windows, and the density-matched result as a caveat. usage: python update_se_capture_slide.py deck.pptx"""
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
FIG = "/Users/dyy/Desktop/HEK293T-ATGTAG/tf_groups_figs/fig_se_capture_nulls.png"; INK = RGBColor(0x1A, 0x1A, 0x1A)
prs = Presentation(sys.argv[1])
s = next(sl for sl in prs.slides if any(sh.has_text_frame and sh.text_frame.text.startswith("Coordinated Binding Network- super-enhancer capture") for sh in sl.shapes))
sh = {x.name: x for x in s.shapes}
sh["Subtitle"].text_frame.paragraphs[0].runs[0].text = "SEdb HEK293T super-enhancers (SE_02_1951, n = 338) vs 1,000 sets of random same-size windows"
pic = [x for x in s.shapes if x.shape_type == 13][0]; pic._element.getparent().remove(pic._element)
s.shapes.add_picture(FIG, Inches(0.2), Inches(1.15), width=Inches(8.4))
def fill(box, paras, size, bold_all=False):
    tf = box.text_frame
    for p in list(tf.paragraphs)[1:]: p._p.getparent().remove(p._p)
    for r in list(tf.paragraphs[0].runs): r._r.getparent().remove(r._r)
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.space_after = Pt(6)
        for txt, b in para:
            r = p.add_run(); r.text = txt; r.font.name = "Cambria"; r.font.size = Pt(size); r.font.bold = b or bold_all; r.font.color.rgb = INK
fill(sh["Bullets"], [
    [("Any network peak: ", True), ("99% of SEs vs 76% of random windows.", False)],
    [("Cliques from ≥2 TFs: ", True), ("68% vs 19% (3.6×).", False)],
    [("Cliques from ≥5 TFs: ", True), ("29% vs 2.6% (11×).", False)],
    [("All comparisons: ", True), ("p = 0.001 (1,000 permutations, minimum possible).", False)],
    [("Per TF: ", True), ("60 of 68 TFs hit SEs more than random (p < 0.05), median 5.6×.", False)],
    [("Note: ", True), ("windows matched for peak density capture SEs equally well, so capture comes from SEs being dense peak clusters.", False)]], 13)
fill(sh["Conclusion"], [[("Conclusion: our TF data captures super-enhancers, nearly all SEs carry network peaks and most carry multi-TF cliques (p = 0.001).", True)]], 14)
prs.save(sys.argv[1]); print("updated")
