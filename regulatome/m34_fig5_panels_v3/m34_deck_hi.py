# m34 Amendment 1: insert slides 5h and 5i before the Figure 5 supplement slide; update the 5c subtitle. usage: m34_deck_hi.py <in> <out>
import sys, zipfile, collections
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from PIL import Image
IN, OUT = sys.argv[1], sys.argv[2]; PL = "/work/users/d/y/dyy12/XuLab/regulatome/m34_fig5_panels_v3/plots/slide"; MK = "FIG5V3"
NEW = [("Fig5h_threshold_sweep", "Regulome- Figure 5h · links found, by knockdown threshold", "Share of Perturb-seq links where our Regulome has the TF bound; dashed = expected from edge count"),
       ("Fig5i_high_confidence_per_tf", "Regulome- Figure 5i · high-confidence links per TF", "Per TF, % of Tier-1 links (top) and |z| ≥ 4 links (bottom) found, against the TF's expected share")]
p = Presentation(IN); n0 = len(p.slides)
def title_of(s):
    for sh in s.shapes:
        if sh.name == f"{MK}-title" and sh.has_text_frame: return sh.text_frame.text
    return ""
ix_supp = [i for i, s in enumerate(p.slides) if title_of(s) == "Regulome- Figure 5 supplement"]; ix_c = [i for i, s in enumerate(p.slides) if title_of(s).startswith("Regulome- Figure 5c")]
assert len(ix_supp) == 1 and len(ix_c) == 1, (ix_supp, ix_c)
assert not any(title_of(s).startswith("Regulome- Figure 5h") for s in p.slides), "5h already present"
for sh in p.slides[ix_c[0]].shapes:
    if sh.name == f"{MK}-subtitle":
        r = sh.text_frame.paragraphs[0].runs; r[0].text = "On a par with bulk ChIP for 5 of 10 TFs, bulk better for 5; some references are not HEK293T"
        for extra in r[1:]: extra.text = ""
lay = {l.name: l for l in p.slide_layouts}["Title and Content"]
def tb(sl, name, x, y, w, h, text, size, bold, color):
    s = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); s.name = name; tf = s.text_frame; tf.word_wrap = True; r = tf.paragraphs[0].add_run(); r.text = text
    f = r.font; f.name = "Cambria"; f.size = Pt(size); f.bold = bold; f.color.rgb = RGBColor.from_string(color)
added = []
for f, title, sub in NEW:
    sl = p.slides.add_slide(lay)
    for ph in list(sl.placeholders): ph._element.getparent().remove(ph._element)
    tb(sl, f"{MK}-title", 0.17, 0.12, 12.9, 0.57, title, 28, True, "262626"); tb(sl, f"{MK}-subtitle", 0.33, 0.68, 12.69, 0.4, sub, 15, False, "595959")
    w, h = Image.open(f"{PL}/{f}.png").size; pic = sl.shapes.add_picture(f"{PL}/{f}.png", Inches(0.2), Inches(1.2), width=Emu(int(w / 300 * 914400)), height=Emu(int(h / 300 * 914400))); pic.name = f"{MK}-{f}"
    print(f, w, h, round(pic.width / 914400, 2), round(pic.height / 914400, 2)); added.append(sl)
lst = p.slides._sldIdLst; els = list(lst); new_els = els[-2:]
for el in new_els: lst.remove(el)
supp_el = list(lst)[ix_supp[0]]; pos = list(lst).index(supp_el)
for k, el in enumerate(new_els): lst.insert(pos + k, el)
p.part.rename_slide_parts([el.rId for el in lst])
for i, s in enumerate(p.slides, 1):
    if s.has_notes_slide: s.notes_slide.part.partname = s.notes_slide.part.partname.__class__(f"/ppt/notesSlides/notesSlide{i}.xml")
names = [s.part.partname for s in p.slides]; assert len(set(names)) == len(names); assert len(p.slides) == n0 + 2
p.save(OUT); q = Presentation(OUT); print("reopened", len(q.slides), "before", n0)
for i in range(11): print(i + 1, title_of(q.slides[i]) or "(divider/other)", "|", next((sh.text_frame.text for sh in q.slides[i].shapes if sh.name == f"{MK}-subtitle" and sh.has_text_frame), "")[:95])
c = collections.Counter(zipfile.ZipFile(OUT).namelist()); dup = [k for k, v in c.items() if v > 1]; print("duplicates", dup); assert not dup
