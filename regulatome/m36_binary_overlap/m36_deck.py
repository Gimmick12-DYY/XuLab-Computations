# m36: insert slide 5j (binary overlap) after 5i, before the Figure 5 supplement. usage: m36_deck.py <in> <out>
import sys, zipfile, collections
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from PIL import Image
IN, OUT = sys.argv[1], sys.argv[2]; MK = "FIG5V3"; F = "/work/users/d/y/dyy12/XuLab/regulatome/m34_fig5_panels_v3/plots/slide/Fig5j_binary_overlap.png"
p = Presentation(IN); n0 = len(p.slides)
def title_of(s): return next((sh.text_frame.text for sh in s.shapes if sh.name == f"{MK}-title" and sh.has_text_frame), "")
assert not any(title_of(s).startswith("Regulome- Figure 5j") for s in p.slides)
ix = [i for i, s in enumerate(p.slides) if title_of(s) == "Regulome- Figure 5 supplement"]; assert len(ix) == 1
lay = {l.name: l for l in p.slide_layouts}["Title and Content"]; sl = p.slides.add_slide(lay)
for ph in list(sl.placeholders): ph._element.getparent().remove(ph._element)
def tb(name, x, y, w, h, text, size, bold, color):
    s = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); s.name = name; tf = s.text_frame; tf.word_wrap = True; r = tf.paragraphs[0].add_run(); r.text = text
    f = r.font; f.name = "Cambria"; f.size = Pt(size); f.bold = bold; f.color.rgb = RGBColor.from_string(color)
tb(f"{MK}-title", 0.17, 0.12, 12.9, 0.57, "Regulome- Figure 5j · binary overlap, Regulome vs Perturb-seq", 28, True, "262626")
tb(f"{MK}-subtitle", 0.33, 0.68, 12.69, 0.4, "Each TF's genes as 1/0 in both; 11,337 genes in both = 17.7% of responders (expected 17.5%)", 15, False, "595959")
w, h = Image.open(F).size; pic = sl.shapes.add_picture(F, Inches(0.2), Inches(1.2), width=Emu(int(w / 300 * 914400)), height=Emu(int(h / 300 * 914400))); pic.name = f"{MK}-Fig5j_binary_overlap"
lst = p.slides._sldIdLst; el = list(lst)[-1]; lst.remove(el); lst.insert(ix[0], el)
p.part.rename_slide_parts([e.rId for e in lst])
for i, s in enumerate(p.slides, 1):
    if s.has_notes_slide: s.notes_slide.part.partname = s.notes_slide.part.partname.__class__(f"/ppt/notesSlides/notesSlide{i}.xml")
assert len(p.slides) == n0 + 1; p.save(OUT); q = Presentation(OUT); print("reopened", len(q.slides), "before", n0)
for i in range(8, 12): print(i + 1, title_of(q.slides[i]))
c = collections.Counter(zipfile.ZipFile(OUT).namelist()); d = [k for k, v in c.items() if v > 1]; print("duplicates", d); assert not d
