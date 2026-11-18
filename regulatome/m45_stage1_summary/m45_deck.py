# m45: add a "11/18/26 Regulome Stage 1" section at the top of the local deck (newest-first). usage: m45_deck.py <in.pptx> <out.pptx>
import sys, zipfile, collections, re
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from PIL import Image
IN, OUT = sys.argv[1], sys.argv[2]; PL = "/work/users/d/y/dyy12/XuLab/regulatome/m45_stage1_summary/plots/slide"; MK = "FIG5S1"
S = [("S1_schematic", "Regulome- Stage 1 · setup", "Two dials only: the Perturb-seq cutoff and the padding around the gene start; 63 TFs, HEK293T", None),
     ("S2_recall_by_cutoff_and_padding", "Regulome- Stage 1 · recall by cutoff and padding", "Recall = share of Perturb-seq genes found in our Regulome; 63 TFs, top-25% sgRNA cells", None),
     ("S3_recall_vs_background", "Regulome- Stage 1 · recall against background", "63 TFs; matched background = the same TF's non-changing genes (|z| < 1), matched on expression and promoter accessibility", None),
     ("S4_dose_response", "Regulome- Stage 1 · share bound by response strength", "55 TFs vs the 8 broad TFs with a knockdown (> 4,000 genes at padding 0); top-25% sgRNA cells", None),
     ("S5_broad_tfs", "Regulome- Stage 1 · broad TFs", "All 63 TFs, 55 without broad TFs, 8 broad TFs only; top-25% sgRNA cells", "Broad TFs give coverage but are flat across response strength.")]
BUL = ["Recall rises with the cutoff: padding 0, 17.7% → 24.4% (|z| ≥ 2 → 5); 10 kb, 22.7% → 36.6%; 50 kb, 36.5% → 50.9%",
       "Against matched background: padding 0 is level at |z| ≥ 2–3 (17.7 vs 18.2%) and ahead at |z| ≥ 4–5 (24.4 vs 21.4%); padding 10 kb is ahead at every cutoff (36.6 vs 26.8% at |z| ≥ 5)",
       "Without the 8 broad TFs the gap is larger (10 kb, |z| ≥ 5: 35.0 vs 21.7%); broad TFs add coverage (45–64%) but stay flat",
       "TF specificity is weak: own list ≈ similar-size lists at |z| ≥ 2 (17.7 vs 17.7%), a few points ahead at |z| ≥ 4–5 (+1.9 and +5.4 points; 57 and 33 TFs)",
       "Cutoff and padding not chosen yet; more padding raises recall and list size together (median 1,429 → 3,511 genes per TF)"]
p = Presentation(IN); n0 = len(p.slides); lay = {l.name: l for l in p.slide_layouts}
assert not any(sh.name.startswith(MK) for s in p.slides for sh in s.shapes), "section already present"
def tb(sl, name, x, y, w, h, text, size, bold, color):
    s = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); s.name = name; tf = s.text_frame; tf.word_wrap = True
    paras = text if isinstance(text, list) else [text]
    for i, t in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); r = para.add_run(); r.text = t; f = r.font; f.name = "Cambria"; f.size = Pt(size); f.bold = bold; f.color.rgb = RGBColor.from_string(color)
        if isinstance(text, list): para.space_after = Pt(12)
    return s
def blank(title, sub):
    sl = p.slides.add_slide(lay["Title and Content"])
    for ph in list(sl.placeholders): ph._element.getparent().remove(ph._element)
    tb(sl, f"{MK}-title", 0.17, 0.12, 12.9, 0.57, title, 28, True, "262626"); tb(sl, f"{MK}-subtitle", 0.33, 0.68, 12.69, 0.4, sub, 15, False, "595959"); return sl
new = []
d = p.slides.add_slide(lay["Title Slide"]); d.shapes.title.text = "11/18/26 Regulome Stage 1"; d.shapes.title.name = f"{MK}-divider"
for ph in d.placeholders:
    if ph.placeholder_format.idx == 1: ph.name = f"{MK}-subtitle"; ph.text = "Simple ABC linking, binary results; judged on recall against background"
new.append(d)
for f, title, sub, note in S:
    sl = blank(title, sub); im = Image.open(f"{PL}/{f}.png"); w, h = im.size
    pic = sl.shapes.add_picture(f"{PL}/{f}.png", Inches(0.2), Inches(1.2), width=Emu(int(w / 300 * 914400)), height=Emu(int(h / 300 * 914400))); pic.name = f"{MK}-{f}"
    if note: tb(sl, f"{MK}-note", 0.6, 1.2 + h / 300 + 0.05, 12.1, 0.45, note, 18, False, "262626")
    assert pic.top + pic.height <= p.slide_height and pic.left + pic.width <= p.slide_width, f
    print(f, w, h, "px ->", round(pic.width / 914400, 2), "x", round(pic.height / 914400, 2), "in", flush=True); new.append(sl)
sl = blank("Regulome- Stage 1 · summary", "63 TFs, top-25% sgRNA cells, HEK293T"); tb(sl, f"{MK}-bullets", 0.6, 1.5, 12.1, 5.6, ["• " + b for b in BUL], 20, False, "262626"); new.append(sl)
bad = [(sh.name, sh.text_frame.text[:80]) for s in new for sh in s.shapes if sh.has_text_frame and re.search(r"expected|fold|chance|precision|validated|functional support|HEK293(?!T)", sh.text_frame.text, re.I)]
print("flagged text:", bad); assert not bad
lst = p.slides._sldIdLst; ids = list(lst); k = len(new)
for el in ids[-k:][::-1]: lst.remove(el); lst.insert(0, el)
p.part.rename_slide_parts([el.rId for el in lst])
for i, s in enumerate(p.slides, 1):
    if s.has_notes_slide: s.notes_slide.part.partname = s.notes_slide.part.partname.__class__(f"/ppt/notesSlides/notesSlide{i}.xml")
names = [s.part.partname for s in p.slides]; assert len(set(names)) == len(names)
nn = [s.notes_slide.part.partname for s in p.slides if s.has_notes_slide]; assert len(set(nn)) == len(nn)
assert len(p.slides) == n0 + k; p.save(OUT)
q = Presentation(OUT); print("reopened: slides", len(q.slides), "(before", n0, "+", k, ")")
for i in range(k + 2):
    s = q.slides[i]; print(i + 1, [sh.text_frame.text[:70] for sh in s.shapes if sh.has_text_frame][:2])
z = zipfile.ZipFile(OUT); c = collections.Counter(z.namelist()); dup = [x for x, v in c.items() if v > 1]; print("zip entries", len(c), "duplicates", dup); assert not dup
