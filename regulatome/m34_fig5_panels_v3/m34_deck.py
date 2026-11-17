# m34: add a "11/17/26 TF Analysis" section with the Figure 5 slides at the top of the local deck (newest-first order). usage: m34_deck.py <in.pptx> <out.pptx>
import sys, copy, zipfile, collections
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from PIL import Image
IN, OUT = sys.argv[1], sys.argv[2]; PL = "/work/users/d/y/dyy12/XuLab/regulatome/m34_fig5_panels_v3/plots/slide"; MK = "FIG5V3"
S = [("Fig5a_layers", "Regulome- Figure 5a · three layers", "Our HITAG binding linked to genes by ABC; Perturb-seq adds support and direction"),
     ("Fig5b_at_a_glance", "Regulome- Figure 5b · the map at a glance", "150,403 TF–gene edges; a gene is bound by a median of 12 TFs; 42% of edges have a promoter peak"),
     ("Fig5c_ours_vs_bulk", "Regulome- Figure 5c · our HITAG vs bulk ChIP, same path", "Overlap with Perturb-seq responders is low even for bulk ChIP; ours is on par for NRF1, NFYA and E4F1"),
     ("Fig5d_NRF1", "Regulome- Figure 5d · TF view: NRF1", "1,610 bound genes, 420 supported by Perturb-seq, 302 go down on knockdown; PGAM1 promoter as an example"),
     ("Fig5e_HSPB1", "Regulome- Figure 5e · gene view: HSPB1", "53 TFs bind HSPB1; knockdown of 9 of them changes it (7 down, 2 up)"),
     ("Fig5f_activator_repressor", "Regulome- Figure 5f · activator or repressor", "Direction of each TF's supported edges in HEK293T; darker = Tier 1 (both cell halves)"),
     ("Fig5g_pairs", "Regulome- Figure 5g · Figure 4 pairs on shared targets", "Targets bound by both members, and how often each member's knockdown changes them")]
SUPP = ["ML model, TF split (76 TFs): 2.3× responders in the top 200, from gene-level features, not from binding",
        "Pipeline audit: every step checked, no bug; bulk ChIP through the same path also reaches only 1.0–1.25×",
        "Perturb-seq limits: known partners do not look alike; STAG2 knockdown 18% with no responders",
        "NRF1 note: our peaks miss TFAM, CYCS and other classic targets that ENCODE NRF1 hits"]
p = Presentation(IN); n0 = len(p.slides); lay = {l.name: l for l in p.slide_layouts}
def tb(sl, name, x, y, w, h, text, size, bold, color):
    s = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); s.name = name; tf = s.text_frame; tf.word_wrap = True
    paras = text if isinstance(text, list) else [text]
    for i, t in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); r = para.add_run(); r.text = t; f = r.font; f.name = "Cambria"; f.size = Pt(size); f.bold = bold; f.color.rgb = RGBColor.from_string(color)
        if isinstance(text, list): para.space_after = Pt(14)
    return s
new = []
d = p.slides.add_slide(lay["Title Slide"]); d.shapes.title.text = "11/17/26 TF Analysis"; d.shapes.title.name = f"{MK}-divider"
for ph in d.placeholders:
    if ph.placeholder_format.idx == 1: ph.name = f"{MK}-subtitle"
new.append(d)
for f, title, sub in S:
    sl = p.slides.add_slide(lay["Title and Content"])
    for ph in list(sl.placeholders): ph._element.getparent().remove(ph._element)
    tb(sl, f"{MK}-title", 0.17, 0.12, 12.9, 0.57, title, 28, True, "262626"); tb(sl, f"{MK}-subtitle", 0.33, 0.68, 12.69, 0.4, sub, 15, False, "595959")
    im = Image.open(f"{PL}/{f}.png"); w, h = im.size; dpi = 300
    pic = sl.shapes.add_picture(f"{PL}/{f}.png", Inches(0.2), Inches(1.2), width=Emu(int(w / dpi * 914400)), height=Emu(int(h / dpi * 914400))); pic.name = f"{MK}-{f}"
    print(f, w, h, "px ->", round(pic.width / 914400, 2), "x", round(pic.height / 914400, 2), "in", flush=True); new.append(sl)
sl = p.slides.add_slide(lay["Title and Content"])
for ph in list(sl.placeholders): ph._element.getparent().remove(ph._element)
tb(sl, f"{MK}-title", 0.17, 0.12, 12.9, 0.57, "Regulome- Figure 5 supplement", 28, True, "262626"); tb(sl, f"{MK}-subtitle", 0.33, 0.68, 12.69, 0.4, "What sits behind the main panels", 15, False, "595959")
tb(sl, f"{MK}-bullets", 0.6, 1.6, 12.1, 4.8, ["• " + b for b in SUPP], 20, False, "262626"); new.append(sl)
# move the new slides to the top (newest section first)
lst = p.slides._sldIdLst; ids = list(lst); k = len(new)
for el in ids[-k:][::-1]: lst.remove(el); lst.insert(0, el)
# renumber slide and notes parts in presentation order
p.part.rename_slide_parts([el.rId for el in lst])
for i, s in enumerate(p.slides, 1):
    if s.has_notes_slide: s.notes_slide.part.partname = s.notes_slide.part.partname.__class__(f"/ppt/notesSlides/notesSlide{i}.xml")
names = [s.part.partname for s in p.slides]; assert len(set(names)) == len(names)
nn = [s.notes_slide.part.partname for s in p.slides if s.has_notes_slide]; assert len(set(nn)) == len(nn)
assert len(p.slides) == n0 + k, (len(p.slides), n0, k)
p.save(OUT)
q = Presentation(OUT); print("reopened: slides", len(q.slides), "(before", n0, "+", k, ")")
for i in range(k + 1):
    s = q.slides[i]; print(i + 1, [sh.name for sh in s.shapes if sh.name.startswith(MK)][:2], s.shapes.title.text if s.shapes.title is not None else [sh.text_frame.text for sh in s.shapes if sh.has_text_frame][:1])
z = zipfile.ZipFile(OUT); c = collections.Counter(z.namelist()); dup = [x for x, v in c.items() if v > 1]; print("zip entries", len(c), "duplicates", dup); assert not dup
