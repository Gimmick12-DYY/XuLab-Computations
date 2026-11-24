# m63: Perturb-seq benchmark slide (what the X-Atlas/Orion data looks like), native shapes, inserted as slide 2.
import sys, copy, re, zipfile, collections
import numpy as np
from pptx import Presentation
from pptx.util import Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
IN, OUT = sys.argv[1], sys.argv[2]
def E(x): return Emu(int(round(x * 914400)))
def C(h): return RGBColor.from_string(h)
ORANGE, LGREY, MGREY, DGREY, TXT, SUB = "D98C1F", "D8D8D8", "9A9A9A", "555555", "262626", "6E6E6E"
RED, BLUE, PALE = "C8553D", "3B6FB6", "EDEDED"
rng = np.random.default_rng(20261124)

def box(shapes, kind, x, y, w, h, fill, name=None):
    s = shapes.add_shape(kind, E(x), E(y), E(w), E(h)); st = s._element.find(qn("p:style"))
    if st is not None: s._element.remove(st)
    s.fill.solid(); s.fill.fore_color.rgb = C(fill); s.line.fill.background()
    if name: s.name = name
    return s
def text(shapes, x, y, w, h, lines, size=12, color=TXT, bold=False, italic=False, font="Arial", align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, wrap=True, name=None, after=0):
    t = shapes.add_textbox(E(x), E(y), E(w), E(h)); tf = t.text_frame; tf.word_wrap = wrap; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for k, line in enumerate(lines):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph(); p.alignment = align; p.space_after = Pt(after)
        r = p.add_run(); r.text = line; f = r.font; f.name = font; f.size = Pt(size); f.bold = bold; f.italic = italic; f.color.rgb = C(color)
    if name: t.name = name
    return t
def label(shapes, x, y, w, h, s, fill, fg):
    b = box(shapes, MSO_SHAPE.RECTANGLE, x, y, w, h, fill); tf = b.text_frame; tf.word_wrap = False; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; r = p.add_run(); r.text = s; r.font.name = "Arial"; r.font.size = Pt(12); r.font.color.rgb = C(fg)
def dots(shapes, x, y, vertical, d=0.045, gap=0.075, col=DGREY):
    for k in range(3): box(shapes, MSO_SHAPE.OVAL, x + (0 if vertical else k * gap), y + (k * gap if vertical else 0), d, d, col)
def header(shapes, x, w, s): text(shapes, x, 1.62, w, 0.4, [s], size=16, bold=True, font="Cambria")

p = Presentation(IN); n0 = len(p.slides)
assert p.slide_width == 12192000 and p.slide_height == 6858000
s1 = [sh.text_frame.text for sh in p.slides[0].shapes if sh.has_text_frame][0]; assert s1.startswith("11/20/26 Regulome Stage 1"), s1
assert not any(sh.name.startswith("PSBENCH") for s in p.slides for sh in s.shapes), "slide already present"
ref = [s for s in p.slides if any(sh.name == "FIG5S1C-title" for sh in s.shapes)]; assert len(ref) == 1; ref = ref[0]
subsrc = [s for s in p.slides if any(sh.name == "FIG5S1B-title" for sh in s.shapes)][0]
new = p.slides.add_slide(ref.slide_layout)
for sh in list(new.shapes): sh._element.getparent().remove(sh._element)
for src, nm, txt, size, col in [(ref, "FIG5S1C-title", "Perturb-seq benchmark: what the data looks like", None, None),
                                (subsrc, "FIG5S1B-subtitle", "X-Atlas/Orion, genome-wide CRISPRi Perturb-seq in HEK293T (Huang et al., bioRxiv 2025)", 14, SUB)]:
    el = copy.deepcopy([sh for sh in src.shapes if sh.name == nm and sh.shape_type == 17][0]._element); new.shapes._spTree.append(el)
    sh = [x for x in new.shapes if x.name == nm][0]; sh.name = "PSBENCH-" + nm.split("-")[1]; para = sh.text_frame.paragraphs[0]
    for extra in list(sh.text_frame.paragraphs)[1:]: extra._p.getparent().remove(extra._p)
    for r in para.runs[1:]: r._r.getparent().remove(r._r)
    para.runs[0].text = txt
    if size: para.runs[0].font.size = Pt(size); para.runs[0].font.color.rgb = C(col)

# panel 1
g = new.shapes.add_group_shape(); g.name = "PSBENCH-panel1-raw"; S = g.shapes
header(S, 0.5, 4.4, "1  Raw data: one row per cell")
for s_, x, w, al in [("guide target", 0.5, 1.32, PP_ALIGN.CENTER), ("lane", 1.85, 0.4, PP_ALIGN.CENTER), ("guide UMIs", 2.3, 0.86, PP_ALIGN.CENTER), ("counts, 38,606 genes", 3.28, 1.65, PP_ALIGN.LEFT)]:
    text(S, x, 2.3, w, 0.28, [s_], color=SUB, align=al, wrap=False)
ROWS = [("NRF1", "L1", ORANGE, "FFFFFF"), ("NRF1", "L2", ORANGE, "FFFFFF"), ("NRF1", "L3", ORANGE, "FFFFFF"), ("CTCF", "L1", LGREY, TXT), ("MAZ", "L2", LGREY, TXT), ("non-targeting", "L1", MGREY, "FFFFFF"), ("non-targeting", "L2", MGREY, "FFFFFF")]
BARS = [0.9, .35, .6, .5, .75, .55, .4]
for i, (lab, lane, fill, fg) in enumerate(ROWS):
    t0 = 2.65 + i * 0.36
    label(S, 0.5, t0 + 0.03, 1.32, 0.30, lab, fill, fg)
    text(S, 1.85, t0 + 0.03, 0.4, 0.30, [lane], color="404040", align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, wrap=False)
    box(S, MSO_SHAPE.RECTANGLE, 2.45, t0 + 0.13, 0.56 * BARS[i], 0.10, DGREY)
    for j in range(6):
        v = int(rng.integers(0x60, 0xEC)); box(S, MSO_SHAPE.RECTANGLE, 3.28 + j * 0.27, t0 + 0.08, 0.2, 0.2, f"{v:02X}{v:02X}{v:02X}")
dots(S, 1.14, 5.22, True); dots(S, 3.28 + (5 * 0.27 + 0.2) / 2 - 0.022, 5.22, True); dots(S, 4.89, 2.65 + 3 * 0.36 + 0.16, False, gap=0.06)
text(S, 0.5, 5.52, 4.4, 0.75, ["4.5 million cells in 223 lanes (10x runs)", "each cell: guides against 1 of 18,903 genes,", "or a non-targeting guide (218,838 control cells)"], size=12.5)

# panel 2
g = new.shapes.add_group_shape(); g.name = "PSBENCH-panel2-knockdown"; S = g.shapes
header(S, 5.55, 3.95, "2  For each TF: knockdown vs control")
text(S, 5.55, 2.3, 3.65, 0.28, ["NRF1 knockdown cells"], size=13, bold=True, color=ORANGE)
for r_ in range(2):
    for c_ in range(8): box(S, MSO_SHAPE.OVAL, 5.6 + c_ * 0.2, 2.62 + r_ * 0.2, 0.15, 0.15, ORANGE)
text(S, 5.55, 3.05, 3.7, 0.45, ["343 cells carry the NRF1 guides;", "we keep the 86 with the most guide (top 25 %)"])
text(S, 5.55, 3.55, 1.0, 0.25, ["vs"], italic=True, color="404040")
text(S, 5.55, 3.8, 3.65, 0.28, ["non-targeting cells, same lanes"], size=13, bold=True, color=SUB)
for r_ in range(3):
    for c_ in range(14): box(S, MSO_SHAPE.OVAL, 5.6 + c_ * 0.2, 4.12 + r_ * 0.2, 0.15, 0.15, MGREY)
text(S, 5.55, 4.75, 3.65, 0.25, ["74,165 cells (about 1,000 per lane)"])
text(S, 5.55, 5.12, 3.7, 0.45, ["all 63 TFs: median 59 cells per TF (25 to 216)", "output: one number per gene = its change"])

# panel 3
g = new.shapes.add_group_shape(); g.name = "PSBENCH-panel3-table"; S = g.shapes
header(S, 9.8, 3.45, "3  Result: TF by gene table")
GX, GY = 10.55, 2.65
text(S, GX, 2.3, 7 * 0.30 + 0.26, 0.28, ["11,228 expressed genes"], color=SUB, align=PP_ALIGN.CENTER, wrap=False)
for i, tf_ in enumerate(["NRF1", "CTCF", "MAZ", "E4F1", "SOX4"]):
    text(S, 9.98, GY + i * 0.36, 0.52, 0.30, [tf_], align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE, wrap=False)
    for j in range(8):
        col = rng.choice([PALE, RED, BLUE], p=[0.66, 0.17, 0.17]); box(S, MSO_SHAPE.RECTANGLE, GX + j * 0.30, GY + i * 0.36, 0.26, 0.30, str(col))
rl = text(S, 9.85 - 0.8, GY + (4 * 0.36 + 0.30) / 2 - 0.14, 1.6, 0.28, ["63 TFs"], color=SUB, align=PP_ALIGN.CENTER, wrap=False); rl.rotation = 270
dots(S, GX + (7 * 0.30 + 0.26) / 2 - 0.022, GY + 5 * 0.36 + 0.02, True); dots(S, GX + 7 * 0.30 + 0.32, GY + 2 * 0.36 + 0.13, False, gap=0.06)
for k, (col, s_) in enumerate([(RED, "up after knockdown"), (BLUE, "down after knockdown")]):
    box(S, MSO_SHAPE.RECTANGLE, 10.0, 4.82 + k * 0.27, 0.18, 0.18, col); text(S, 10.27, 4.79 + k * 0.27, 2.6, 0.25, [s_], wrap=False)
text(S, 9.8, 5.45, 3.4, 0.7, ["a cutoff on the change gives the", "Perturb-seq target genes of each TF", "(|z| ≥ 2: median 873 genes per TF)"])

# arrows and footnote
for k, (x0, x1) in enumerate([(5.10, 5.42), (9.25, 9.65)]):
    box(new.shapes, MSO_SHAPE.RIGHT_ARROW, x0, 7.5 - 3.55 - 0.11, x1 - x0, 0.22, MGREY, name=f"PSBENCH-arrow{k + 1}")
text(new.shapes, 0.5, 6.88, 12.3, 0.3, ["Squares, bars and dots are a sketch, not data values. Knockdown of the TF's own gene in the cells we use: median 63 %."], color=SUB, font="Cambria", name="PSBENCH-footnote")
new.notes_slide.notes_text_frame.text = "Sketch of the X-Atlas/Orion HEK293T Perturb-seq data. Numbers: guide calls and lanes (HEK293T_filtered_guide_calls_per_cell.csv.gz, panel_ntc_cells), m23 cell selection (top 25 % by guide UMIs, NTC of the same lanes), m57 report (63 x 11,228; |z| >= 2 median 873). Builder: XuLab regulatome/m63_perturbseq_schematic/m63_deck.py."

# all text >= 12 pt
for sh in new.shapes:
    for x in (sh.shapes if sh.shape_type == 6 else [sh]):
        if x.has_text_frame:
            for pp in x.text_frame.paragraphs:
                for r in pp.runs: assert r.font.size is None or r.font.size >= Pt(12), (x.name, r.text)
lst = p.slides._sldIdLst; el = list(lst)[-1]; lst.remove(el); lst.insert(1, el)
p.part.rename_slide_parts([e.rId for e in lst])
for i, s in enumerate(p.slides, 1):
    if s.has_notes_slide: s.notes_slide.part.partname = s.notes_slide.part.partname.__class__(f"/ppt/notesSlides/notesSlide{i}.xml")
names = [s.part.partname for s in p.slides]; assert len(set(names)) == len(names)
nn = [s.notes_slide.part.partname for s in p.slides if s.has_notes_slide]; assert len(set(nn)) == len(nn)
alltxt = " ".join(x.text_frame.text for sh in new.shapes for x in (sh.shapes if sh.shape_type == 6 else [sh]) if x.has_text_frame) + new.notes_slide.notes_text_frame.text
assert not re.search(r"claude|anthropic|generated with", alltxt, re.I)
p.save(OUT); z = zipfile.ZipFile(OUT); dup = [k for k, v in collections.Counter(z.namelist()).items() if v > 1]; assert not dup, dup
q = Presentation(OUT); assert len(q.slides) == n0 + 1
for i in range(4): print(i + 1, q.slides[i].part.partname, [sh.text_frame.text[:70] for sh in q.slides[i].shapes if sh.has_text_frame][:1])
c29 = [i + 1 for i, s in enumerate(q.slides) if any(sh.name == "FIG5S1C-table-cutoff" for sh in s.shapes)]
print("slides", n0, "->", len(q.slides), "| cutoff slide now", c29, "| groups", [sh.name for sh in q.slides[1].shapes]); print("done")
