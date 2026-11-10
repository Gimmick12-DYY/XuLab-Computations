"""Add the '11/2/26 Jie requests' section at the top of the deck. usage: python build_jie_slides.py in.pptx out.pptx"""
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
IN, OUT = sys.argv[1], sys.argv[2]; I = "img"; INK = "1A1A1A"; GR = "8C8C8C"
p = Presentation(IN); LAY = {l.name: l for l in p.slide_layouts}; n0 = len(p.slides)
hic = next(i for i, s in enumerate(p.slides, 1) if any(sh.has_text_frame and sh.text_frame.text.startswith("4E-H12") for sh in s.shapes)); NEW = 4
def run(par, t, sz, b=False, c=INK):
    r = par.add_run(); r.text = t; r.font.name = "Cambria"; r.font.size = Pt(sz); r.font.bold = b; r.font.color.rgb = RGBColor.from_string(c); return r
def tb(s, x, y, w, h, t, sz=14, b=False, c=INK, name=None, al=None):
    o = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = o.text_frame; tf.word_wrap = True
    if name: o.name = name
    for k, line in enumerate(t if isinstance(t, list) else [t]):
        par = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        if al: par.alignment = al
        if isinstance(line, tuple): run(par, line[0], sz, True, c); run(par, " " + line[1], sz, False, c)
        else: run(par, line, sz, b, c)
    return o
def new(t, sub):
    s = p.slides.add_slide(LAY["Title and Content"])
    for ph in list(s.placeholders): ph._element.getparent().remove(ph._element)
    tb(s, 0.17, 0.12, 12.9, 0.57, t, 28, True, name="Title"); tb(s, 0.33, 0.68, 12.69, 0.4, sub, 15, False, GR, name="Subtitle"); return s
def pic(s, f, x, y, w=None, h=None): return s.shapes.add_picture(f"{I}/{f}", Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)
def table(s, rows, widths, x, y, rh, sz=14, hl=None):
    T = s.shapes.add_table(len(rows), len(widths), Inches(x), Inches(y), Inches(sum(widths)), Inches(rh * len(rows))); T.name = "Summary"; t = T.table
    for j, w in enumerate(widths): t.columns[j].width = Inches(w)
    t.rows[0].height = Inches(0.42)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = t.cell(i, j); c.vertical_anchor = MSO_ANCHOR.MIDDLE; c.fill.solid(); c.fill.fore_color.rgb = RGBColor.from_string("E3E3E3" if i == 0 else ("FCE9A8" if hl and i in hl else "FFFFFF")); tf = c.text_frame
            for k, line in enumerate(v.split("\n")):
                par = tf.paragraphs[0] if k == 0 else tf.add_paragraph(); run(par, line, sz, i == 0 or j == 0)
# section title
s0 = p.slides.add_slide(LAY["Title Slide"]); s0.shapes.title.text = "11/2/26 Jie requests"
for ph in list(s0.placeholders):
    if ph != s0.shapes.title and ph.placeholder_format.idx != 0: ph._element.getparent().remove(ph._element)
# (a) summary
s = new("Cobinding Complex Analysis- how many complexes", "79 TFs, 3,081 pairs; known and recent complexes checked against our data, novel pairs from structure")
table(s, [["Group", "Pairs / complexes", "What we recover"],
          ["Known classic\n10 complexes", "STRING ≥ 700 / literature:\nPRC2, NuRD, cohesin, MAZ–ZNF143, others", "3 by accessibility-corrected co-binding (percentile):\nPRC2 EZH2–MTF2 0.99 · MAZ–ZNF143 0.96 · cohesin CTCF–STAG2 0.93*"],
          ["Recently revealed\n6 pairs", "CAP-SELEX 2025: FOXK1–MLX, MLX–MSX2, HOXD10–SREBF2\nG9a/GLP: WIZ–ZNF644\nNAR 2024: ZNF512B–RBBP4 · Nature 2018: ADNP–HP1 (ChAHP)", "2 with co-binding support:\nFOXK1–MLX, WIZ–ZNF644"],
          ["Novel\n2 pairs", "Not in STRING or CAP-SELEX;\nAlphaFold 3 strong", "MLX–SREBF2 (next slide)\nHOXC4–RBBP4\nNeither passes our co-binding edge rule"]], [2.3, 5.2, 5.2], 0.33, 1.25, 1.0, 15, hl=[3])
tb(s, 0.33, 6.5, 12.7, 0.75, ["*Cohesin leans on CTCF, whose function our technique may affect. NuRD is not recovered here (0.50).", f"Item 3, Hi-C clique triangle: slide {hic + NEW}."], 12, False, GR, name="Footnote")
# (b) MLX-SREBF2
s = new("Cobinding Complex Analysis- novel example: MLX–SREBF2", "Structure agrees across tools; co-binding falls short of our edge rule")
HY = 1.15; tb(s, 0.33, HY, 4, 0.4, "Correlation", 17, True, name="Heading"); tb(s, 4.75, HY, 4.3, 0.4, "Structure", 17, True, name="Heading"); tb(s, 9.2, HY, 4, 0.4, "Motif", 17, True, name="Heading")
pic(s, "crop_MLX_SREBF2.png", 0.33, 1.5, h=2.95); pic(s, "dots_MLX_SREBF2.png", 0.33, 4.55, h=1.7)
tb(s, 0.33, 6.3, 4.1, 0.4, "0 of 4 metrics at ≥ 0.90; the edge needs 2", 13, True, name="Caption")
pic(s, "MLX_SREBF2_AF3_iface_trim.png", 4.65, 1.6, w=4.3)
tb(s, 4.75, 4.95, 4.3, 1.6, [("AlphaFold 3:", "ipTM 0.45, ipSAE 0.57; strong on 5 of 5 seeds"), ("Boltz-2:", "strong, trimmed and full length"), ("Interface:", "half of the contacts in the bHLH domains")], 13, name="Caption")
k_ = tb(s, 4.75, 4.6, 3.0, 0.35, "MLX  ", 12, True, "2166AC", name="Key"); run(k_.text_frame.paragraphs[0], "SREBF2", 12, True, "B2182B")
for f, y in (("logo_mlx.png", 1.55), ("logo_srebf2_ebox.png", 3.1), ("logo_srebf2_sre.png", 4.65)): pic(s, f, 9.25, y, h=1.45)
tb(s, 0.33, 6.72, 12.7, 0.6, "Both are bHLH-ZIP E-box binders, so the fold may dimerise readily rather than mark a specific partner. CAP-SELEX: tested, not called. Composite-motif test: not detectable at this depth.", 11, False, GR, name="Footnote")
# (c) HOXC4-RBBP4
s = new("Cobinding Complex Analysis- second example: HOXC4–RBBP4", "Lower confidence: AlphaFold 3 supports it, full-length Boltz-2 does not")
tb(s, 0.33, 1.15, 5, 0.4, "Structure (AlphaFold 3)", 17, True, name="Heading"); pic(s, "HOXC4_RBBP4_AF3_iface_trim.png", 0.45, 1.65, h=4.6)
t_ = tb(s, 0.45, 6.4, 5, 0.35, "HOXC4  ", 12, True, "2166AC", name="Key"); run(t_.text_frame.paragraphs[0], "RBBP4", 12, True, "B2182B")
tb(s, 6.9, 1.15, 6, 0.4, "Evidence", 17, True, name="Heading")
table(s, [["Test", "Result"], ["AlphaFold 3", "Strong on 3 of 5 seeds\nipTM 0.78, ipSAE 0.66"], ["Boltz-2", "Trimmed strong; full length none"], ["Partner", "RBBP4 is a hub partner in the screen"]], [1.7, 4.3], 6.9, 1.6, 0.55, 14)
tb(s, 6.9, 4.05, 6, 0.4, "Correlation", 17, True, name="Heading"); pic(s, "dots_HOXC4_RBBP4.png", 6.9, 4.45, w=4.1)
tb(s, 6.9, 6.5, 6, 0.4, "1 of 4 metrics at ≥ 0.90; the edge needs 2", 13, True, name="Caption")
# move to the front
L = p.slides._sldIdLst; ids = list(L)
for k, e in enumerate(ids[n0:]): L.remove(e); L.insert(k, e)
p.save(OUT); print("slides", n0, "->", len(p.slides), "Hi-C clique slide now", hic + NEW)
