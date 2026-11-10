"""MLX-SREBF2 structure slide rebuilt on the AF3 model with E-box DNA; slide 1 structural-support cell updated. usage: edit_dna_slide.py in.pptx out.pptx"""
import sys
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
p = Presentation(sys.argv[1]); slides = list(p.slides); ttl = lambda s: next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()), ""); s = next(x for x in slides if ttl(x).startswith("Novel MLX–SREBF2 – layer 1"))
def tb(s, x, y, w, h, paras, size=14, name="Conclusion"):
    t = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); t.name = name; f = t.text_frame; f.word_wrap = True
    for i, (b, v) in enumerate(paras):
        q = f.paragraphs[0] if i == 0 else f.add_paragraph(); q.space_after = Pt(4)
        for tx, bold in (((b, True), (v, False)) if b else ((v, False),)):
            r = q.add_run(); r.text = tx; r.font.size = Pt(size); r.font.name = "Cambria"; r.font.bold = bold; r.font.color.rgb = RGBColor.from_string("1A1A1A")
def pic(f, x, y, w, h):
    px = Image.open(f).size; assert abs(px[0] / w - 600) < 2 and abs(px[1] / h - 600) < 2, (f, px); s.shapes.add_picture(f, Inches(x), Inches(y), Inches(w), Inches(h))
old = s.notes_slide.notes_text_frame.text
for sh in [x for x in s.shapes if x.name not in ("Title", "Subtitle")]: sh._element.getparent().remove(sh._element)
r = next(x for x in s.shapes if x.name == "Subtitle").text_frame.paragraphs[0].runs; r[0].text = "AF3 places the MLX–SREBF2 bHLH-ZIP dimer on E-box DNA, with both basic regions in the major groove"
for x in r[1:]: x.text = ""
pic("super/af3_MLX_dna.png", 0.2, 1.05, 12.9, 3.45); pic("fig/pae_MLX_dna.png", 0.2, 4.55, 2.6, 1.95); pic("fig/sitemap_MLX_dna.png", 6.0, 4.55, 7.1, 1.95)
tb(s, 2.85, 4.7, 3.1, 1.75, [("AF3 3.0, bHLH-ZIP + E-box DNA, 5 seeds", ""), ("", "ipTM 0.87, ipSAE 0.74"), ("", "Protein–DNA ipTM 0.77 / 0.74")], size=13, name="Scores")
tb(s, 0.3, 6.5, 12.1, 0.6, [("Conclusion: ", "MLX–SREBF2 forms a bHLH-ZIP heterodimer that binds the E-box like known bHLH pairs; the non-partner MLX–MAX scores as high, so the structure supports compatibility, not partner specificity.")], size=13)
s.notes_slide.notes_text_frame.text = "\n".join(["AF3 3.0.1, MLX 129–220 and SREBF2 330–401 with the 1NKP E-box oligo, seeds 1–5, best-ranked model shown. Glu to the first C of CACGTG: MLX E138 3.0 Å, SREBF2 E339 2.8 Å (OE1 to N4). MLX Y183–SREBF2 Q381 2.8 Å.", "Same fragments, other DNA: SRE (1AM9 oligo) ipTM 0.87, ipSAE 0.74, protein–DNA 0.81–0.82 / 0.73–0.76; dinucleotide-shuffled scrambled E-box oligo ipTM 0.87, ipSAE 0.74, protein–DNA 0.77 / 0.73–0.74. AF3 does not separate these sequences.", "Controls: MLX–MLXIP + E-box ipTM 0.88, ipSAE 0.76; SREBF2 homodimer + SRE ipTM 0.87, ipSAE 0.76; MAX–MLX + E-box (reported non-partner) ipTM 0.90, ipSAE 0.81. Both basic regions contact bases from the major groove in all 30 models.", "Earlier model without DNA, full-length proteins: ipTM 0.45, ipSAE 0.57, 5 of 5 seeds strong; interface HLH-ZIP 38 of 38 contact residues on MLX, 31 of 40 on SREBF2; basic region 0 and 1. Boltz-2: strong, trimmed and full length.", old])
t = next(x for x in slides[0].shapes if x.has_table).table; c = list(next(r for r in t.rows if r.cells[0].text.strip() == "MLX–SREBF2").cells)[2]; rr = c.text_frame.paragraphs[0].runs; rr[0].text = "AF3 dimer on E-box DNA (5/5 seeds); not partner-specific"
for x in rr[1:]: x.text = ""
xml = s.part._element.xml; n = 0
for rid, rel in list(s.part.rels.items()):
    if rel.reltype.endswith("/image") and f'"{rid}"' not in xml: s.part.rels.pop(rid); n += 1
print("unused images dropped", n); p.save(sys.argv[2]); print("ok", len(slides))
