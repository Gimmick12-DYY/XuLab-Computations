"""MLX-SREBF2 structure slide on the full-length AF3 model with E-box DNA: swaps the three panels, updates the scores line, subtitle, conclusion and notes. usage: edit_dna_fl.py in.pptx out.pptx"""
import sys
from pptx import Presentation
from pptx.util import Inches
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
p = Presentation(sys.argv[1]); s = next(x for x in p.slides if any(sh.name == "Title" and "MLX–SREBF2" in sh.text_frame.text and "layer 1" in sh.text_frame.text for sh in x.shapes if sh.has_text_frame))
for sh in [x for x in s.shapes if x.shape_type == 13]: sh._element.getparent().remove(sh._element)
def pic(f, x, y, w, h):
    px = Image.open(f).size; assert abs(px[0] / w - 600) < 2 and abs(px[1] / h - 600) < 2, (f, px); s.shapes.add_picture(f, Inches(x), Inches(y), Inches(w), Inches(h))
pic("super/af3_MLX_dnafl_main.png", 0.1, 1.05, 6.3, 5.4); pic("super/af3_MLX_dnafl_close.png", 6.5, 1.05, 6.6, 2.8); pic("fig/sitemap_MLX_dnafl.png", 6.5, 3.9, 6.6, 1.95)
def setrun(name, frag, text):
    sh = next(x for x in s.shapes if x.name == name); r = sh.text_frame.paragraphs[0].runs; k = next(i for i, x in enumerate(r) if frag in x.text); r[k].text = text
    for x in r[k + 1:]: x.text = ""
setrun("Scores", "AF3 3.0", "AF3 3.0, full-length + E-box DNA, 5/5 seeds strong: ipTM 0.50, ipSAE 0.68; protein–DNA ipTM 0.71 / 0.81")
setrun("Subtitle", "AF3 places", "AF3 places full-length MLX–SREBF2 on E-box DNA as a bHLH-ZIP dimer, with both basic regions in the major groove")
setrun("Conclusion", "forms a bHLH-ZIP", "MLX–SREBF2 forms a bHLH-ZIP heterodimer that binds the E-box like known bHLH pairs; on bHLH-ZIP fragments the non-partner MLX–MAX scores as high, so the structure supports compatibility, not partner specificity.")
n = s.notes_slide.notes_text_frame; n.text = "\n".join(["Shown: AF3 3.0.1, full-length MLX (298 aa) and SREBF2 (1141 aa) with the 1NKP E-box oligo, seeds 1–5, best-ranked model; pLDDT below 50 and segments under 8 residues hidden. Medians: ipTM 0.50, ipSAE 0.68, protein–DNA ipTM 0.71 (MLX) and 0.81–0.82 (SREBF2), base contacts 9 and 9, major groove; fraction disordered 0.41. Fragment run: ipTM 0.87, ipSAE 0.74, protein–DNA 0.77 / 0.74, base contacts 10 and 10. Glu to the first C of CACGTG: MLX E138 3.0 Å, SREBF2 E339 2.7 Å (OE2 to N4). MLX Y183–SREBF2 Q381 2.8 Å.", "Interface in the shown model (residues within 5 Å): MLX 48 of 53 in bHLH-ZIP, SREBF2 37 of 65 in bHLH-ZIP; the other SREBF2 contacts lie around 402–470, 565–582, 651–668 and 862–866. Nuclear SREBF2 is the cleaved N-terminal fragment, so the TM and SCAP-binding regions would not be present on DNA in the nucleus; the contacts outside bHLH-ZIP are not interpreted.", n.text])
xml = s.part._element.xml; k = 0
for rid, rel in list(s.part.rels.items()):
    if rel.reltype.endswith("/image") and f'"{rid}"' not in xml: s.part.rels.pop(rid); k += 1
print("unused images dropped", k); p.save(sys.argv[2]); print("ok", len(list(p.slides)))
