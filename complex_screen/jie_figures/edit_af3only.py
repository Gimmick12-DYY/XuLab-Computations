"""Rebuild the MLX-SREBF2 model-vs-experiment slide as the AF3 model only, with the 1NKP comparison moved to the speaker notes. Slide 6 and slide 3 stay as they are. usage: edit_af3only.py in.pptx out.pptx"""
import json, sys
from pptx import Presentation
from pptx.util import Inches
J = json.load(open("super/super.json"))["MLX"]; A = json.load(open("super/af3only.json")); p = Presentation(sys.argv[1]); seen = 0; NEW = "Novel MLX–SREBF2 – layer 1: AF3 interface"
NOTE = f"AF3 MLX–SREBF2, bHLH-ZIP region shown (ipTM 0.45, ipSAE 0.57, 5 of 5 seeds strong). Contact MLX Y183–SREBF2 Q381, {A['d']} Å. Basic regions are free; the model was run without DNA. Comparison with experiment, not shown: PDB 1NKP (MYC–MAX on DNA, 1.8 Å), Cα RMSD {J['rmsd']} Å over {J['n']} residues, MLX mapped to MAX and SREBF2 to MYC. The equivalent contact there, MAX Y70–MYC Q422 (UniProt; author numbering Y249, Q954), is {J['c1']['ref']} Å. MYC E378 (author E910) to adenine N6 is {J['c2']['ref']} Å; SREBF2 E339 to the same base after superposition is {J['c2']['model']} Å. No MLX or SREBF2 heterodimer structure exists."
for s in p.slides:
    tt = [sh for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()]
    if not tt or not (tt[0].text_frame.text.startswith("Novel MLX–SREBF2") and "model vs experiment" in tt[0].text_frame.text): continue
    tt[0].text_frame.paragraphs[0].runs[0].text = NEW
    for r in tt[0].text_frame.paragraphs[0].runs[1:]: r.text = ""
    for sh in s.shapes:
        if sh.name == "Subtitle":
            sh.text_frame.paragraphs[0].runs[0].text = "The two helices pack as a bHLH-ZIP dimer, with the basic DNA-binding regions left free"
    pics = [sh for sh in s.shapes if sh.shape_type == 13]; assert len(pics) == 1; o = pics[0]; L, T_ = o.left, o.top; o._element.getparent().remove(o._element); s.shapes.add_picture("super/af3_MLX.png", L, T_, Inches(12.9), Inches(5.6)); s.notes_slide.notes_text_frame.text = NOTE; seen += 1
assert seen == 1, seen
# replaced pictures leave their old image attached to the slide; drop those so the deck does not keep growing
n_drop = 0
for s in list(p.slides)[:9]:
    x = s.part._element.xml
    for rid, rel in list(s.part.rels.items()):
        if rel.reltype.endswith("/image") and f'"{rid}"' not in x: s.part.rels.pop(rid); n_drop += 1
print("unused images dropped", n_drop)
p.save(sys.argv[2]); print("ok", len(p.slides))
