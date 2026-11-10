"""Swap the picture on the two model-vs-experiment slides and put the removed side text in the speaker notes. usage: edit_super.py in.pptx out.pptx"""
import json, sys
from pptx import Presentation
from pptx.util import Inches
J = json.load(open("super/super.json")); p = Presentation(sys.argv[1]); m, q = J["MLX"], J["PRC2"]; seen = 0
NOTES = {"Novel MLX–SREBF2 – layer 1: model vs experiment": ("super/super_MLX.png", f"Grey = experiment, PDB 1NKP (MYC–MAX on DNA, 1.8 Å). Red = AF3 MLX–SREBF2, bHLH-ZIP region shown. Fit: Cα RMSD {m['rmsd']} Å over {m['n']} residues, MLX mapped to MAX, SREBF2 to MYC. Dimer contact (MAX Y70–MYC Q422 (1NKP author numbering Y249, Q954) vs MLX Y183–SREBF2 Q381): {m['c1']['ref']} Å vs {m['c1']['model']} Å. Basic region to E-box (MYC E378–adenine N6, author numbering E910 vs SREBF2 E339): {m['c2']['ref']} Å vs {m['c2']['model']} Å; the model has no DNA, so the basic helices sit away from the bases."),
         "PRC2 core EZH2–MTF2- layer 1: model vs experiment": ("super/super_PRC2.png", f"Grey = experiment, PDB 6NQ3 (SUZ12–RBBP4–PHF19–JARID2, 2.89 Å; SUZ12 and PHF19 shown). Blue = AF3 SUZ12 and MTF2 from the EZH2–SUZ12–MTF2 trimer, EZH2 hidden. Fit: Cα RMSD {q['rmsd']} Å over {q['n']} of {q['site_n']} SUZ12 site residues. Contact (SUZ12 Q95–PHF19 Y542 vs SUZ12 Q95–MTF2 Y555): {q['c1']['ref']} Å vs {q['c1']['model']} Å. No MTF2 structure exists; PHF19 is its paralog. AF3 places the MTF2 C-terminus on a different SUZ12 surface ({len(q['shared_suz12'])} of {q['ref_suz12_iface_n']} site residues shared).")}
for s in p.slides:
    t = next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()), "")
    if t == "Novel MLX–SREBF2- layer 1: model vs experiment":
        next(sh for sh in s.shapes if sh.has_text_frame and sh.text_frame.text == t).text_frame.paragraphs[0].runs[0].text = "Novel MLX–SREBF2 – layer 1: model vs experiment"; t = "Novel MLX–SREBF2 – layer 1: model vs experiment"
    if t not in NOTES: continue
    f, n = NOTES[t]; pics = [sh for sh in s.shapes if sh.shape_type == 13]; assert len(pics) == 1; o = pics[0]; L, T_ = o.left, o.top; o._element.getparent().remove(o._element); s.shapes.add_picture(f, L, T_, Inches(12.9), Inches(5.6)); s.notes_slide.notes_text_frame.text = n; seen += 1
assert seen == 2; p.save(sys.argv[2]); print("ok", len(p.slides))
