"""Fill the EZH2-MTF2 structure slide with the trimer model, add one superposition slide after each structure slide, update the table footnote's slide numbers. Nothing else changes. usage: add_super_and_prc2.py in.pptx out.pptx"""
import json, sys
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
IN, OUT = sys.argv[1], sys.argv[2]; p = Presentation(IN); slides = list(p.slides); FR = json.load(open("fig/frames.json")); ttl = lambda s: next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()), ""); LAY = {l.name: l for l in p.slide_layouts}; n0 = len(slides)
idx = lambda st: next(i for i, s in enumerate(slides) if ttl(s).startswith(st)); ip, im = idx("PRC2 core EZH2–MTF2- layer 1"), idx("Novel MLX–SREBF2- layer 1")
def run(par, t, sz, b=False, c="1A1A1A"):
    r = par.add_run(); r.text = t; r.font.name = "Cambria"; r.font.size = Pt(sz); r.font.bold = b; r.font.color.rgb = RGBColor.from_string(c)
def tb(s, x, y, w, h, t, sz=14, b=False, c="1A1A1A", name=None):
    o = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = o.text_frame; tf.word_wrap = True
    if name: o.name = name
    for k, line in enumerate(t if isinstance(t, list) else [t]):
        par = tf.paragraphs[0] if k == 0 else tf.add_paragraph(); par.space_after = Pt(3)
        if isinstance(line, tuple): run(par, line[0], sz, True, c); run(par, " " + line[1], sz, False, c)
        else: run(par, line, sz, b, c)
def pic(s, f, x, y):
    k = f.split("/")[-1][:-4]; w, h = FR.get(k, (12.9, 5.6)); px = Image.open(f).size; assert abs(px[0] / w - 600) < 2 and abs(px[1] / h - 600) < 2, (f, px, w, h); s.shapes.add_picture(f, Inches(x), Inches(y), Inches(w), Inches(h))
s = slides[ip]; rm = [sh for sh in s.shapes if sh.name not in ("Title", "Subtitle")]; print("slide", ip + 1, "removing", [sh.name for sh in rm])
for sh in rm: sh._element.getparent().remove(sh._element)
Sc = pd.read_csv("img/af3_prc2_scores.tsv", sep="\t"); T_ = Sc[Sc.job == "EZH2_SUZ12_MTF2"].set_index("pair"); D_ = Sc[Sc.job == "EZH2_MTF2"].iloc[0]; rw = lambda r: f"ipTM {r.iptm_top:.2f}, ipSAE {r.ipsae_max:.2f}, {r.seeds_strong}/5 seeds strong"
pic(s, "fig/struct_PRC2.png", 0.25, 1.1); pic(s, "fig/sitemap_PRC2.png", 7.0, 1.1); pic(s, "fig/pae_PRC2.png", 7.0, 3.95)
tb(s, 9.8, 3.95, 3.4, 2.4, [("EZH2–SUZ12:", rw(T_.loc["A-B"])), ("SUZ12–MTF2:", rw(T_.loc["B-C"])), ("EZH2–MTF2:", rw(T_.loc["A-C"])), ("EZH2–MTF2 alone:", rw(D_))], 12, name="Scores")
tb(s, 0.3, 6.4, 12.7, 0.9, "AF3 EZH2–SUZ12–MTF2 trimer. Purple = DNA-binding site (MTF2 EH; Li et al. 2017; InterPro IPR063986). Black ticks = interface residues within 5 Å of another chain. Domains from UniProt and InterPro.", 12, False, "4D4D4D", name="Footnote")
def new(title, sub, f):
    o = p.slides.add_slide(LAY["Title and Content"])
    for ph in list(o.placeholders): ph._element.getparent().remove(ph._element)
    tb(o, 0.17, 0.12, 12.9, 0.57, title, 28, True, name="Title"); tb(o, 0.33, 0.66, 12.69, 0.4, sub, 15, False, "8C8C8C", name="Subtitle"); pic(o, f, 0.2, 1.1); return o
a = new("PRC2 core EZH2–MTF2- layer 1: model vs experiment", "On the closest PCL structure (SUZ12–PHF19), AF3 does not put the MTF2 C-terminus in the PHF19 site", "super/super_PRC2.png"); b = new("Novel MLX–SREBF2- layer 1: model vs experiment", "The AF3 dimer overlays the MYC–MAX–DNA structure; no MLX or SREBF2 heterodimer structure exists", "super/super_MLX.png")
Ls = p.slides._sldIdLst; ids = list(Ls); ea, eb = ids[n0], ids[n0 + 1]; Ls.remove(ea); Ls.remove(eb); Ls.insert(ip + 1, ea); Ls.insert(im + 2, eb)
new_list = list(p.slides); m = next(i for i, s_ in enumerate(new_list, 1) if ttl(s_).startswith("Cobinding Analysis")); hic = next(i for i, s_ in enumerate(new_list, 1) if ttl(s_).startswith("4E-H12")); hit = 0
for sh in new_list[0].shapes:
    if sh.has_text_frame:
        for par in sh.text_frame.paragraphs:
            for r in par.runs:
                if "(slide 182)" in r.text: r.text = r.text.replace("(slide 182)", f"(slide {m})"); hit += 1
                if "triangle: slide 26" in r.text: r.text = r.text.replace("triangle: slide 26", f"triangle: slide {hic}"); hit += 1
assert hit == 2, hit
p.save(OUT); print("slides", n0, "->", len(new_list), "matrix", m, "Hi-C", hic, [ttl(s_)[:42] for s_ in new_list[:10]])
