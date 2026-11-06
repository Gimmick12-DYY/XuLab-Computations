"""Insert 4E′, 4E″, 4E‴ (hybrid TF + Hi-C loop model) right after slide 4E, copying 4E's title/subtitle style and layout."""
import copy
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches
from PIL import Image

DECK = Path("/Users/dyy/Desktop/HEK293T-ATGTAG/293T Yuyang local.pptx")
H = Path("/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model/tf_polymer_hic")
FH, FC = H / "figure_hub_flow", H / "figure_compare"
p = Presentation(DECK)
def title(s): return next((sh.text_frame.text for sh in s.shapes if sh.name == "Title" and sh.has_text_frame), "")
titles = [title(s) for s in p.slides]
i4e = next(i for i, t in enumerate(titles) if t.startswith("4E ·"))
isa = next(i for i, t in enumerate(titles) if t.startswith("S4A"))
assert not any(t.startswith("4E′") for t in titles)
E = p.slides[i4e]; tpl = {sh.name: sh for sh in E.shapes if sh.name in ("Title", "Subtitle")}
tpl["Bullets"] = next(sh for sh in p.slides[isa].shapes if sh.name == "Bullets")

SL = [
    dict(title="4E′ · Adding Hi-C loops keeps the TF-driven hub rosette",
         sub="TF bridging (as 4E) + all 165 Peakachu 0.7 loops as soft springs, each on with p = 0.2–0.8 by Hi-C O/E; selected conformation",
         imgs=[(FH / "hub_flow_selected_ghost_preview.png", 0.4, 1.15, None, 6.1), (FH / "hub_flow_selected_schematic_preview.png", 6.7, 1.1, 6.1, None)]),
    dict(title="4E″ · A typical conformation: a smaller hub flower",
         sub="Same model; conformation with the median hub content (6 of 23 elements, 2 of 24 edges) and 6 petals; 1 petal closed by a Hi-C loop",
         imgs=[(FH / "hub_flow_typical_ghost_preview.png", 0.4, 1.15, None, 6.1), (FH / "hub_flow_typical_schematic_preview.png", 6.7, 1.1, 6.1, None)]),
    dict(title="4E‴ · Hi-C loops improve the fit to Hi-C, not Cicero specificity",
         sub="TF only (grey, 4E) vs TF + Hi-C loops (orange); 10,000 conformations each; loops are inputs in the hybrid, so they are not a test",
         imgs=[(FC / "v2_vs_hybrid_preview.png", 0.45, 1.2, 12.4, None)],
         bullets=[("Hi-C: ", "with the restrained loop pixels removed, O/E Spearman rises from 0.11 to 0.23 (raw 0.35 → 0.46); P(s) gains long-range contacts."),
                  ("Cicero: ", "partners are 3–5× closer than distance-matched pairs, but ~1.0–1.1× vs TF-bound pairs matched for TF count (as without loops)."),
                  ("Hub flower: ", "≥3 hub cores share a flower in 62% of conformations vs 64% for matched sites; hub-core pairs co-cluster 1.21× (p = 0.01); 5% of petals sit on a restrained loop.")]),
]

def put_text(shape, text):
    tf = shape.text_frame; runs = tf.paragraphs[0].runs; runs[0].text = text
    for r in runs[1:]: r._r.getparent().remove(r._r)
    for q in tf.paragraphs[1:]: q._p.getparent().remove(q._p)
def put_bullets(shape, items):
    tf = shape.text_frame; base = copy.deepcopy(tf.paragraphs[0]._p)
    for q in list(tf.paragraphs): q._p.getparent().remove(q._p)
    for _ in items: tf._txBody.append(copy.deepcopy(base))
    for (lab, txt), para in zip(items, tf.paragraphs):
        rs = para.runs; rs[0].text = lab; rs[1].text = txt
        for r in rs[2:]: r._r.getparent().remove(r._r)

new = []
for spec in SL:
    s = p.slides.add_slide(E.slide_layout)
    for ph in list(s.placeholders): ph._element.getparent().remove(ph._element)
    for nm in ("Title", "Subtitle"): s.shapes._spTree.append(copy.deepcopy(tpl[nm]._element))
    sh = {x.name: x for x in s.shapes}; put_text(sh["Title"], spec["title"]); put_text(sh["Subtitle"], spec["sub"])
    for f, x, y, w, h in spec["imgs"]:
        iw, ih = Image.open(f).size
        if h is None: h = w * ih / iw
        if w is None: w = h * iw / ih
        s.shapes.add_picture(str(f), Inches(x), Inches(y), Inches(w), Inches(h))
    if "bullets" in spec:
        el = copy.deepcopy(tpl["Bullets"]._element); s.shapes._spTree.append(el)
        b = [x for x in s.shapes if x.name == "Bullets"][0]
        b.left, b.top, b.width, b.height = Inches(0.5), Inches(5.25), Inches(12.3), Inches(2.0)
        put_bullets(b, spec["bullets"])
    new.append(p.slides._sldIdLst[-1])
lst = p.slides._sldIdLst
for k, el in enumerate(new): lst.remove(el); lst.insert(i4e + 1 + k, el)
p.save(DECK)
print(f"4E is slide {i4e + 1}; inserted slides {i4e + 2}-{i4e + 1 + len(new)}")
