"""Insert 4E-H1..H3 (Hi-C-first models) right after slide 4E‴, copying 4E's title/subtitle style and layout."""
import copy
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches
from PIL import Image

DECK = Path("/Users/dyy/Desktop/HEK293T-ATGTAG/293T Yuyang local.pptx")
H = Path("/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model/tf_polymer_hicfirst")
F1, F2, FC = H / "hicfirst/figure_hub_flow", H / "hicfirst_tf/figure_hub_flow", H / "compare"
p = Presentation(DECK)
def title(s): return next((sh.text_frame.text for sh in s.shapes if sh.name == "Title" and sh.has_text_frame), "")
titles = [title(s) for s in p.slides]
i4e = next(i for i, t in enumerate(titles) if t.startswith("4E ·")); i4l = next(i for i, t in enumerate(titles) if t.startswith("4E‴"))
isa = next(i for i, t in enumerate(titles) if t.startswith("S4A"))
assert not any(t.startswith("4E-H") for t in titles)
E = p.slides[i4e]; tpl = {sh.name: sh for sh in E.shapes if sh.name in ("Title", "Subtitle")}
tpl["Bullets"] = next(sh for sh in p.slides[isa].shapes if sh.name == "Bullets")

SL = [
    dict(title="4E-H1 · Hi-C first: restraining the Hi-C map assembles the hub",
         sub="677 Hi-C pixels (O/E ≥ 2, p = 0.1–0.6) + 165 loops (p = 0.2–0.8) + 18 red edges (always on); no TF attraction; typical conformation",
         imgs=[(F1 / "hub_flow_typical_ghost_preview.png", 0.4, 1.15, None, 6.1), (F1 / "hub_flow_typical_schematic_preview.png", 6.7, 1.1, 6.1, None)]),
    dict(title="4E-H2 · Hi-C first + TF bridging: TF cores form on the Hi-C hub",
         sub="Same Hi-C restraints + the 4E TF bridging (16 kT); selected: 13/23 elements, 13/24 edges in one flower (top 0.9%; typical: 6 and 4)",
         imgs=[(F2 / "hub_flow_selected_ghost_preview.png", 0.4, 1.15, None, 6.1), (F2 / "hub_flow_selected_schematic_preview.png", 6.7, 1.1, 6.1, None)]),
    dict(title="4E-H3 · Hi-C first fits Hi-C best; hub contacts come from restraints",
         sub="Five models, common held-out sets; Hi-C pixels, loops and red edges are inputs in the Hi-C-first models, so only held-out pixels/edges test them",
         imgs=[(FC / "model_table_preview.png", 0.5, 1.2, 12.3, None)],
         bullets=[("Hi-C: ", "Hi-C-first fits unrestrained Hi-C best (O/E 0.37 vs 0.29 loops + TF, 0.13 TF only); on the 176 held-out pixels loops + TF is higher (0.28 vs 0.23)."),
                  ("Hub: ", "it assembles in Hi-C-first models because the 18 red edges are restrained; unrestrained blue edges are not closer (1.0×; 1.7× with TF), and no model puts Cicero partners closer than TF-bound matched pairs."),
                  ("Best: ", "Hi-C first + TF (same Hi-C fit, more contact at held-out edges; ≥3 hub cores in one TF flower 53% vs 44% null). 23 of its 200 runs were lost to numerical blow-ups.")]),
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
        b.left, b.top, b.width, b.height = Inches(0.5), Inches(4.45), Inches(12.3), Inches(2.9)
        put_bullets(b, spec["bullets"])
    new.append(p.slides._sldIdLst[-1])
lst = p.slides._sldIdLst
for k, el in enumerate(new): lst.remove(el); lst.insert(i4l + 1 + k, el)
p.save(DECK)
print(f"4E‴ is slide {i4l + 1}; inserted slides {i4l + 2}-{i4l + 1 + len(new)}")
