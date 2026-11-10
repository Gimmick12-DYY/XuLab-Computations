"""Correlation slides: six element sets in the tables, slide 1 percentiles from the all-elements set, plain wording, new conclusions. Matrix images redrawn from the all-elements set. usage: edit_corr_round.py in.pptx out.pptx"""
import copy, glob, math, sys
import pandas as pd
from pptx import Presentation
from pptx.util import Inches
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
p = Presentation(sys.argv[1]); slides = list(p.slides); ttl = lambda s: next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()), ""); find = lambda st: next(s for s in slides if ttl(s).startswith(st)); fl = lambda v: f"{math.floor(v * 100) / 100:.2f}"
D = next(d for d in ("data/corr_sets", "data/corr_sets_all") if glob.glob(f"{d}/pct_tss.tsv"))
def rd(n):
    m = pd.read_csv(f"{D}/pct_{n}.tsv", sep="\t", index_col=0); m.index = m.index.str.strip().str.upper(); m.columns = m.columns.str.strip().str.upper(); return m
A, Nq, Ts = rd("all"), rd("noquies"), rd("tss")
def settext(c, v, bold=None):
    rr = c.text_frame.paragraphs[0].runs; rr[0].text = v
    for x in rr[1:]: x.text = ""
    if bold is not None: rr[0].font.bold = bold
def sett(sh, paras):
    ps = sh.text_frame.paragraphs; assert len(ps) >= len(paras)
    for q, v in zip(ps, paras):
        rr = q.runs; rr[0].text = v
        for x in rr[1:]: x.text = ""
    for q in ps[len(paras):]: q._p.getparent().remove(q._p)
FOOT = ["Each element's signal over its 79-TF mean, then Pearson, SpQN, percentile; elements with ≥ 20 reads. Matrix at left: all elements.", "Local cluster = smallest branch of that set's TF tree (average linkage, 1 − Pearson r) containing both TFs; the pair is ranked among TF pairs in that branch. In brackets: TFs in the branch. — = not computed.", "TSS = TssA, TssFlnk, TssFlnkU, TssFlnkD, TssBiv. Repressive = ReprPC, ReprPCWk, Het, ZNF/Rpts. Polycomb = ReprPC, ReprPCWk."]
def swap(s, g):
    pic = next(x for x in s.shapes if x.name == "Matrix"); px = Image.open(g).size; assert px == (4200, 3750), px; _, rid = s.part.get_or_add_image_part(g); pic._element.blipFill.find("{http://schemas.openxmlformats.org/drawingml/2006/main}blip").set("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed", rid); xml = s.part._element.xml
    for r, rel in list(s.part.rels.items()):
        if rel.reltype.endswith("/image") and f'"{r}"' not in xml: s.part.rels.pop(r)
def corr(s, a, b, concl, subtitle=None):
    swap(s, f"fig/heatmap_all_{a}_{b}.png")
    sh = next(x for x in s.shapes if x.has_table); t = sh.table; old = {r.cells[0].text.replace(" only", ""): (r.cells[1].text, r.cells[2].text, r.cells[3].text) for r in list(t.rows)[1:]}; assert set(old) == {"Enhancer", "Repressive", "Polycomb"}, old; tb = t._tbl; trs = tb.tr_lst; proto = trs[1]
    for k in range(3): tb.insert(tb.index(proto), copy.deepcopy(proto))
    vals = [("All elements", fl(A.loc[a, b]), "—"), ("No quiescent", fl(Nq.loc[a, b]), "—"), ("TSS", fl(Ts.loc[a, b]), "—")] + [(k, old[k][0], f"{old[k][1]} ({old[k][2]})") for k in ("Enhancer", "Repressive", "Polycomb")]
    settext(t.cell(0, 0), "Elements"); settext(t.cell(0, 1), "Percentile"); settext(t.cell(0, 2), "Within local cluster (TFs)")
    for i, (l, x, y) in enumerate(vals, 1): settext(t.cell(i, 0), l); settext(t.cell(i, 1), x); settext(t.cell(i, 2), y)
    for tr in tb.tr_lst: tr.remove(tr.tc_lst[3])
    g = tb.tblGrid; g.remove(g.gridCol_lst[3])
    for c, w in zip(t.columns, (2.2, 1.3, 2.3)): c.width = Inches(w)
    for i, r in enumerate(t.rows): r.height = Inches(.5 if i == 0 else .34)
    sh.name = "Percentiles"; sh.top = Inches(1.3); f = next(x for x in s.shapes if x.name == "Footnote"); f.top = Inches(4.05); sett(f, FOOT); sett(next(x for x in s.shapes if x.name == "Conclusion"), [concl]); r0 = next(x for x in s.shapes if x.name == "Conclusion").text_frame.paragraphs[0]; q = copy.deepcopy(r0.runs[0]._r); r0._p.insert(list(r0._p).index(r0.runs[0]._r), q); r0.runs[0].text = "Conclusion: "; r0.runs[0].font.bold = True; r0.runs[1].font.bold = False
    if subtitle: sett(next(x for x in s.shapes if x.name == "Subtitle"), [subtitle])
    print(a, b, vals)
corr(find("PRC2 core EZH2–MTF2- layer 2"), "EZH2", "MTF2", "EZH2–MTF2 is high across every element set (percentile 0.90–0.96).", "ChromHMM matrix, all elements (percentile); PRC2 members outlined")
corr(find("Novel MLX–SREBF2- layer 2"), "MLX", "SREBF2", "MLX–SREBF2 is moderate overall (0.71 on all elements), higher on enhancer and Polycomb (0.78, 0.81) and low on repressive (0.44).", "Percentile 0.44–0.81 across element sets. MLX and SREBF2 join only high in the tree, so local ≈ global")
s1 = slides[0]; t = next(x for x in s1.shapes if x.has_table).table
def pairs(n):
    a, b = n.split("–")
    if "/" not in b: return [(a, b)]
    return [(a, x if x.startswith(("RBBP", "HP", "EZH")) else b.split("/")[0][:-1] + x) for x in b.split("/")]
mo = next(r for r in t.rows if r.cells[0].text.strip() == "MLX–SREBF2"); settext(list(mo.cells)[4], "Shared E-box (Tomtom q = 9 × 10⁻⁷)")
for r in list(t.rows)[1:]:
    c = r.cells[3]; was = any(x.font.bold for q in c.text_frame.paragraphs for x in q.runs); oldv = c.text; pp = pairs(r.cells[0].text); v = sum(A.loc[a, b] for a, b in pp) / len(pp); nv = fl(v); now = float(nv) >= .90; settext(c, nv, now); print(r.cells[0].text, oldv, "->", nv, "BOLD CHANGE" if was != now else "", "now bold" if now else "")
s6 = find("Novel MLX–SREBF2- layer 3"); pic6 = next(x for x in s6.shapes if x.shape_type == 13); g = "fig/motif_shared2.png"; px = Image.open(g).size; assert px == (5040, 3210), px; _, rid = s6.part.get_or_add_image_part(g); pic6._element.blipFill.find("{http://schemas.openxmlformats.org/drawingml/2006/main}blip").set("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed", rid); xml = s6.part._element.xml
for r_, rel in list(s6.part.rels.items()):
    if rel.reltype.endswith("/image") and f'"{r_}"' not in xml: s6.part.rels.pop(r_)
sett(next(x for x in s6.shapes if x.name == "Subtitle"), ["MLX and SREBF2 recognise the same E-box (CACGTG)"]); sm = next(x for x in s6.shapes if x.name == "Summary"); ps = sm.text_frame.paragraphs; assert len(ps) == 3 and ps[2].text.startswith("MLX vs SREBF2 SRE"); ps[2]._p.getparent().remove(ps[2]._p); cc = next(x for x in s6.shapes if x.name == "Conclusion").text_frame.paragraphs[0]; assert cc.runs[0].text == "Conclusion: "; cc.runs[1].text = "MLX and SREBF2 share the CACGTG E-box motif (Tomtom q = 9 × 10⁻⁷)."
for x in cc.runs[2:]: x.text = ""
n6 = s6.notes_slide.notes_text_frame; n6.text = n6.text + "\nSREBF2 SRE (MA0596.1), not shown: on the reverse strand it matches the MLX motif only at the ATCAC half-site, not significant (Tomtom q 0.06)."
hit = 0
import re
for sh in s1.shapes:
    if sh.has_text_frame:
        for q in sh.text_frame.paragraphs:
            full = "".join(x.text for x in q.runs); new = re.sub(r"≥ ?0\. ?80", "≥ 0.90", full); new = re.sub(r"normalized ChromHMM matrix, all elements, SpQN \(slide \d+\)", "ChromHMM, all elements, SpQN, each element over its 79-TF mean", new)
            if new != full:
                q.runs[0].text = new; hit += 1
                for x in q.runs[1:]: x.text = ""
print("subtitle/footnote edits", hit); assert hit == 2, hit
for x in slides[:7]:
    for sh in x.shapes:
        tx = sh.text_frame.text if sh.has_text_frame else (" ".join(c.text for r in sh.table.rows for c in r.cells) if sh.has_table else ""); assert "orrected" not in tx or not ttl(x).endswith("correlation"), (ttl(x), tx)
p.save(sys.argv[2]); print("ok", len(slides))
