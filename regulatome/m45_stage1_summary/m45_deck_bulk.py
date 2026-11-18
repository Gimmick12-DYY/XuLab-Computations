# m45 Amendment 2: bulk ChIP (HEK293T / HEK293 only) vs our HITAG per TF in the Stage 1 section; pooled K562-containing table removed. usage: m45_deck_bulk.py <in.pptx> <out.pptx>
import sys, zipfile, collections, re
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
IN, OUT = sys.argv[1], sys.argv[2]; M = "/work/users/d/y/dyy12/XuLab/regulatome/m45_stage1_summary"; MK = "FIG5S1"
T = pd.read_csv(f"{M}/out/11_bulk_chip_hek_per_tf.tsv", sep="\t", dtype={"padding": str}); TF7 = list(dict.fromkeys(T.TF))
CUT = ["|z| >= 2", "|z| >= 3", "|z| >= 4", "|z| >= 5"]; CL = dict(zip(CUT, ["|z| ≥ 2", "|z| ≥ 3", "|z| ≥ 4", "|z| ≥ 5"]))
INK, GREY, HEAD = "262626", "595959", "E3E3E3"
fp = lambda p: "0.001" if p < 0.0015 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"
REFS = "Bulk ChIP lists (HEK293T / HEK293 only; K562 lists for NRF1, E4F1, NFYA removed): CTCF HEK293T (GSE103651); ZNF282 HEK293T (GSM2466515); MAZ HEK293 (ENCFF529XTF); ZIC2 HEK293 (ENCFF359GIY); ZNF777 HEK293 (ENCFF735CAY); ZBTB7A HEK293 (ENCFF114DGV); SETDB1 HEK293 (ENCODE reads ENCFF000ZTR)."
p = Presentation(IN); n0 = len(p.slides); lay = {l.name: l for l in p.slide_layouts}
assert not any(sh.name.startswith(f"{MK}-table-bulktf") for s in p.slides for sh in s.shapes), "already present"
def tb(sl, name, x, y, w, h, text, size, bold, color):
    s = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); s.name = name; tf = s.text_frame; tf.word_wrap = True
    for i, t in enumerate(text if isinstance(text, list) else [text]):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); r = para.add_run(); r.text = t; f = r.font; f.name = "Cambria"; f.size = Pt(size); f.bold = bold; f.color.rgb = RGBColor.from_string(color)
    return s
def blank(title, sub):
    sl = p.slides.add_slide(lay["Title and Content"])
    for ph in list(sl.placeholders): ph._element.getparent().remove(ph._element)
    tb(sl, f"{MK}-title", 0.17, 0.12, 12.9, 0.57, title, 28, True, INK); tb(sl, f"{MK}-subtitle", 0.33, 0.68, 12.69, 0.4, sub, 15, False, GREY); return sl
def settext(sh, text):
    para = sh.text_frame.paragraphs[0]; para.runs[0].text = text
    for x in para.runs[1:]: x.text = ""
    for extra in list(sh.text_frame.paragraphs[1:]): extra._p.getparent().remove(extra._p)
# 1. slide with the pooled bulk table: drop that table, keep leave-one-out
host = None
for i, s in enumerate(p.slides):
    for sh in list(s.shapes):
        if sh.name == f"{MK}-table-bulk": sh._element.getparent().remove(sh._element); host = i
assert host is not None
s = p.slides[host]
for sh in s.shapes:
    if sh.name == f"{MK}-title": settext(sh, "Regulome- Stage 1 · padding 10 kb: leave one TF out")
    if sh.name == f"{MK}-subtitle": settext(sh, "Matched gene shuffle with each of the 63 TFs left out in turn; HEK293T, top-25% sgRNA cells")
    if sh.name == f"{MK}-note": settext(sh, "Gap = recall observed minus shuffle mean. No single TF carries the 10 kb result.")
# 2. per-TF slides
H1 = ["TF", "Cutoff", "Perturb-seq genes", "Bulk ChIP", "", "", "", "", "Our HITAG", "", "", "", ""]
H2 = ["", "", "", "Genes in list", "In both", "Recall", "Shuffle mean [range]", "p", "Genes in list", "In both", "Recall", "Shuffle mean [range]", "p"]
CW = [1.05, 0.8, 1.0, 0.85, 0.7, 0.75, 1.55, 0.6, 0.85, 0.7, 0.75, 1.55, 0.6]
new = []
for pad in ("0", "10 kb"):
    for cuts, lab in ((CUT[:2], "|z| ≥ 2–3"), (CUT[2:], "|z| ≥ 4–5")):
        sl = blank(f"Regulome- Stage 1 · bulk ChIP per TF, padding {pad}, {lab}", "Bulk ChIP (HEK293T / HEK293) vs our HITAG; recall %, matched gene shuffle (1,000); bold: p < 0.05; top-25% sgRNA cells")
        rows = []
        for t in TF7:
            for c in cuts:
                r = T[(T.padding == pad) & (T.TF == t) & (T.cutoff == c)].iloc[0]; n = int(r["Perturb-seq genes"]); row = [t if c == cuts[0] else "", CL[c], f"{n:,}"]
                for src in ("bulk ChIP", "our HITAG"):
                    if n < 10: row += [(f"{int(r[f'{src}: genes in list']):,}", False), "–", ("n < 10", False), "–", "–"]
                    else:
                        b = r[f"{src}: p (one-sided)"] < 0.05
                        row += [(f"{int(r[f'{src}: genes in list']):,}", False), (f"{int(r[f'{src}: in both'])}", False), (f"{r[f'{src}: recall (%)']:.1f}", b), (f"{r[f'{src}: shuffle mean (%)']:.1f} [{r[f'{src}: shuffle 2.5-97.5 (%)'].replace('-', '–')}]", False), (fp(r[f"{src}: p (one-sided)"]), b)]
                rows.append(row)
        nr = len(rows) + 2; gs = sl.shapes.add_table(nr, len(CW), Inches(0.25), Inches(1.2), Inches(sum(CW)), Inches(0.33 * nr)); gs.name = f"{MK}-table-bulktf-{pad.replace(' ', '')}-{cuts[0][-1]}"; tt = gs.table; tt.horz_banding = False
        for j, w in enumerate(CW): tt.columns[j].width = Inches(w)
        for i, row in enumerate([H1, H2] + rows):
            tt.rows[i].height = Inches(0.33)
            for j, v in enumerate(row):
                txt, bold = v if isinstance(v, tuple) else (v, False); c = tt.cell(i, j); c.fill.solid(); c.fill.fore_color.rgb = RGBColor.from_string(HEAD if i < 2 else "FFFFFF")
                c.margin_left = c.margin_right = Inches(0.04); c.margin_top = c.margin_bottom = Inches(0.02); tf = c.text_frame; tf.word_wrap = True
                r_ = tf.paragraphs[0].add_run(); r_.text = txt; f = r_.font; f.name = "Cambria"; f.size = Pt(12); f.bold = i < 2 or j == 0 or bold; f.color.rgb = RGBColor.from_string(INK)
        tt.cell(0, 3).merge(tt.cell(0, 7)); tt.cell(0, 8).merge(tt.cell(0, 12))
        for j in range(3): tt.cell(0, j).merge(tt.cell(1, j))
        tb(sl, f"{MK}-note", 0.25, 1.2 + 0.33 * nr + 0.1, 12.8, 0.9, REFS, 12, False, GREY); new.append(sl)
bad = [(sh.name, sh.text_frame.text[:80]) for s in new for sh in s.shapes if sh.has_text_frame and sh.name != f"{MK}-note" and re.search(r"expected|fold|chance|precision|validated|K562|HepG2", sh.text_frame.text, re.I)]
print("flagged:", bad); assert not bad
lst = p.slides._sldIdLst; ids = list(lst); k = len(new)
for j, el in enumerate(ids[-k:]): lst.remove(el); lst.insert(host + 1 + j, el)
p.part.rename_slide_parts([el.rId for el in lst])
for i, s in enumerate(p.slides, 1):
    if s.has_notes_slide: s.notes_slide.part.partname = s.notes_slide.part.partname.__class__(f"/ppt/notesSlides/notesSlide{i}.xml")
names = [s.part.partname for s in p.slides]; assert len(set(names)) == len(names)
nn = [s.notes_slide.part.partname for s in p.slides if s.has_notes_slide]; assert len(set(nn)) == len(nn)
# no K562 / HepG2 number left in the Stage 1 section (except the footnote that says they were removed)
st1 = []; 
for s in p.slides:
    tt_ = " ".join(sh.text_frame.text for sh in s.shapes if sh.has_text_frame)
    if "11/17/26" in tt_ and not any(x.name.startswith(MK) for x in s.shapes): break
    st1.append(s)
left = [(i + 1, sh.name) for i, s in enumerate(st1) for sh in s.shapes if (sh.has_text_frame and sh.name != f"{MK}-note" and re.search(r"K562|HepG2", sh.text_frame.text)) or (sh.has_table and any(re.search(r"K562|HepG2", c.text) for row in sh.table.rows for c in row.cells))]
print("K562/HepG2 left in Stage 1 section:", left); assert not left
assert len(p.slides) == n0 + k; p.save(OUT)
q = Presentation(OUT); print("reopened: slides", len(q.slides), "(before", n0, "+", k, ")")
for i in range(len(st1) + k + 1):
    s = q.slides[i]; print(i + 1, [sh.text_frame.text[:75] for sh in s.shapes if sh.has_text_frame][:1])
z = zipfile.ZipFile(OUT); c = collections.Counter(z.namelist()); dup = [x for x, v in c.items() if v > 1]; print("zip entries", len(c), "duplicates", dup); assert not dup
