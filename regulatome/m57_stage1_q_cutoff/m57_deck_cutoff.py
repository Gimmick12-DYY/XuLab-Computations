# m57: one slide with the Perturb-seq cutoff table (|z| >= 2, gene score top 10 %, fold change >= 1.1): genes per TF, recall, precision; inserted after the last Stage 1 slide.
import sys, copy, re, zipfile, collections
import pandas as pd
from pptx import Presentation
from pptx.util import Pt, Emu
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn
IN, OUT = sys.argv[1], sys.argv[2]
TR = pd.read_excel("/work/users/d/y/dyy12/XuLab/regulatome/m57_stage1_q_cutoff/out/stage1_precision_recall_report.xlsx", sheet_name="target_rules").set_index("target rule")
ROWS = [("|z| ≥ 2", TR.loc["|z| >= 2"]), ("Gene score, top 10 % per TF", TR.loc["top 10 % of |score| per TF"]), ("Fold change ≥ 1.1", TR.loc["fold >= 1.1"])]
CELLS = [["Perturb-seq cutoff", "Genes per TF (median)", "Recall", "Precision"]] + [[lab, f"{int(r['per TF median']):,}", f"{r['headline recall %']:.1f}%", f"{r['headline precision %']:.1f}%"] for lab, r in ROWS]
p = Presentation(IN); n0 = len(p.slides)
src = [s for s in p.slides if any(sh.name == "FIG5S1B-table-peakset" for sh in s.shapes)]; assert len(src) == 1; src = src[0]
idx = [i for i, s in enumerate(p.slides) if any(sh.name.startswith("FIG5S1") for sh in s.shapes)]; last = max(idx)
assert not any(sh.name == "FIG5S1C-table-cutoff" for s in p.slides for sh in s.shapes), "slide already present"
new = p.slides.add_slide(src.slide_layout)
for sh in list(new.shapes): sh._element.getparent().remove(sh._element)
for sh in src.shapes:
    if sh.name == "FIG5S1B-title": new.shapes._spTree.append(copy.deepcopy(sh._element))
tt = [sh for sh in new.shapes if sh.name == "FIG5S1B-title"][0]; tt.name = "FIG5S1C-title"; para = tt.text_frame.paragraphs[0]
for r in para.runs[1:]: r._r.getparent().remove(r._r)
para.runs[0].text = "Perturb-seq cutoff: genes per TF, recall and precision"
stbl = [sh for sh in src.shapes if sh.name == "FIG5S1B-table-peakset"][0]; style_id = stbl._element.graphic.graphicData.tbl.tblPr.find(qn("a:tableStyleId")).text
L, T, W = Emu(274320), Emu(1143000), Emu(11612880); colw = [Emu(4206240), Emu(2468880), Emu(2468880), Emu(2468880)]; hh, rh = Emu(457200), Emu(640080)
gf = new.shapes.add_table(4, 4, L, T, W, hh + 3 * rh); gf.name = "FIG5S1C-table-cutoff"; tb = gf.table
tb._tbl.tblPr.find(qn("a:tableStyleId")).text = style_id
for j, w in enumerate(colw): tb.columns[j].width = w
tb.rows[0].height = hh
for i in range(1, 4): tb.rows[i].height = rh
for i, row in enumerate(CELLS):
    for j, val in enumerate(row):
        c = tb.cell(i, j); c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0xE3, 0xE3, 0xE3) if i == 0 else RGBColor(0xFF, 0xFF, 0xFF)
        c.margin_left = c.margin_right = Emu(91440); c.vertical_anchor = 3 if False else c.vertical_anchor
        tf = c.text_frame; tf.paragraphs[0].text = ""; run = tf.paragraphs[0].add_run(); run.text = val; f = run.font; f.name = "Cambria"; f.size = Pt(16); f.bold = (i == 0 or j == 0); f.color.rgb = RGBColor(0x26, 0x26, 0x26)
tx = new.shapes.add_textbox(L, T + hh + 3 * rh + Emu(182880), W, Emu(457200)); tx.name = "FIG5S1C-note"; tf = tx.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run(); r.text = "Our links: top 10,000 peaks per TF, promoter + ABC + 10 kb; 63 TFs. Recall = shared / Perturb-seq genes; precision = shared / our linked genes."; r.font.name = "Cambria"; r.font.size = Pt(14); r.font.color.rgb = RGBColor(0x59, 0x59, 0x59)
new.notes_slide.notes_text_frame.text = "Source: ~/Desktop/HEK293T-ATGTAG/perturbseq_matrix/binary_overlap/stage1_q_cutoff/stage1_precision_recall_report.xlsx, sheet target_rules (Longleaf job 10507161)."
lst = p.slides._sldIdLst; el = list(lst)[-1]; lst.remove(el); lst.insert(last + 1, el)
txt = " ".join(sh.text_frame.text for sh in new.shapes if sh.has_text_frame) + " " + new.notes_slide.notes_text_frame.text
assert not re.search(r"claude|anthropic|generated with", txt, re.I)
names = [s.part.partname for s in p.slides]; assert len(set(names)) == len(names)
p.save(OUT); z = zipfile.ZipFile(OUT); dup = [k for k, v in collections.Counter(z.namelist()).items() if v > 1]; assert not dup, dup
q = Presentation(OUT); assert len(q.slides) == n0 + 1; pos = [i for i, s in enumerate(q.slides) if any(sh.name == "FIG5S1C-table-cutoff" for sh in s.shapes)][0]
print("slides", n0, "->", len(q.slides), "| new slide number", pos + 1, "| previous Stage 1 slide number", last + 1)
for row in CELLS: print(" | ".join(row))
print("done")
