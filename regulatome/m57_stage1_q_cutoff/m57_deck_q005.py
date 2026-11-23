# m57: cutoff slide switched to the q <= 0.05 peak setting (promoter + ABC + 10 kb), with one definition line each for |z|, gene score and fold change.
import sys
import pandas as pd
from pptx import Presentation
from pptx.util import Pt, Emu
from pptx.dml.color import RGBColor
IN, OUT = sys.argv[1], sys.argv[2]; X = "/work/users/d/y/dyy12/XuLab/regulatome/m57_stage1_q_cutoff/out"
def q(sheet):
    d = pd.read_excel(f"{X}/stage1_precision_recall_report.xlsx", sheet_name=sheet); r = d[d.setting == "q <= 0.05"].iloc[0]; return int(r["targets per TF (median)"]), r["recall %"], r["precision %"], int(r["TFs with no links"])
z2, t10 = q("report_z2"), q("report_top10")
F = pd.read_excel(f"{X}/stage1_fold_change_recall_precision_raw.xlsx", sheet_name="pooled"); F["padding"] = F.padding.astype(str)
r = F[(F["peak set"] == "q <= 0.05") & (F.padding == "10 kb") & (F["rule family"] == "A fold change alone") & (F.rule == "fold >= 1.1") & (F.direction == "either")].iloc[0]; fc = (int(r["targets per TF, median"]), r["recall (%)"], r["precision (%)"])
VAL = [["|z| ≥ 2", f"{z2[0]:,}", f"{z2[2]:.1f}%", f"{z2[1]:.1f}%"], ["Gene score, top 10 % per TF", f"{t10[0]:,}", f"{t10[2]:.1f}%", f"{t10[1]:.1f}%"], ["Fold change ≥ 1.1", f"{fc[0]:,}", f"{fc[2]:.1f}%", f"{fc[1]:.1f}%"]]
p = Presentation(IN); n0 = len(p.slides)
hit = [(i, s) for i, s in enumerate(p.slides) if any(sh.name == "FIG5S1C-table-cutoff" for sh in s.shapes)]; assert len(hit) == 1; i, s = hit[0]
tbl = [sh for sh in s.shapes if sh.name == "FIG5S1C-table-cutoff"][0]; t = tbl.table
assert [t.cell(0, c).text for c in range(4)] == ["Perturb-seq cutoff", "Genes per TF (median)", "Precision", "Recall"]
for ri, row in enumerate(VAL, start=1):
    for c, v in enumerate(row): t.cell(ri, c).text_frame.paragraphs[0].runs[0].text = v
note = [sh for sh in s.shapes if sh.name == "FIG5S1C-note"][0]; tf = note.text_frame; tf.word_wrap = True
LINES = [f"Our links: MACS2 peaks with q < 0.05 per TF, promoter + ABC + 10 kb; 63 TFs ({z2[3]} with no linked genes). Recall = shared / Perturb-seq genes; precision = shared / our linked genes.",
         "|z|: difference in mean log1p(CP10k) between the TF's knockdown cells (top 25 % by sgRNA) and batch-matched control cells, divided by its standard error; |z| ≥ 2 ≈ nominal p < 0.05.",
         "Gene score: per knockdown cell, (expression − control mean) / control SD in the same batch, averaged over the TF's cells (X-Atlas/Orion); top 10 % = the 10 % of genes with the largest |score| for that TF.",
         "Fold change: mean CP10k in knockdown cells / mean CP10k in batch-matched control cells (+ 0.1 each), either direction, no significance test."]
for para in list(tf.paragraphs)[1:]: para._p.getparent().remove(para._p)
p0 = tf.paragraphs[0]
for rr in p0.runs[1:]: rr._r.getparent().remove(rr._r)
for k, line in enumerate(LINES):
    para = p0 if k == 0 else tf.add_paragraph()
    run = para.runs[0] if (k == 0 and para.runs) else para.add_run(); run.text = line; run.font.name = "Cambria"; run.font.size = Pt(13); run.font.color.rgb = RGBColor(0x59, 0x59, 0x59); para.space_after = Pt(4)
note.top = tbl.top + tbl.height + Emu(182880); note.height = Emu(1554480)
s.notes_slide.notes_text_frame.text = "Source: ~/Desktop/HEK293T-ATGTAG/perturbseq_matrix/binary_overlap/stage1_q_cutoff/stage1_precision_recall_report.xlsx (report_z2 / report_top10, row q <= 0.05; Longleaf job 10507161) and stage1_fold_change_recall_precision_raw.xlsx (job 10493662)."
p.save(OUT); qq = Presentation(OUT); assert len(qq.slides) == n0
t = [sh for sh in qq.slides[i].shapes if sh.name == "FIG5S1C-table-cutoff"][0].table
print("slide", i + 1, "of", n0); [print(" | ".join(t.cell(r_, c).text for c in range(4))) for r_ in range(4)]
print("\n".join(LINES)); print("done")
