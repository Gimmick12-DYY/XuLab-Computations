# m48 Amendment 1: one slide with the padding table (top 10,000, 0 / 10 kb / 50 kb), cloned from the user's current peaks-per-TF slide and inserted right after it.
import sys, copy, re, zipfile, collections
import pandas as pd
from pptx import Presentation
from pptx.util import Pt
IN, OUT = sys.argv[1], sys.argv[2]; O = "/work/users/d/y/dyy12/XuLab/regulatome/m48_stage1_top10k/out"
TAB = pd.read_csv(f"{O}/10_padding_table.tsv", sep="\t", dtype=str)
p = Presentation(IN); n0 = len(p.slides)
src_i = [i for i, s in enumerate(p.slides) if any(sh.name == "FIG5S1B-table-peakset" for sh in s.shapes)]; assert len(src_i) == 1, src_i; src_i = src_i[0]; src = p.slides[src_i]
assert not any(sh.name == "FIG5S1B-table-padding" for s in p.slides for sh in s.shapes), "padding slide already present"
new = p.slides.add_slide(src.slide_layout)
for sh in list(new.shapes): sh._element.getparent().remove(sh._element)
for sh in src.shapes: new.shapes._spTree.append(copy.deepcopy(sh._element))
def settext(tf, text):
    para = tf.paragraphs[0]; rs = para.runs
    for r in rs[1:]: r._r.getparent().remove(r._r)
    rs[0].text = text
    for q in tf.paragraphs[1:]: q._p.getparent().remove(q._p)
for sh in new.shapes:
    if sh.name == "FIG5S1B-subtitle": settext(sh.text_frame, "Recall = share of Perturb-seq genes found in our Regulome; gap to matched gene shuffle (points); 63 TFs, top 10,000 peaks")
    if sh.name == "FIG5S1B-table-peakset":
        sh.name = "FIG5S1B-table-padding"; t = sh.table; labels = [t.cell(r, 0).text for r in range(len(t.rows))]
        keep = [0, labels.index("top 5,000"), labels.index("top 10,000"), labels.index("top 20,000")]  # header, then plain / bold (chosen) / plain row formats
        tbl = t._tbl; trs = tbl.tr_lst
        for j, tr in enumerate(trs):
            if j not in keep: tbl.remove(tr)
        settext(t.cell(0, 0).text_frame, "Padding")
        for r, (_, row) in enumerate(TAB.iterrows(), start=1):
            for c, col in enumerate(TAB.columns): settext(t.cell(r, c).text_frame, row[col])
        sh.height = sum(r.height for r in t.rows)
# move the new slide to just after the source slide
lst = p.slides._sldIdLst; el = list(lst)[-1]; lst.remove(el); lst.insert(src_i + 1, el)
ns = new.notes_slide; ns.notes_text_frame.text = ("Padding table, top 10,000 lab peaks per TF, 63 TFs. Link = summit in a promoter or canonical Hi-C ABC-linked element, plus summit within the padding of the TSS. "
  "Cells: pooled recall (one-sided p, matched gene shuffle within NTC expression x promoter ATAC deciles, 1,000 shuffles; same run as the peaks-per-TF table). Source: m48_stage1_top10k/out/stage1_top10k_padding_0_10_50kb.xlsx.")
txt = " ".join(sh.text_frame.text for sh in new.shapes if sh.has_text_frame) + " " + " ".join(c.text for sh in new.shapes if sh.has_table for r in sh.table.rows for c in r.cells)
assert not re.search(r"claude|generated|anthropic|AI-", txt + ns.notes_text_frame.text, re.I)
names = [s.part.partname for s in p.slides]; assert len(set(names)) == len(names)
p.save(OUT)
z = zipfile.ZipFile(OUT); dup = [k for k, v in collections.Counter(z.namelist()).items() if v > 1]; assert not dup, dup
q = Presentation(OUT); assert len(q.slides) == n0 + 1
s3, s4 = q.slides[src_i], q.slides[src_i + 1]
print("slides", n0, "->", len(q.slides), "| inserted at", src_i + 2, "after", [sh.name for sh in s3.shapes if sh.has_table])
for sh in s4.shapes:
    if sh.has_table:
        for r in sh.table.rows: print(" | ".join(c.text for c in r.cells), "| bold" if r.cells[0].text_frame.paragraphs[0].runs[0].font.bold else "")
    elif sh.has_text_frame: print(sh.name, ":", sh.text_frame.text)
print("done")
