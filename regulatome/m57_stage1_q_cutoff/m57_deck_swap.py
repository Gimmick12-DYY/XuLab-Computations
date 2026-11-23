# m57: put the cutoff slide's columns in the requested order (genes per TF, precision, recall).
import sys, copy
from pptx import Presentation
IN, OUT = sys.argv[1], sys.argv[2]; p = Presentation(IN); n0 = len(p.slides)
hit = [(i, sh) for i, s in enumerate(p.slides) for sh in s.shapes if sh.name == "FIG5S1C-table-cutoff"]; assert len(hit) == 1; i, sh = hit[0]; t = sh.table
assert t.cell(0, 2).text == "Recall" and t.cell(0, 3).text == "Precision"
for r in range(len(t.rows)):
    a, b = t.cell(r, 2)._tc, t.cell(r, 3)._tc; ta, tb = copy.deepcopy(a.txBody), copy.deepcopy(b.txBody); a.remove(a.txBody); b.remove(b.txBody); a.insert(0, tb); b.insert(0, ta)
p.save(OUT); q = Presentation(OUT); assert len(q.slides) == n0
t = [sh for sh in q.slides[i].shapes if sh.name == "FIG5S1C-table-cutoff"][0].table
print("slide", i + 1, "of", n0); [print(" | ".join(t.cell(r, c).text for c in range(4))) for r in range(len(t.rows))]
