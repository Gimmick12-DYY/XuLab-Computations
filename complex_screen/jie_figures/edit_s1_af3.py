# Slide 1: AF3 tiers into Structural support cells of the Boltz-only novel rows and into the notes list. usage: edit_s1_af3.py deck out
import sys, copy
import pandas as pd
from pptx import Presentation
A = pd.read_csv("data/summary_novel_af3.tsv", sep="\t"); A = A[A.run.notna()]
def af3(r):
    if r.run == "1 seed": return f"AF3 {r.af3_tier} (1 seed)"
    return f"AF3 strong {int(r.seeds)}/5" if r.af3_tier == "strong" else f"AF3 {r.af3_tier}"
M = {f"{r.a}–{r.b}": af3(r) for r in A.itertuples()}
def settext(par, txt):
    rs = par.runs; rs[0].text = txt
    for x in rs[1:]: x._r.getparent().remove(x._r)
p = Presentation(sys.argv[1]); s = p.slides[0]
T = next(x for x in s.shapes if x.name == "Findings").table; n = 0
for row in T.rows:
    c = row.cells[3]
    if "AF3 not run" in c.text and row.cells[0].text in M:
        settext(c.text_frame.paragraphs[0], c.text.replace("AF3 not run", M[row.cells[0].text])); n += 1
assert n == 9, n
P = s.notes_slide.notes_text_frame.paragraphs; k = 0
for q in P:
    t = "".join(r.text for r in q.runs)
    if "AF3 not run" in t:
        settext(q, t.replace("AF3 not run", M[t.split(":")[0]])); k += 1
assert k == 77, k
new = copy.deepcopy(P[1]._p); P[1]._p.addnext(new)
settext(s.notes_slide.notes_text_frame.paragraphs[2], sys.argv[3] if len(sys.argv) > 3 else open("data/s1_af3_note.txt").read().strip())
p.save(sys.argv[2]); print("ok", n, k)
