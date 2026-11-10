"""Fill the local-cluster cells for all elements, no quiescent and TSS on the two correlation slides. usage: edit_lc.py in.pptx out.pptx"""
import sys
from pptx import Presentation
p = Presentation(sys.argv[1]); slides = list(p.slides); ttl = lambda s: next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()), ""); find = lambda st: next(s for s in slides if ttl(s).startswith(st))
V = {"PRC2 core EZH2–MTF2- layer 2": {"All elements": "0.81 (31)", "No quiescent": "0.82 (28)", "TSS": "0.70 (35)"}, "Novel MLX–SREBF2- layer 2": {"All elements": "0.55 (60)", "No quiescent": "0.56 (60)", "TSS": "0.56 (55)"}}
def st(q_runs, v):
    q_runs[0].text = v
    for x in q_runs[1:]: x.text = ""
for k, d in V.items():
    s = find(k); t = next(x for x in s.shapes if x.has_table).table
    for r in list(t.rows)[1:]:
        lab = r.cells[0].text.strip()
        if lab in d: assert r.cells[2].text.strip() == "—", (lab, r.cells[2].text); st(r.cells[2].text_frame.paragraphs[0].runs, d[lab])
    f = next(x for x in s.shapes if x.name == "Footnote"); hit = 0
    for q in f.text_frame.paragraphs:
        full = "".join(x.text for x in q.runs)
        if " — = not computed." in full: st(q.runs, full.replace(" — = not computed.", "")); hit += 1
    assert hit == 1, hit
s = find("Novel MLX–SREBF2- layer 2"); sb = next(x for x in s.shapes if x.name == "Subtitle"); st(sb.text_frame.paragraphs[0].runs, "Percentile 0.44–0.81 across element sets; MLX and SREBF2 share only a large branch of the tree (48–79 TFs)")
p.save(sys.argv[2]); print("ok", len(slides))
