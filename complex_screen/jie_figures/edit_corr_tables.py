"""Replace only the percentile table and its footnote on the two correlation slides; every other shape and slide stays as it is. usage: edit_corr_tables.py in.pptx out.pptx"""
import json, sys
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR
IN, OUT = sys.argv[1], sys.argv[2]; L = json.load(open("fig/local_cluster.json")); p = Presentation(IN); f2 = lambda x: f"{np.floor(x * 100 + 1e-9) / 100:.2f}"; LAB = {"enhancer": "Enhancer", "repressive": "Repressive", "polycomb": "Polycomb only"}
def run(par, t, sz, b=False, c="1A1A1A"):
    r = par.add_run(); r.text = t; r.font.name = "Cambria"; r.font.size = Pt(sz); r.font.bold = b; r.font.color.rgb = RGBColor.from_string(c)
done = []
for s in p.slides:
    tt = next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()), "")
    key = "EZH2-MTF2" if tt.startswith("PRC2 core EZH2–MTF2- layer 2") else ("MLX-SREBF2" if tt.startswith("Novel MLX–SREBF2- layer 2") else None)
    if not key: continue
    gone = [sh for sh in s.shapes if sh.has_table or (sh.has_text_frame and sh.name in ("Footnote", "Summary", "Reading"))]; print(tt[:44], "removing", [sh.name for sh in gone], "| keeping", [sh.name for sh in s.shapes if sh not in gone])
    for sh in gone: sh._element.getparent().remove(sh._element)
    rows = [["Corrected elements", "Percentile", "Within local cluster", "TFs in cluster"]] + [[LAB[k], f2(v["pct"]), f2(v["local"]), str(v["n_tfs"])] for k, v in L[key].items()]; W = [2.2, 1.1, 1.4, 1.1]
    T = s.shapes.add_table(len(rows), 4, Inches(7.3), Inches(1.6), Inches(sum(W)), Inches(0.5 * len(rows))); T.name = "Corrected percentiles"; t = T.table
    for j, w in enumerate(W): t.columns[j].width = Inches(w)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = t.cell(i, j); c.vertical_anchor = MSO_ANCHOR.MIDDLE; c.fill.solid(); c.fill.fore_color.rgb = RGBColor.from_string("E3E3E3" if i == 0 else "FFFFFF"); run(c.text_frame.paragraphs[0], v, 14, i == 0 or j == 0)
    o = s.shapes.add_textbox(Inches(7.25), Inches(3.8), Inches(5.9), Inches(2.6)); o.name = "Footnote"; tf = o.text_frame; tf.word_wrap = True
    for k, line in enumerate(["Corrected = accessibility-corrected: each element's signal over its 79-TF mean, then Pearson, SpQN, percentile; elements with ≥ 20 reads. The matrix shown is before this correction.", "Local cluster = smallest branch of that set's TF tree (average linkage, 1 − Pearson r) containing both TFs. Within local cluster ranks the pair among TF pairs in that branch.", "Repressive = ReprPC, ReprPCWk, Het, ZNF/Rpts. Polycomb = ReprPC, ReprPCWk."]):
        par = tf.paragraphs[0] if k == 0 else tf.add_paragraph(); par.space_after = Pt(3); run(par, line, 12, False, "4D4D4D")
    done.append(key)
assert sorted(done) == ["EZH2-MTF2", "MLX-SREBF2"], done
p.save(OUT); print("saved", OUT.split("/")[-1], len(p.slides))
