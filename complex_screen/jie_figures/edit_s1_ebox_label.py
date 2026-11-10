# Slide 1: relabel the MLX-SREBF2 Motif cell, with column widths adjusted so it stays on one line. usage: edit_s1_ebox_label.py deck out
import sys
from pptx import Presentation
from pptx.util import Inches
p = Presentation(sys.argv[1]); T = next(x for x in p.slides[0].shapes if x.name == "Findings").table
c = next(r.cells[4] for r in T.rows if r.cells[0].text == "MLX–SREBF2"); assert c.text == "Shared E-box (q = 9 × 10⁻⁷)", c.text
c.text_frame.paragraphs[0].runs[0].text = "E-box motif similarity (q = 9 × 10⁻⁷)"
tot = sum(x.width for x in T.columns)
for i, d in {0: -.06, 1: -.10, 2: -.10, 3: -.12, 4: .48, 5: -.10}.items(): T.columns[i].width += Inches(d)
assert sum(x.width for x in T.columns) == tot
p.save(sys.argv[2]); print("ok", [round(x.width / 914400, 2) for x in T.columns])
