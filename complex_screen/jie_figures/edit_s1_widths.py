# Slide 1: 0.2 in from Motif to Structural support so the AF3 cells stay on one line. usage: edit_s1_widths.py deck out
import sys
from pptx import Presentation
from pptx.util import Inches
p = Presentation(sys.argv[1]); T = next(x for x in p.slides[0].shapes if x.name == "Findings").table
T.columns[3].width += Inches(.2); T.columns[4].width -= Inches(.2)
p.save(sys.argv[2]); print("ok", T.columns[3].width / 914400, T.columns[4].width / 914400)
