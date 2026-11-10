# Per-slide PNG previews (approximate render, tables included), using contact_sheet.py at full scale. usage: preview_jie.py deck first last outdir
import sys, subprocess
deck, a, b, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
src = open("../contact_sheet.py").read().replace("sc = 0.5", "sc = 1.0").replace("cols = 2;", "cols = 1;").replace("hh = text(td, 0, 0, cw, c.text_frame.paragraphs, False) + 4", "hh = text(td, 0, 0, cw, c.text_frame.paragraphs, False) + 4 - 9 * DPI / 72 * len(c.text_frame.paragraphs)")  # tables have no paragraph gap on the slide
for n in range(a, b + 1):
    sys.argv = ["x", deck, str(n), str(n), f"{out}/slide_{n}.png"]; exec(src, {})
