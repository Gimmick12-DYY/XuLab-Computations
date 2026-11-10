# Copy sheet: section title + 12 slides of native tables (top 10 HOMER known/de novo and AME per run) for SREBF2 and MLX, inserted at the very top. usage: edit_motif_sheets.py deck out
import csv, glob, os, re, sys
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
H = "/private/tmp/claude-501/-Users-dyy-Desktop-HEK293T-ATGTAG/acc96b1a-dc2b-4d74-ada1-3f2fa43a4e7a/scratchpad/homer_all"
BASE = "/work/users/d/y/dyy12/XuLab/motif_analysis/"; INK = RGBColor(0x1A, 0x1A, 0x1A); GREY = RGBColor(0x59, 0x59, 0x59); FS = 12
NOTE = {("srebf2", "motif", "homer"): "-bg with 1,635 bp background vs 231 bp targets; not comparable", ("srebf2", "motif", "homer_ocr"): "-cpg -olen 3 (MLX run without)"}
def box(s, x, y, w, h, text, sz, bold=False, color=INK, name=None):
    b = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = b.text_frame; tf.word_wrap = True; r = tf.paragraphs[0].add_run(); r.text = text; r.font.size = Pt(sz); r.font.bold = bold; r.font.color.rgb = color; r.font.name = "Cambria"
    if name: b.name = name
    return b
def table(s, x, y, widths, head, rows, name):
    G = s.shapes.add_table(len(rows) + 1, len(head), Inches(x), Inches(y), Inches(sum(widths)), Inches(.26 * (len(rows) + 1))); G.name = name; T = G.table; T.first_row = True; T.horz_banding = False
    for i, w in enumerate(widths): T.columns[i].width = Inches(w)
    for i, row in enumerate([head] + rows):
        T.rows[i].height = Inches(.26)
        for j, v in enumerate(row):
            c = T.cell(i, j); c.margin_left = c.margin_right = Inches(.05); c.margin_top = c.margin_bottom = Inches(.005); c.vertical_anchor = 3; r = c.text_frame.paragraphs[0].add_run(); r.text = str(v); r.font.size = Pt(FS); r.font.bold = i == 0; r.font.color.rgb = INK; r.font.name = "Cambria"
            c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0xE3, 0xE3, 0xE3) if i == 0 else RGBColor(0xFF, 0xFF, 0xFF)
def known(f):
    R = list(csv.reader(open(f), delimiter="\t")); return R[0], R[1:]
def denovo(d):
    L = []
    for f in glob.glob(f"{d}/motif*.motif"):
        if f.endswith("RV.motif"): continue
        h = open(f).readline().rstrip("\n").split("\t"); m = re.search(r"T:[\d.]+\(([\d.]+%)\),B:[\d.]+\(([\d.]+%)\)", h[5]); g = re.search(r"BestGuess:(.*)$", h[1])
        L.append((float(h[3]), [h[0][1:], h[3], m[1], m[2], g[1] if g else "–"]))
    return [x[1] for x in sorted(L, key=lambda x: x[0])[:10]]
def settings(tf, run, bg):
    cmd = open(f"{H}/params/{run}_{tf}_{bg}.txt").readline().strip(); fl = re.sub(r"\S*/", "", cmd.split(" hg38 ", 1)[1]).split(" ", 1)[1] if " hg38 " in cmd else cmd; h, _ = known(f"{H}/{run}/{tf}/{bg}/knownResults.txt")
    t = re.search(r"of (\d+)\)", h[5])[1]; b = re.search(r"of (\d+)\)", h[7])[1]; s = f"Targets {int(t):,}, background {int(b):,}; hg38; {fl.strip()}"
    return s + ("; " + NOTE[(tf, run, bg)] if (tf, run, bg) in NOTE else "")
def homer_slide(p, tf, run, bg):
    s = p.slides.add_slide(BLANK); box(s, .17, .12, 12.9, .57, f"{tf.upper()} · {run} · {bg}", 28, True, name="Title"); box(s, .12, .66, 12.9, .3, settings(tf, run, bg), 11, color=GREY, name="Settings")
    h, R = known(f"{H}/{run}/{tf}/{bg}/knownResults.txt"); n = sum(float(r[4]) < .05 for r in R); A = [[r[0].split("/")[0], r[3], r[4], r[6], r[8]] for r in R[:10]]; D = denovo(f"{H}/{run}/{tf}/{bg}/homerResults")
    box(s, .12, 1.08, 6, .3, f"HOMER known: {n} of {len(R)} at q < 0.05", 12, True, name="Known caption"); box(s, 6.15, 1.08, 6.8, .3, f"HOMER de novo: top 10 by log p ({len(glob.glob(f'{H}/{run}/{tf}/{bg}/homerResults/motif*[0-9].motif'))} motifs)", 12, True, name="De novo caption")
    table(s, .17, 1.38, [2.4, 1.05, .72, .95, .8], ["Motif", "log p", "q", "% target", "% bg"], A, "HOMER known"); table(s, 6.2, 1.38, [1.45, 1.15, .95, .72, 2.6], ["Consensus", "log p", "% target", "% bg", "Best match"], D, "HOMER de novo")
    box(s, .12, 6.95, 12.9, .3, f"{run}/{tf}/{bg}/knownResults.txt; {run}/{tf}/{bg}/homerResults/motif*.motif; settings: {run}/{tf}/{bg}/motifFindingParameters.txt", 10, color=GREY, name="Paths")
def ame_read(tf, run, bg):
    f = f"{H}/{run}/{tf}/{bg}/{tf}/ame.tsv"; L = [l.rstrip("\n").split("\t") for l in open(f) if l.strip() and not l.startswith("#")]; c = [l for l in open(f) if l.startswith("# ame")][0]
    return L[0], L[1:], c
def ame_slide(p, tf, run):
    s = p.slides.add_slide(BLANK); box(s, .17, .12, 12.9, .57, f"{tf.upper()} · {run} · ame_ocr + ame", 28, True, name="Title"); txt = []
    for k, (bg, x) in enumerate((("ame_ocr", .17), ("ame", 6.6))):
        h, R, c = ame_read(tf, run, bg); ix = {v: i for i, v in enumerate(h)}; n = sum(float(r[ix["E-value"]]) < .05 for r in R); ctl = "--control " + tf + ".background.fa" if "--control" in c else "(no --control)"
        txt.append(f"{bg}: ame --scoring max {ctl}; HOCOMOCOv11 full; " + (f"pos {int(R[0][ix['pos']]):,}, neg {int(R[0][ix['neg']]):,}" if bg == "ame_ocr" else f"{len(R)} rows")); box(s, x - .05, 1.08, 6, .3, f"{bg}: {n} of {len(R)} listed at E < 0.05", 12, True, name=f"{bg} caption")
        table(s, x, 1.38, [2.1, 1.0, 1.0, 1.0, .62, .62], ["Motif", "p", "adj p", "E", "%TP", "%FP"], [[r[ix["motif_ID"]], r[ix["p-value"]], r[ix["adj_p-value"]], r[ix["E-value"]], r[ix["%TP"]], r[ix["%FP"]]] for r in R[:10]], bg)
    box(s, .12, .66, 12.9, .3, "; ".join(txt), 11, color=GREY, name="Settings"); box(s, .12, 6.95, 12.9, .3, f"{run}/{tf}/ame_ocr/{tf}/ame.tsv; {run}/{tf}/ame/{tf}/ame.tsv", 10, color=GREY, name="Paths")
p = Presentation(sys.argv[1]); n0 = len(p.slides); BLANK = next(l for l in p.slide_layouts if l.name == "Blank")
s = p.slides.add_slide(BLANK); box(s, .5, 2.9, 12.3, .9, "11/3/26 Motif results (MLX, SREBF2)", 36, True, name="Title"); box(s, .5, 3.75, 12.3, .4, BASE, 12, color=GREY, name="Base path")
for tf in ("srebf2", "mlx"):
    for run in ("motif_bins", "motif"):
        homer_slide(p, tf, run, "homer_ocr"); homer_slide(p, tf, run, "homer"); ame_slide(p, tf, run)
L = p.slides._sldIdLst; ids = list(L); new = ids[n0:]; assert len(new) == 13
for x in new: L.remove(x)
for i, x in enumerate(new): L.insert(i, x)
p.save(sys.argv[2]); print("ok", n0, len(p.slides))
