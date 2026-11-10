"""Slide 1 wording: paralog names on HP1 rows, source-explicit structure and motif cells, footnote and subtitle; Motif column widened to keep cells on one line. usage: edit_s1_item6.py in.pptx out.pptx"""
import sys
from pptx import Presentation
from pptx.util import Emu, Inches, Pt
p = Presentation(sys.argv[1]); s = p.slides[0]; assert any(x.has_text_frame and x.text_frame.text.startswith("Cobinding Complex Analysis- all complexes") for x in s.shapes)
G = next(x for x in s.shapes if x.name == "Findings"); T = G.table; tot = sum(c.width for c in T.columns)
def st(c, old, new):
    q = c.text_frame.paragraphs[0]; assert len(q.runs) >= 1 and q.text == old, (q.text, old); q.runs[0].text = new
    for x in q.runs[1:]: x._r.getparent().remove(x._r)
PAIR = {"HOXB8–HP1": "HOXB8–HP1 (CBX1)", "HP1–ZNF143": "HP1 (CBX1)–ZNF143", "HP1–PCGF6": "HP1 (CBX1)–PCGF6", "ADNP–HP1": "ADNP–HP1 (CBX1)"}
CELL = {("EZH2–MTF2", 3): ("AF3 Trimer with SUZ12", "AF3 via SUZ12 in trimer; direct none"), ("MLX–SREBF2", 3): ("AF3 strong 5/5, on E-box DNA", "AF3 strong 5/5; E-box oligo supplied"), ("MLX–SREBF2", 4): ("E-box motif similarity (q = 9 × 10⁻⁷)", "JASPAR E-box similarity (q = 9 × 10⁻⁷)"), ("MEF2A–MEF2D", 4): ("JASPAR both", "JASPAR motif exists (both)"), ("MAZ–ZNF143", 4): ("JASPAR both", "JASPAR motif exists (both)"), ("CTCF–STAG2", 4): ("JASPAR CTCF", "JASPAR motif exists (CTCF)"), ("HOXD10–SREBF2", 4): ("CAP-SELEX call; JASPAR both", "CAP-SELEX call; JASPAR motif exists (both)"), ("FOXK1–MLX", 4): ("CAP-SELEX call; JASPAR both", "CAP-SELEX call (subfamily); JASPAR motif exists (both)")}
done = 0
for r in list(T.rows)[1:]:
    n = r.cells[0].text
    for (a, j), (o, v) in CELL.items():
        if a == n: st(r.cells[j], o, v); done += 1
    if n in PAIR: st(r.cells[0], n, PAIR[n]); done += 1
assert done == len(CELL) + len(PAIR), done
W = [1.76, 1.76, 1.85, 3.2, 4.08]
for c, w in zip(T.columns, W): c.width = Inches(w)
T.columns[4].width = Emu(T.columns[4].width + tot - sum(c.width for c in T.columns))
sub = next(x for x in s.shapes if x.name == "Subtitle"); q = sub.text_frame.paragraphs[0]; assert q.text.startswith("11 known, 2 CAP-SELEX"); q.runs[0].text = q.text.replace("11 known", "11 selected known", 1); q.runs[0].font.size = Pt(14)
for x in q.runs[1:]: x._r.getparent().remove(x._r)
ft = next(x for x in s.shapes if x.name == "Footnote"); q = ft.text_frame.paragraphs[0]; assert "strong before weak" in q.text; q.runs[0].text = q.text.replace("strong before weak", "Boltz strong before weak", 1)
for x in q.runs[1:]: x._r.getparent().remove(x._r)
p.save(sys.argv[2]); print("ok", len(p.slides))
