"""Shared-motif slide for MLX-SREBF2, EZH2-MTF2 motif slide removed, bar map and PAE back on the MLX-SREBF2 structure slide, conclusion lines, table cell and footnote numbers. usage: edit_motif_round.py in.pptx out.pptx"""
import sys
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
p = Presentation(sys.argv[1]); slides = list(p.slides); ttl = lambda s: next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()), ""); find = lambda st: next(s for s in slides if ttl(s).startswith(st))
def tb(s, x, y, w, h, paras, size=14, name="Conclusion", color="1A1A1A"):
    t = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); t.name = name; f = t.text_frame; f.word_wrap = True
    for i, (b, v) in enumerate(paras):
        q = f.paragraphs[0] if i == 0 else f.add_paragraph(); q.space_after = Pt(4)
        for tx, bold in (((b, True), (v, False)) if b else ((v, False),)):
            r = q.add_run(); r.text = tx; r.font.size = Pt(size); r.font.name = "Cambria"; r.font.bold = bold; r.font.color.rgb = RGBColor.from_string(color)
    return t
def pic(s, f, x, y, w, h):
    px = Image.open(f).size; assert abs(px[0] / w - 600) < 2 and abs(px[1] / h - 600) < 2, (f, px); return s.shapes.add_picture(f, Inches(x), Inches(y), Inches(w), Inches(h))
def strip(s):
    for sh in [x for x in s.shapes if x.name not in ("Title", "Subtitle")]: sh._element.getparent().remove(sh._element)
def sub(s, v):
    r = next(x for x in s.shapes if x.name == "Subtitle").text_frame.paragraphs[0].runs; r[0].text = v
    for x in r[1:]: x.text = ""
C = "Conclusion: "; s2, s3, s4, s5, s6, s7 = find("PRC2 core EZH2–MTF2- layer 1"), find("PRC2 core EZH2–MTF2- layer 2"), find("PRC2 core EZH2–MTF2- layer 3"), find("Novel MLX–SREBF2 – layer 1"), find("Novel MLX–SREBF2- layer 2"), find("Novel MLX–SREBF2- layer 3")
tb(s2, 0.3, 6.82, 12.1, 0.42, [(C, "AF3 links EZH2 and MTF2 only through SUZ12, with no direct interface.")])
tb(s3, 7.25, 6.42, 5.4, 0.8, [(C, "EZH2–MTF2 stays high after accessibility correction (percentile 0.93–0.96).")])
n3 = s3.notes_slide.notes_text_frame; n3.text = "\n".join(x for x in (n3.text, "Motif, not shown: neither EZH2 nor MTF2 has a JASPAR motif, so a shared motif cannot be tested. MTF2's EH domain binds unmethylated CpG (Li et al. 2017). Co-bound vs GC- and ATAC-matched elements: CpG o/e ratio 0.99 (95% CI 0.97–1.01), within 40 control pairs.") if x)
strip(s5); pic(s5, "super/af3_MLX_merged2.png", 0.2, 1.05, 12.9, 3.45); pic(s5, "fig/pae_MLX_v2.png", 0.25, 4.55, 2.3, 1.95); pic(s5, "fig/sitemap_MLX_v2.png", 6.0, 4.55, 7.1, 1.95)
tb(s5, 2.7, 4.65, 3.2, 1.85, [("AlphaFold 3", ""), ("", "ipTM 0.45"), ("", "ipSAE 0.57"), ("", "5 of 5 seeds strong")], name="Scores"); tb(s5, 0.3, 6.55, 12.1, 0.42, [(C, "AF3 models MLX–SREBF2 as a strong bHLH-ZIP dimer: contacts on the HLH-ZIP, basic regions free, no DNA in the model. Black ticks = interface.")], size=13)
tb(s6, 7.25, 6.42, 5.4, 0.8, [(C, "High on corrected enhancer and Polycomb (0.78, 0.81); the drop to 0.44 on repressive shows part of the signal was accessibility.")], size=13)
old = s7.notes_slide.notes_text_frame.text if s7.has_notes_slide else ""; strip(s7); sub(s7, "MLX and SREBF2 recognise nearly the same E-box; SREBF2's SRE shares only a half-site with it"); pic(s7, "fig/motif_shared.png", 0.25, 1.1, 8.4, 5.35)
tb(s7, 8.95, 1.75, 4.15, 3.2, [("Tomtom, JASPAR 2024 CORE", ""), ("MLX vs SREBF2 E-box: ", "q = 9 × 10⁻⁷ (p = 2 × 10⁻⁹), offset 0, same strand"), ("MLX vs SREBF2 SRE: ", "q = 0.06 (p = 0.003), offset 0, reverse strand")], name="Summary")
tb(s7, 0.3, 6.5, 12.1, 0.5, [(C, "MLX and SREBF2 share the CACGTG E-box motif; the SRE matches MLX only at ATCAC, which is not significant.")])
s7.notes_slide.notes_text_frame.text = "\n".join(x for x in (old, "Tomtom 5.5.9, Pearson distance, min overlap 5, against all 879 motifs in the JASPAR 2024 CORE vertebrate file. MA0663.1 vs MA0828.3: p 1.66e-9, q 9.08e-7, overlap 10, offset 0, same strand. MA0663.1 vs MA0596.1: p 0.00289, q 0.0579, overlap 10, offset 0, reverse strand. Not shown: CAP-SELEX tested the pair and did not call it. Composite co-binding is not detectable at this depth (ratio 0.80, 95% CI 0.37–1.21, 15 elements at p ≤ 1e-5; 0.99, 0.85–1.17, 153 elements at p ≤ 1e-4). RBBP7–SRCAP at the same 15 elements gives 1.50 (1.07–1.99), so that small set is not reliable.") if x)
Ls = p.slides._sldIdLst; i4 = slides.index(s4); e = list(Ls)[i4]; Ls.remove(e); p.part.drop_rel(e.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")); new = list(p.slides); m = next(k for k, x in enumerate(new, 1) if ttl(x).startswith("Cobinding Analysis")); h = next(k for k, x in enumerate(new, 1) if ttl(x).startswith("4E-H12")); hit = 0
for sh in new[0].shapes:
    if sh.has_text_frame:
        for par in sh.text_frame.paragraphs:
            for r in par.runs:
                if "(slide 182)" in r.text: r.text = r.text.replace("(slide 182)", f"(slide {m})"); hit += 1
                if "triangle: slide 26" in r.text: r.text = r.text.replace("triangle: slide 26", f"triangle: slide {h}"); hit += 1
    if sh.has_table:
        for row in sh.table.rows:
            if row.cells[0].text.strip() == "EZH2–MTF2":
                c = list(row.cells)[-1]; assert c.text.strip() == "MTF2 CpG (literature)", c.text; rr = c.text_frame.paragraphs[0].runs; rr[0].text = "n/a (EZH2 has no DNA motif)"; hit += 1
                for x in rr[1:]: x.text = ""
assert hit == 3, hit; n_drop = 0
for x in new[:7]:
    xml = x.part._element.xml
    for rid, rel in list(x.part.rels.items()):
        if rel.reltype.endswith("/image") and f'"{rid}"' not in xml: x.part.rels.pop(rid); n_drop += 1
print("unused images dropped", n_drop, "| matrix", m, "Hi-C", h); p.save(sys.argv[2]); print("ok", len(new))
