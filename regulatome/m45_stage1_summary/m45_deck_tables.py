# m45: Stage 1 section update: table slides (native tables) before the summary, counts on the background figure, summary bullet 2. usage: m45_deck_tables.py <in.pptx> <out.pptx>
import sys, zipfile, collections, re
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
IN, OUT = sys.argv[1], sys.argv[2]; R = "/work/users/d/y/dyy12/XuLab/regulatome"; M = f"{R}/m45_stage1_summary"; T = f"{M}/out"; T40 = f"{R}/m40_overlap_reference/out"; PL = f"{M}/plots/slide"; MK = "FIG5S1"
rd = lambda n, d=T: pd.read_csv(f"{d}/{n}.tsv", sep="\t", dtype={"padding": str})
T1, T2, A6, LO, BK, C9, RS = rd("1_recall_pooled"), rd("2_recall_vs_background"), rd("6_shuffle_by_tf_set"), rd("7_leave_one_tf_out_summary"), rd("8_bulk_chip_vs_ours_10kb"), rd("9_counts_per_cutoff"), rd("10_regulome_size")
P2, NEG = rd("perm2_tf_labels_pooled", T40), rd("control_negative_shuffled_data", T40)
CUT = ["|z| >= 2", "|z| >= 3", "|z| >= 4", "|z| >= 5"]; CL = ["|z| ≥ 2", "|z| ≥ 3", "|z| ≥ 4", "|z| ≥ 5"]; PADS = ["0", "10 kb", "50 kb"]
SETS = [("all 63", "All 63 TFs"), ("without NRF1, E4F1, DNMT1", "Without NRF1, E4F1, DNMT1"), ("without broad TFs", "Without the 8 broad TFs"), ("without both", "Without both groups")]
INK, GREY, HEAD = "262626", "595959", "E3E3E3"; CELLS = "HEK293T, top-25% sgRNA cells"
BUL2 = "No padding: ahead of a matched shuffle only at |z| ≥ 4–5, and only because of NRF1, E4F1, DNMT1. Padding 10 kb: ahead at |z| ≥ 3–4 with or without them (no single TF carries it)."
fp = lambda p: "p = 0.001" if p < 0.0015 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"
rng_ = lambda s: "[" + s.replace("-", "–") + "]"
def shuf(r, n=True):
    l = [f"{r['recall observed (%)']:.1f} vs {r['shuffle mean (%)']:.1f} {rng_(r['shuffle 2.5-97.5 (%)'])}", fp(r["p (one-sided)"])]
    if n: l.append(f"{int(r['TFs'])} TFs, {int(r['Perturb-seq genes']):,} genes")
    return (l, r["p (one-sided)"] < 0.05)
p = Presentation(IN); n0 = len(p.slides); lay = {l.name: l for l in p.slide_layouts}
assert not any(sh.name.startswith(f"{MK}-table") for s in p.slides for sh in s.shapes), "tables already present"
def tb(sl, name, x, y, w, h, text, size, bold, color):
    s = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); s.name = name; tf = s.text_frame; tf.word_wrap = True
    for i, t in enumerate(text if isinstance(text, list) else [text]):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); r = para.add_run(); r.text = t; f = r.font; f.name = "Cambria"; f.size = Pt(size); f.bold = bold; f.color.rgb = RGBColor.from_string(color)
    return s
def blank(title, sub):
    sl = p.slides.add_slide(lay["Title and Content"])
    for ph in list(sl.placeholders): ph._element.getparent().remove(ph._element)
    tb(sl, f"{MK}-title", 0.17, 0.12, 12.9, 0.57, title, 28, True, INK); tb(sl, f"{MK}-subtitle", 0.33, 0.68, 12.69, 0.4, sub, 15, False, GREY); return sl
def table(sl, name, rows, x, y, colw, rowh, size=14, hsize=None):
    nr, nc = len(rows), len(rows[0]); gs = sl.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(sum(colw)), Inches(rowh * nr)); gs.name = name; t = gs.table; t.horz_banding = False
    for j, w in enumerate(colw): t.columns[j].width = Inches(w)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(min(rowh, 0.45) if i == 0 else rowh)
        for j, v in enumerate(row):
            lines, bold = v if isinstance(v, tuple) else (v, False)
            if isinstance(lines, str): lines = [lines]
            c = t.cell(i, j); c.fill.solid(); c.fill.fore_color.rgb = RGBColor.from_string(HEAD if i == 0 else "FFFFFF"); c.margin_left = c.margin_right = Inches(0.06); c.margin_top = c.margin_bottom = Inches(0.03)
            tf = c.text_frame; tf.word_wrap = True
            for k, ln in enumerate(lines):
                para = tf.paragraphs[0] if k == 0 else tf.add_paragraph(); r = para.add_run(); r.text = ln; f = r.font; f.name = "Cambria"; f.size = Pt((hsize or size) if i == 0 else (size if k == 0 else max(size - 1, 12)))
                f.bold = (i == 0) or (j == 0) or (bold and k == 0); f.color.rgb = RGBColor.from_string(INK if k == 0 else GREY)
    return gs
new = []
# T1 counts
sl = blank("Regulome- Stage 1 · genes and TFs at each cutoff", f"Perturb-seq genes = TF–gene pairs with |z| ≥ cutoff, own gene removed; {CELLS}")
H = ["Cutoff", "Perturb-seq genes", "TFs with ≥ 1 gene", "TFs with ≥ 10 genes", "Median genes per TF"]
for k, (vn, lab, y) in enumerate((("all 63 TFs", "All 63 TFs", 1.25), ("without broad TFs (55)", "55 TFs (broad TFs removed)", 3.5))):
    rows = [[lab] + H[1:]]
    for c, cl in zip(CUT, CL):
        r = C9[(C9.TFs == vn) & (C9.cutoff == c)].iloc[0]; rows.append([cl, f"{int(r['Perturb-seq genes (TF-gene pairs)']):,}", f"{int(r['TFs with >= 1 gene'])}", f"{int(r['TFs with >= 10 genes'])}", f"{r['median Perturb-seq genes per TF']:g}"])
    table(sl, f"{MK}-table-counts-{k+1}", rows, 0.3, y, [3.3, 2.3, 2.3, 2.3, 2.3], 0.4)
rows = [["Our Regulome (63 TFs)"] + [f"padding {x}" for x in PADS]]
rows.append(["TF–gene links"] + [f"{int(RS[RS.padding == x].iloc[0]['Regulome TF-gene links over the 63 TFs']):,}" for x in PADS])
rows.append(["Median genes per TF"] + [f"{int(RS[RS.padding == x].iloc[0]['median Regulome genes per TF']):,}" for x in PADS])
table(sl, f"{MK}-table-regulome-size", rows, 0.3, 5.75, [3.3, 2.3, 2.3, 2.3], 0.4); new.append(sl)
# T2 recall and background
for vn, lab in (("all 63 TFs", "all 63 TFs"), ("without broad TFs (55)", "55 TFs, broad TFs removed"), ("broad TFs only (8)", "8 broad TFs only")):
    sl = blank(f"Regulome- Stage 1 · recall and background, {lab}", f"Recall = share of Perturb-seq genes found in our Regulome (%); {CELLS}")
    rows = [["Padding", "Cutoff", "Genes in both", "Recall", "Matched background", "All non-changing genes"]]
    for pd_ in PADS:
        for c, cl in zip(CUT, CL):
            a = T1[(T1.padding == pd_) & (T1.TFs == vn) & (T1.cutoff == c)].iloc[0]; b = T2[(T2.padding == pd_) & (T2.TFs == vn) & (T2.cutoff == c)].iloc[0]
            rows.append([pd_ if c == CUT[0] else "", cl, f"{int(a['in both']):,}", f"{a['recall: share of Perturb-seq genes found in our Regulome (%)']:.1f}", f"{b['matched background (%)']:.1f}", f"{b['all non-changing genes (%)']:.1f}"])
    table(sl, f"{MK}-table-recall-{vn.split()[0]}", rows, 0.3, 1.2, [1.6, 1.6, 2.2, 1.9, 2.6, 2.8], 0.43, 13); new.append(sl)
# T3 matched shuffle per padding
for pd_ in PADS:
    sl = blank(f"Regulome- Stage 1 · matched gene shuffle, padding {pd_}", f"Recall observed vs shuffle mean [2.5–97.5], %; 1,000 shuffles within expression × promoter-accessibility deciles; bold: p < 0.05")
    rows = [["TFs"] + CL]
    for s, lab in SETS: rows.append([lab] + [shuf(A6[(A6.padding == pd_) & (A6.cutoff == c) & (A6.set == s)].iloc[0]) for c in CUT])
    table(sl, f"{MK}-table-shuffle-{pd_.replace(' ', '')}", rows, 0.3, 1.25, [3.1, 2.4, 2.4, 2.4, 2.4], 1.12, 14); tb(sl, f"{MK}-note", 0.3, 6.95, 12.5, 0.35, CELLS, 12, False, GREY); new.append(sl)
# T4 leave-one-out + bulk
sl = blank("Regulome- Stage 1 · padding 10 kb: single TFs and bulk ChIP", f"Same matched gene shuffle; {CELLS}; bold: p < 0.05")
rows = [["Leave one TF out (63)", "All 63: gap, p", "Gap over leave-outs", "p over leave-outs", "Largest drop"]]
for _, r in LO.iterrows():
    rows.append([CL[CUT.index(r["cutoff"])], f"+{r['all 63, gap (points)']:.1f} points, {fp(r['all 63, p'])}", f"+{r['leave-one-out gap min']:.1f} to +{r['leave-one-out gap max']:.1f} points", f"{fp(r['leave-one-out p min'])[4:]} to {fp(r['leave-one-out p max'])[4:]}", f"without {r['TF whose removal lowers the gap most']}: +{r['gap without it']:.1f}"])
table(sl, f"{MK}-table-loo", rows, 0.3, 1.2, [3.0, 2.6, 2.6, 1.9, 2.5], 0.42)
rows = [["Same 10 TFs, padding 10 kb"] + CL]
for nm, lab in (("bulk ChIP, 10 TFs pooled", "Bulk ChIP"), ("our HITAG, same 10 TFs pooled", "Our HITAG")): rows.append([lab] + [shuf(BK[(BK.cutoff == c) & (BK.list == nm)].iloc[0]) for c in CUT])
table(sl, f"{MK}-table-bulk", rows, 0.3, 2.75, [3.0, 2.4, 2.4, 2.4, 2.4], 1.05)
tb(sl, f"{MK}-note", 0.3, 5.95, 12.6, 1.3, ["Gap = recall observed minus shuffle mean. No single TF carries the 10 kb result.", "Bulk ChIP lists: CTCF HEK293T (GSE103651); NRF1 K562 (ENCFF259YUE); E4F1 K562 (ENCFF683WRK); NFYA K562; MAZ, ZIC2, ZNF282, ZNF777, ZBTB7A, SETDB1 HEK293."], 13, False, GREY); new.append(sl)
# T5 TF-label shuffle (padding 0) + negative control
sl = blank("Regulome- Stage 1 · TF-label shuffle, padding 0", "Each TF's Perturb-seq genes paired with another TF's list; observed vs shuffle mean [2.5–97.5], %; 1,000 shuffles; bold: p < 0.05")
rows = [["Pairing"] + CL]
for v, tfs, lab in (("all TFs (derangement)", "all TFs", "Any other TF, all 63"), ("all TFs (derangement)", "without NRF1, E4F1, DNMT1", "Any other TF, without the 3"), ("same list-size quintile", "all TFs", "Similar list size, all 63"), ("same list-size quintile", "without NRF1, E4F1, DNMT1", "Similar list size, without the 3")):
    rows.append([lab] + [(lambda r: ([f"{r['share found, observed (%)']:.1f} vs {r['permutation mean (%)']:.1f} {rng_(r['permutation 2.5-97.5 (%)'])}", fp(r["p (one-sided)"]), f"{int(r['n TFs'])} TFs, {int(r['Perturb-seq genes']):,} genes"], r["p (one-sided)"] < 0.05))(P2[(P2.rule == "A") & (P2["Perturb-seq genes cutoff"] == c) & (P2.version == v) & (P2.TFs == tfs)].iloc[0]) for c in CUT])
table(sl, f"{MK}-table-tflabel", rows, 0.3, 1.25, [3.1, 2.4, 2.4, 2.4, 2.4], 1.05)
ng = NEG[NEG.rule == "A"]; tb(sl, f"{MK}-note", 0.3, 6.6, 12.6, 0.7, f"The 3 = NRF1, E4F1, DNMT1. Negative control (one shuffled Perturb-seq matrix): per-TF p median " + ", ".join(f"{x:.2f}" for x in ng["median p"]) + "; " + ", ".join(f"{100*x:.0f}%" for x in ng["share of TFs with p < 0.05"]) + f" of TFs below 0.05. {CELLS}", 12, False, GREY); new.append(sl)
# figure on the background slide: counts under each cutoff
done = set()
for s in p.slides:
    for pic in [sh for sh in s.shapes if sh.name == f"{MK}-S3_recall_vs_background"]:
        l, t_ = pic.left, pic.top; el = pic._element; parent = el.getparent(); idx = list(parent).index(el)
        from PIL import Image; w, h = Image.open(f"{PL}/S3_recall_vs_background.png").size
        nw = s.shapes.add_picture(f"{PL}/S3_recall_vs_background.png", l, t_, width=int(w / 300 * 914400), height=int(h / 300 * 914400)); nw.name = pic.name; parent.remove(el); parent.remove(nw._element); parent.insert(idx, nw._element)
        assert nw.top + nw.height <= p.slide_height; done.add("fig")
    for sh in s.shapes:
        if sh.name == f"{MK}-bullets":
            para = sh.text_frame.paragraphs[1]; para.runs[0].text = "• " + BUL2
            for x in para.runs[1:]: x.text = ""
            done.add("bullet")
assert done == {"fig", "bullet"}, done
bad = [(sh.name, sh.text_frame.text[:80]) for s in new for sh in s.shapes if sh.has_text_frame and re.search(r"expected|fold|chance|precision|validated|functional support|HEK293(?![T ,.;)])", sh.text_frame.text, re.I)]
print("flagged text:", bad); assert not bad
lst = p.slides._sldIdLst; ids = list(lst); k = len(new); summ = next(i for i, s in enumerate(p.slides) if any(sh.name == f"{MK}-bullets" for sh in s.shapes))
for j, el in enumerate(ids[-k:]): lst.remove(el); lst.insert(summ + j, el)
p.part.rename_slide_parts([el.rId for el in lst])
for i, s in enumerate(p.slides, 1):
    if s.has_notes_slide: s.notes_slide.part.partname = s.notes_slide.part.partname.__class__(f"/ppt/notesSlides/notesSlide{i}.xml")
names = [s.part.partname for s in p.slides]; assert len(set(names)) == len(names)
nn = [s.notes_slide.part.partname for s in p.slides if s.has_notes_slide]; assert len(set(nn)) == len(nn)
assert len(p.slides) == n0 + k; p.save(OUT)
q = Presentation(OUT); print("reopened: slides", len(q.slides), "(before", n0, "+", k, ")")
for i in range(summ + k + 2):
    s = q.slides[i]; print(i + 1, [sh.text_frame.text[:70] for sh in s.shapes if sh.has_text_frame][:1])
z = zipfile.ZipFile(OUT); c = collections.Counter(z.namelist()); dup = [x for x, v in c.items() if v > 1]; print("zip entries", len(c), "duplicates", dup); assert not dup
