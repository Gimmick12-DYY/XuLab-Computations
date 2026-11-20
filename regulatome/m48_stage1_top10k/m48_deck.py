# m48: new front section "11/20/26 Regulome Stage 1: top 10,000 peaks" in the local deck (figures + native tables from m45/m46/m47/m48 result files);
# also corrects the bulk ChIP footnote on the 11/18 slides (bulk lists = each file's top 5,000 peaks). usage: m48_deck.py <in.pptx> <out.pptx>
import sys, re, zipfile, collections
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
IN, OUT = sys.argv[1], sys.argv[2]; R = "/work/users/d/y/dyy12/XuLab/regulatome"; MK = "FIG5S1B"
A45, A46, A47, A48 = f"{R}/m45_stage1_summary/out", f"{R}/m46_stage1_imputed/out", f"{R}/m47_stage1_peak_set_size/out", f"{R}/m48_stage1_top10k/out"; PL = f"{R}/m48_stage1_top10k/plots/slide"
rd = lambda d, n: pd.read_csv(f"{d}/{n}.tsv", sep="\t", dtype={"padding": str})
INK, GREY, HEAD = "262626", "595959", "E3E3E3"; CUT = ["|z| >= 2", "|z| >= 3", "|z| >= 4", "|z| >= 5"]; CL = ["|z| ≥ 2", "|z| ≥ 3", "|z| ≥ 4", "|z| ≥ 5"]
fp = lambda p: "p 0.001" if p < 0.0015 else (f"p {p:.3f}" if p < 0.1 else f"p {p:.2f}")
sg = lambda x: f"{x:+.1f}"
p = Presentation(IN); n0 = len(p.slides); lay = {l.name: l for l in p.slide_layouts}
assert not any(sh.name.startswith(MK + "-") for s in p.slides for sh in s.shapes), "section already present"
def tb(sl, name, x, y, w, h, text, size, bold, color):
    s = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); s.name = name; tf = s.text_frame; tf.word_wrap = True
    for i, t in enumerate(text if isinstance(text, list) else [text]):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); r = para.add_run(); r.text = t; f = r.font; f.name = "Cambria"; f.size = Pt(size); f.bold = bold; f.color.rgb = RGBColor.from_string(color)
        if isinstance(text, list): para.space_after = Pt(10)
    return s
def blank(title, sub):
    sl = p.slides.add_slide(lay["Title and Content"])
    for ph in list(sl.placeholders): ph._element.getparent().remove(ph._element)
    tb(sl, f"{MK}-title", 0.17, 0.12, 12.9, 0.57, title, 28, True, INK); tb(sl, f"{MK}-subtitle", 0.33, 0.68, 12.69, 0.4, sub, 15, False, GREY); return sl
def table(sl, name, rows, x, y, colw, rowh, size=13, hh=None):
    nr, nc = len(rows), len(rows[0]); gs = sl.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(sum(colw)), Inches(rowh * nr)); gs.name = name; t = gs.table; t.horz_banding = False
    for j, w in enumerate(colw): t.columns[j].width = Inches(w)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches((hh or rowh) if i == 0 else rowh)
        for j, v in enumerate(row):
            lines = v if isinstance(v, list) else [str(v)]; c = t.cell(i, j); c.fill.solid(); c.fill.fore_color.rgb = RGBColor.from_string(HEAD if i == 0 else "FFFFFF")
            c.margin_left = c.margin_right = Inches(0.05); c.margin_top = c.margin_bottom = Inches(0.02); tf = c.text_frame; tf.word_wrap = True
            for k, ln in enumerate(lines):
                para = tf.paragraphs[0] if k == 0 else tf.add_paragraph(); r = para.add_run(); r.text = ln; f = r.font; f.name = "Cambria"; f.size = Pt(size if k == 0 else max(size - 1, 12))
                f.bold = i == 0 or j == 0; f.color.rgb = RGBColor.from_string(INK if k == 0 else GREY)
    return gs
def figure(sl, name):
    im = Image.open(f"{PL}/{name}.png"); w, h = im.size
    pic = sl.shapes.add_picture(f"{PL}/{name}.png", Inches(0.2), Inches(1.15), width=Emu(int(w / 300 * 914400)), height=Emu(int(h / 300 * 914400))); pic.name = f"{MK}-{name}"
    assert pic.top + pic.height <= p.slide_height; return pic
def note(sl, text, y=6.85): tb(sl, f"{MK}-note", 0.35, y, 12.6, 0.5, text, 13, False, GREY)
CELLS = "HEK293T, top-25% sgRNA cells"; new = []
# 1 divider
d = p.slides.add_slide(lay["Title Slide"]); d.shapes.title.text = "11/20/26 Regulome Stage 1: top 10,000 peaks"; d.shapes.title.name = f"{MK}-divider"
for ph in d.placeholders:
    if ph.placeholder_format.idx == 1: ph.name = f"{MK}-subtitle"; ph.text = "Top 10,000 lab peaks per TF, 10 kb padding; recall against matched background"
new.append(d)
# 2 schematic
sl = blank("Regulome- Stage 1 · setup (top 10,000 peaks)", f"Two dials: the Perturb-seq cutoff and the padding around the gene start; 63 TFs, {CELLS}"); figure(sl, "S1_schematic"); new.append(sl)
# 3 peak set choice (m47)
T47 = rd(A47, "1_recall_and_shuffle"); S47 = rd(A47, "2_list_sizes")
sl = blank("Regulome- Stage 1 · how many peaks per TF", "Recall = share of Perturb-seq genes found in our Regulome; gap to matched gene shuffle (points); 63 TFs, 10 kb")
rows = [["Peaks per TF", "Median genes per TF"] + CL]
for ps in ["top 2,000", "top 5,000", "top 10,000", "top 20,000", "top 50,000", "all q < 0.05", "full peak set"]:
    med = S47[(S47["peak set"] == ps) & (S47.padding == "10 kb") & (S47.TFs == "all 63 TFs")]["linked genes per TF, median"].iloc[0]
    r = [ps, f"{med:,.0f}"]
    for c in CUT:
        x = T47[(T47["peak set"] == ps) & (T47.padding == "10 kb") & (T47.cutoff == c) & (T47.TFs == "all 63 TFs")].iloc[0]; r.append([f"{x['recall (%)']:.1f}% | gap {sg(x['gap (points)'])}", fp(x["p (one-sided)"])])
    rows.append(r)
table(sl, f"{MK}-table-peakset", rows, 0.3, 1.25, [2.2, 1.9, 2.15, 2.15, 2.15, 2.15], 0.62, 13, 0.42); note(sl, "q < 0.05 and the full peak set lose the gap; top 5,000 to 20,000 keep it. Top 10,000 chosen.", 6.45); new.append(sl)
# 4-7 figures
for f, t_, s_ in (("S2_recall_by_cutoff_and_padding", "recall by cutoff and padding", f"Recall = share of Perturb-seq genes found in our Regulome; 63 TFs, top 10,000 peaks, {CELLS}"),
                  ("S3_recall_vs_background", "recall against background", "63 TFs, top 10,000; matched background = same TF's non-changing genes, matched on expression and promoter ATAC"),
                  ("S4_dose_response", "share bound by response strength", f"55 TFs vs the 8 broad TFs with a knockdown; top 10,000 peaks; {CELLS}"),
                  ("S5_broad_tfs", "broad TFs", f"All 63 TFs, 55 without broad TFs, 8 broad TFs only; top 10,000 peaks; {CELLS}")):
    sl = blank(f"Regulome- Stage 1 · {t_}", s_); figure(sl, f); new.append(sl)
# 8 counts and Regulome size
C9 = rd(A45, "9_counts_per_cutoff"); P1 = rd(A48, "1_recall_pooled"); V4 = rd(A48, "regulome_v4_summary").iloc[0]
sl = blank("Regulome- Stage 1 · genes and TFs at each cutoff", f"Perturb-seq genes = TF–gene pairs with |z| ≥ cutoff, own gene removed; {CELLS}")
rows = [["Cutoff", "Perturb-seq genes (63 TFs)", "TFs with ≥ 1 gene", "TFs with ≥ 10 genes", "Median genes per TF"]]
for c, cl in zip(CUT, CL):
    r = C9[(C9.TFs == "all 63 TFs") & (C9.cutoff == c)].iloc[0]; rows.append([cl, f"{int(r['Perturb-seq genes (TF-gene pairs)']):,}", f"{int(r['TFs with >= 1 gene'])}", f"{int(r['TFs with >= 10 genes'])}", f"{r['median Perturb-seq genes per TF']:g}"])
table(sl, f"{MK}-table-counts", rows, 0.3, 1.25, [2.2, 3.0, 2.4, 2.4, 2.4], 0.4, 14)
rows = [["Our Regulome, top 10,000 (63 TFs)", "padding 0", "padding 10 kb", "padding 50 kb"]]
g = P1[(P1.TFs == "all 63 TFs") & (P1.cutoff == "|z| >= 2")].set_index("padding")
rows.append(["TF–gene links"] + [f"{int(g.loc[x, 'our genes']):,}" for x in ("0", "10 kb", "50 kb")]); rows.append(["Median genes per TF"] + [f"{g.loc[x, 'median our genes per TF']:,.0f}" for x in ("0", "10 kb", "50 kb")])
table(sl, f"{MK}-table-size", rows, 0.3, 3.65, [4.0, 2.5, 2.5, 2.5], 0.42, 14)
note(sl, f"Regulome v4 (top 10,000, 10 kb, 79 TFs): {int(V4['links']):,} links, {int(V4['genes']):,} genes, median {V4['median genes per TF']:,.0f} genes per TF and {V4['median TFs per gene']:.0f} TFs per gene.", 5.2); new.append(sl)
# 9 recall + matched background, top 5k vs 10k
B5, B10 = rd(A45, "2_recall_vs_background"), rd(A48, "2_recall_vs_background")
sl = blank("Regulome- Stage 1 · recall vs matched background, top 5k vs 10k", f"Recall vs the matched background (%), padding 10 kb; {CELLS}")
rows = [["TFs", "Cutoff", "Top 5,000: recall", "Top 5,000: matched background", "Top 10,000: recall", "Top 10,000: matched background"]]
for grp, lab in (("all 63 TFs", "All 63"), ("without broad TFs (55)", "55 non-broad"), ("broad TFs only (8)", "8 broad")):
    for c, cl in zip(CUT, CL):
        a = B5[(B5.padding == "10 kb") & (B5.TFs == grp) & (B5.cutoff == c)].iloc[0]; b = B10[(B10.padding == "10 kb") & (B10.TFs == grp) & (B10.cutoff == c)].iloc[0]
        rows.append([lab if c == CUT[0] else "", cl, f"{a['recall, changed genes (%)']:.1f}", f"{a['matched background (%)']:.1f}", f"{b['recall, changed genes (%)']:.1f}", f"{b['matched background (%)']:.1f}"])
table(sl, f"{MK}-table-background", rows, 0.3, 1.2, [1.7, 1.3, 2.1, 2.7, 2.1, 2.7], 0.39, 13, 0.45); new.append(sl)
# 10 shuffle by TF set, 10 kb (top 10k with top 5k line)
S5, S10 = rd(A45, "6_shuffle_by_tf_set"), rd(A48, "6_shuffle_by_tf_set")
SETS = [("all 63", "All 63 TFs"), ("without NRF1, E4F1, DNMT1", "Without NRF1, E4F1, DNMT1"), ("without broad TFs", "Without the 8 broad TFs"), ("without both", "Without both groups")]
def cellsh(T, pad, c, s_): x = T[(T.padding == pad) & (T.cutoff == c) & (T.set == s_)].iloc[0]; return x
sl = blank("Regulome- Stage 1 · matched gene shuffle by TF set, padding 10 kb", "Gap = recall minus shuffle mean (points); 1,000 shuffles within expression × promoter-accessibility deciles; grey line = top 5,000")
rows = [["TFs"] + CL]
for s_, lab in SETS:
    r = [lab]
    for c in CUT:
        b = cellsh(S10, "10 kb", c, s_); a = cellsh(S5, "10 kb", c, s_)
        r.append([f"{b['recall observed (%)']:.1f}% | gap {sg(b['observed minus shuffle mean (points)'])} | {fp(b['p (one-sided)'])}", f"top 5,000: gap {sg(a['observed minus shuffle mean (points)'])} | {fp(a['p (one-sided)'])}"])
    rows.append(r)
table(sl, f"{MK}-table-shuffle10", rows, 0.3, 1.25, [2.9, 2.45, 2.45, 2.45, 2.45], 0.95, 13, 0.42); note(sl, f"Top 10,000; n TFs: 63 / 60 / 55 / 52; {CELLS}"); new.append(sl)
sl = blank("Regulome- Stage 1 · matched gene shuffle, padding 0 and 50 kb", "Top 10,000; gap = recall minus shuffle mean (points); 1,000 shuffles")
for k, pad in enumerate(("0", "50 kb")):
    rows = [[f"Padding {pad}"] + CL]
    for s_, lab in SETS:
        rows.append([lab] + [(lambda b: f"{b['recall observed (%)']:.1f}% | {sg(b['observed minus shuffle mean (points)'])} | {fp(b['p (one-sided)'])}")(cellsh(S10, pad, c, s_)) for c in CUT])
    table(sl, f"{MK}-table-shuffle-{pad.replace(' ', '')}", rows, 0.3, 1.2 + k * 2.85, [2.9, 2.45, 2.45, 2.45, 2.45], 0.5, 13)
new.append(sl)
# 11 leave-one-out + TF-label shuffle
LO = rd(A48, "7_leave_one_tf_out_summary"); TL = rd(A48, "8_tf_label_shuffle")
sl = blank("Regulome- Stage 1 · single TFs and TF labels, padding 10 kb", f"Top 10,000; 63 TFs; {CELLS}")
rows = [["Leave one TF out (63)", "All 63: gap, p", "Gap over leave-outs", "p over leave-outs", "Largest drop"]]
for _, r in LO.iterrows():
    rows.append([CL[CUT.index(r["cutoff"])], f"{sg(r['all 63, gap (points)'])} points, {fp(r['all 63, p'])}", f"{sg(r['leave-one-out gap min'])} to {sg(r['leave-one-out gap max'])} points", f"{fp(r['leave-one-out p min'])[2:]} to {fp(r['leave-one-out p max'])[2:]}", f"without {r['TF whose removal lowers the gap most']}: {sg(r['gap without it'])}"])
table(sl, f"{MK}-table-loo", rows, 0.3, 1.2, [3.0, 2.6, 2.6, 2.0, 2.4], 0.42, 13)
rows = [["TF-label shuffle (own list vs another TF's list)"] + CL]
for pr, lab in (("any other TF (derangement)", "Any other TF"), ("same list-size quintile", "TF of similar list size")):
    for s_, sl_ in (("all TFs", "all 63"), ("without NRF1, E4F1, DNMT1", "without the 3")):
        rows.append([f"{lab}, {sl_}"] + [(lambda x: f"{x['recall own lists (%)']:.1f} vs {x['shuffle mean (%)']:.1f} | {fp(x['p (one-sided)'])}")(TL[(TL.padding == "10 kb") & (TL.cutoff == c) & (TL.pairing == pr) & (TL.TFs == s_)].iloc[0]) for c in CUT])
table(sl, f"{MK}-table-tflabel", rows, 0.3, 2.95, [3.4, 2.3, 2.3, 2.3, 2.3], 0.5, 13, 0.55); note(sl, "The 3 = NRF1, E4F1, DNMT1. Own list vs another TF's list: not significant (TF specificity not shown at this stage).", 6.1); new.append(sl)
# 12 share of our genes found in Perturb-seq, observed vs shuffle
PR = rd(A47, "3_precision")
sl = blank("Regulome- Stage 1 · our genes found in Perturb-seq, observed vs shuffle", f"Share of our genes found in Perturb-seq (%): observed vs matched shuffle | observed / shuffle | p; 63 TFs; {CELLS}")
rows = [["Peaks, padding"] + CL]
for ps in ("top 5,000", "top 10,000"):
    for pad in ("0", "10 kb", "50 kb"):
        r = [f"{ps}, padding {pad}"]
        for c in CUT:
            x = PR[(PR["peak set"] == ps) & (PR.padding == pad) & (PR.cutoff == c) & (PR.TFs == "all 63 TFs")].iloc[0]; r.append([f"{x['precision observed (%)']:.2f} vs {x['precision shuffle mean (%)']:.2f}", f"{x['observed / shuffle']:.2f}x | {fp(x['p (one-sided, same for both)'])}"])
        rows.append(r)
table(sl, f"{MK}-table-ourgenes", rows, 0.3, 1.25, [2.9, 2.45, 2.45, 2.45, 2.45], 0.72, 13, 0.42); new.append(sl)
# 13 bulk panel
sl = blank("Regulome- Stage 1 · bulk ChIP and our data, per TF", f"Top 10,000 peaks each, padding 10 kb, |z| ≥ 3; recall and observed / matched shuffle; {CELLS}"); figure(sl, "S6_bulk_chip_vs_ours_per_tf"); new.append(sl)
# 14 bulk side table per TF (two slides)
BS = rd(A48, "9_bulk_chip_side_table"); BS = BS[BS.padding == "10 kb"]
BNOTE = "Bulk: CTCF HEK293T (GSE103651); ZNF282 HEK293T (GSM2466515; 8,514 peaks, all used); MAZ, ZIC2, ZNF777 (6,223 peaks, all used), SETDB1 HEK293 (SETDB1 from file name). ZBTB7A removed; no K562 or HepG2."
for cuts, lab in ((["|z| ≥ 2", "|z| ≥ 3"], "|z| ≥ 2–3"), (["|z| ≥ 4", "|z| ≥ 5"], "|z| ≥ 4–5")):
    sl = blank(f"Regulome- Stage 1 · bulk ChIP vs ours per TF, top 10,000, {lab}", "Padding 10 kb; recall | observed / shuffle | p per list; paired p = ours vs bulk on one shared shuffle")
    rows = [["TF (bulk cell line)", "Cutoff", "Perturb-seq genes", "Bulk ChIP", "Our HITAG", "Ours − bulk (ratio), paired p"]]
    for t in ["CTCF", "ZNF282", "MAZ", "ZIC2", "ZNF777", "SETDB1"]:
        for c in cuts:
            x = BS[(BS.TF == t) & (BS.cutoff == c)].iloc[0]
            if str(x["bulk: recall (%)"]) == "n < 10": rows.append([f"{t} ({x['bulk ChIP cell line'].split(' ')[0]})" if c == cuts[0] else "", c, f"{int(x['Perturb-seq genes'])}", "n < 10", "n < 10", "n < 10"]); continue
            f_ = lambda k: f"{float(x[k + ': recall (%)']):.1f}% | {float(x[k + ': observed / shuffle']):.2f}x | {fp(float(x[k + ': p']))}"
            rows.append([f"{t} ({x['bulk ChIP cell line'].split(' ')[0]})" if c == cuts[0] else "", c, f"{int(x['Perturb-seq genes']):,}", f_("bulk"), f_("ours"), f"{float(x['ratio difference (ours - bulk)']):+.2f} | {fp(float(x['paired p (two-sided)']))}"])
    table(sl, f"{MK}-table-bulk-{cuts[0][-1]}", rows, 0.3, 1.2, [2.3, 1.2, 1.5, 2.8, 2.8, 2.15], 0.38, 13, 0.45); note(sl, BNOTE, 6.4); new.append(sl)
# 15 coordinated binding
CB = rd(A47, "6_coordinated_binding"); AD = rd(A47, "7_coordinated_binding_added_links")
sl = blank("Regulome- Stage 1 · adding coordinated-binding links", f"Padding 10 kb, 63 TFs; recall | gap to matched shuffle (points) | p; {CELLS}")
rows = [["List"] + CL]
for ps in ("top 5,000", "top 10,000", "top 20,000"):
    for ver, lab in (("base", ps), ("base + coordinated binding", f"{ps} + coordinated binding")):
        rows.append([lab] + [(lambda x: f"{x['recall (%)']:.1f}% | {sg(x['gap (points)'])} | {fp(x['p (one-sided)'])}")(CB[(CB["peak set"] == ps) & (CB.padding == "10 kb") & (CB.links == ver) & (CB.cutoff == c) & (CB.TFs == "all 63 TFs")].iloc[0]) for c in CUT])
table(sl, f"{MK}-table-coord", rows, 0.3, 1.2, [3.4, 2.3, 2.3, 2.3, 2.3], 0.45, 13)
rows = [["Added links alone: share responding (added vs base)"] + CL]
for ps in ("top 5,000", "top 10,000", "top 20,000"):
    rows.append([f"{ps} ({int(AD[(AD['peak set'] == ps) & (AD.padding == '10 kb') & (AD.TFs == 'all 63 TFs')]['added links'].iloc[0]):,} added)"] + [(lambda x: f"{x['share of added links responding (%)']:.2f} vs {x['share of base links responding (%)']:.2f} %")(AD[(AD["peak set"] == ps) & (AD.padding == "10 kb") & (AD.cutoff == c) & (AD.TFs == "all 63 TFs")].iloc[0]) for c in CUT])
table(sl, f"{MK}-table-coord-added", rows, 0.3, 4.5, [3.4, 2.3, 2.3, 2.3, 2.3], 0.45, 13, 0.55); note(sl, "Coordinated-binding links add 1–2 points of recall; the links they add respond at about the rate of the base list or lower.", 6.6); new.append(sl)
# 16 imputed MACS in Stage 1
I0, I5 = rd(A46, "14_masks_macs_pooled"), rd(A46, "14_masks_macs_pooled_drop5"); IM = pd.concat([I0, I5], ignore_index=True)
sl = blank("Regulome- Stage 1 · imputed data: MACS3 peaks per ATAC mask", "62 TFs; lists cut to the lab list size; gap to matched shuffle (points) | p; lab = top 5,000 lab peaks (raw data)")
LAB = [("lab peaks", "top 5,000", "Lab peaks, top 5,000 (raw)"), ("P3b [Mpeak]", "equal size", "Imputed, current mask (ATAC peaks)"), ("P3b [Mdrop5]", "equal size", "Imputed, September mask (ATAC CPM > 0.078)"), ("P3b [M0]", "equal size", "Imputed, no mask")]
for k, pad in enumerate(("10 kb", "50 kb")):
    rows = [[f"Padding {pad}"] + CL]
    for l_, v_, lab in LAB:
        g = IM[(IM.list == l_) & (IM.version == v_) & (IM.padding == pad) & (IM.TFs == "all")].drop_duplicates(["cutoff"])
        rows.append([lab] + [(lambda x: f"{sg(x['observed minus shuffle mean (points)'])} | {fp(x['p (one-sided)'])}")(g[g.cutoff == c].iloc[0]) for c in CUT])
    table(sl, f"{MK}-table-imputed-{pad.replace(' ', '')}", rows, 0.3, 1.2 + k * 2.6, [4.1, 2.2, 2.2, 2.2, 2.2], 0.44, 13)
note(sl, "Imputed peaks: MACS3 callpeak on aggregated imputed calls with HEK293T ATAC as control. Lab peaks (raw data) stay the Stage 1 input.", 6.45); new.append(sl)
# 17 imputed vs bulk benchmark
BB = rd(A46, "18_bulk_benchmark_masks")
sl = blank("Imputed data vs bulk ChIP per ATAC mask (all-universe benchmark)", "AUROC | Bin.neg % | Peak.pos %; positives = TF-bound cCRE bins, negatives = no-cCRE bins with zero bulk reads (genome-wide)")
rows = [["TF (bulk reference)", "Raw", "Imputed, current mask", "Imputed, September mask", "Imputed, no mask"]]
for tf, ps, lab in (("CTCF", "CTCF consensus (maj 2/3), zero-read negatives", "CTCF consensus (HEK293 + HEK293T)"), ("CTCF", "all_universe", "CTCF (HEK293)"), ("ZNF282", "all_universe", "ZNF282 (HEK293T)"), ("MAZ", "all_universe", "MAZ (HEK293)"), ("ZIC2", "all_universe", "ZIC2 (HEK293)"), ("ZNF777", "all_universe", "ZNF777 (HEK293)"), ("ZBTB7A", "all_universe", "ZBTB7A (HEK293)")):
    g = BB[(BB.TF == tf) & (BB["positive set"] == ps)].set_index("Pipeline")
    rows.append([lab] + [(lambda x: f"{x['AUROC']:.3f} | {x['Bin.neg %']:.1f} | {x['Peak.pos %']:.1f}")(g.loc[k]) for k in ("raw", "unified (Mpeak, current)", "Mdrop5", "M0")])
table(sl, f"{MK}-table-bulkbench", rows, 0.3, 1.2, [3.6, 2.2, 2.3, 2.4, 2.2], 0.5, 13); note(sl, "NFYA left out (K562 reference). September mask = imputed calls kept only where mean ATAC CPM > 0.078; regions below it are low-accessibility.", 5.4); new.append(sl)
# 18 summary
b = B10[(B10.padding == "10 kb") & (B10.TFs == "all 63 TFs")].set_index("cutoff").loc[CUT]
sx = S10[(S10.padding == "10 kb") & (S10.set == "without NRF1, E4F1, DNMT1")].set_index("cutoff").loc[CUT]
BUL = [f"Setting: top 10,000 lab peaks per TF + 10 kb padding. Recall {' / '.join(f'{v:.1f}' for v in b['recall, changed genes (%)'])}% vs matched background {' / '.join(f'{v:.1f}' for v in b['matched background (%)'])}% (|z| ≥ 2–5)",
       f"Above the matched shuffle at |z| ≥ 3–4 with and without NRF1, E4F1, DNMT1 (without: {sg(sx.loc['|z| >= 3', 'observed minus shuffle mean (points)'])} / {sg(sx.loc['|z| >= 4', 'observed minus shuffle mean (points)'])} points, {fp(sx.loc['|z| >= 4', 'p (one-sided)'])}), and with any single TF left out",
       "TF specificity not shown at this stage: a TF's own list does no better than another TF's list",
       "On a par with HEK bulk ChIP at equal peak number: ours better for ZNF282 at |z| ≥ 4; bulk better for CTCF and SETDB1",
       "Next: Stage 2 adds parallel-profiling information for recall; Stage 3 a model to rank candidate links"]
sl = blank("Regulome- Stage 1 · summary (top 10,000 peaks)", f"63 TFs, {CELLS}"); tb(sl, f"{MK}-bullets", 0.6, 1.4, 12.1, 5.6, ["• " + x for x in BUL], 19, False, INK); new.append(sl)
bad = [(sh.name, sh.text_frame.text[:80]) for s in new for sh in s.shapes if sh.has_text_frame and re.search(r"validated|expected|fold\b|closed|precision|chance", sh.text_frame.text, re.I)]
bad += [(sh.name, "table") for s in new for sh in s.shapes if sh.has_table and any(re.search(r"validated|expected|fold\b|closed|precision|chance", c.text, re.I) for row in sh.table.rows for c in row.cells)]
print("flagged:", bad); assert not bad
# 11/18 section: bulk ChIP footnote correction (bulk lists = each file's top 5,000 peaks)
fixed = 0
for s in list(p.slides)[:n0]:
    if any(sh.name.startswith("FIG5S1-table-bulktf") for sh in s.shapes):
        for sh in s.shapes:
            if sh.name == "FIG5S1-note" and sh.has_text_frame and "top 5,000" not in sh.text_frame.text:
                r = sh.text_frame.paragraphs[0].runs[0]; r.text = "Bulk ChIP lists = each file's top 5,000 peaks by signal. " + r.text; fixed += 1
print("11/18 bulk footnotes corrected:", fixed)
lst = p.slides._sldIdLst; ids = list(lst); k = len(new)
for el in ids[-k:][::-1]: lst.remove(el); lst.insert(0, el)
p.part.rename_slide_parts([el.rId for el in lst])
for i, s in enumerate(p.slides, 1):
    if s.has_notes_slide: s.notes_slide.part.partname = s.notes_slide.part.partname.__class__(f"/ppt/notesSlides/notesSlide{i}.xml")
names = [s.part.partname for s in p.slides]; assert len(set(names)) == len(names)
nn = [s.notes_slide.part.partname for s in p.slides if s.has_notes_slide]; assert len(set(nn)) == len(nn)
assert len(p.slides) == n0 + k; p.save(OUT)
q = Presentation(OUT); print("reopened: slides", len(q.slides), "(before", n0, "+", k, ")")
for i in range(k + 2): print(i + 1, [sh.text_frame.text[:70] for sh in q.slides[i].shapes if sh.has_text_frame][:1])
z = zipfile.ZipFile(OUT); c = collections.Counter(z.namelist()); dup = [x for x, v in c.items() if v > 1]; print("zip entries", len(c), "duplicates", dup); assert not dup
