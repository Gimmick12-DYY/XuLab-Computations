"""Rebuild the '11/2/26 Jie requests' section in place: keeps the section title, replaces everything up to the next section. Figures go in at their exact render size (fig/frames.json).
usage: python build_jie_v2.py in.pptx out.pptx [prc2]"""
import json, sys
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR
Image.MAX_IMAGE_PIXELS = None
IN, OUT = sys.argv[1], sys.argv[2]; PRC2 = "prc2" in sys.argv[3:]; INK = "1A1A1A"; GR = "4D4D4D"; FR = json.load(open("fig/frames.json"))
p = Presentation(IN); LAY = {l.name: l for l in p.slide_layouts}; slides = list(p.slides)
ttl = lambda s: next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()), "")
assert ttl(slides[0]) == "11/2/26 Jie requests"; nxt = next(i for i, s in enumerate(slides) if ttl(s).startswith("10/30/26")); old = list(p.slides._sldIdLst)[1:nxt]; print("replacing:", [ttl(s)[:50] for s in slides[1:nxt]])
hic_old = next(i for i, s in enumerate(slides, 1) if ttl(s).startswith("4E-H12")); n0 = len(slides)
def run(par, t, sz, b=False, c=INK):
    r = par.add_run(); r.text = t; r.font.name = "Cambria"; r.font.size = Pt(sz); r.font.bold = b; r.font.color.rgb = RGBColor.from_string(c)
def tb(s, x, y, w, h, t, sz=14, b=False, c=INK, name=None):
    o = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = o.text_frame; tf.word_wrap = True
    if name: o.name = name
    for k, line in enumerate(t if isinstance(t, list) else [t]):
        par = tf.paragraphs[0] if k == 0 else tf.add_paragraph(); par.space_after = Pt(3)
        if isinstance(line, tuple): run(par, line[0], sz, True, c); run(par, " " + line[1], sz, False, c)
        else: run(par, line, sz, b, c)
    return o
def new(t, sub):
    s = p.slides.add_slide(LAY["Title and Content"])
    for ph in list(s.placeholders): ph._element.getparent().remove(ph._element)
    tb(s, 0.17, 0.12, 12.9, 0.57, t, 28, True, name="Title"); tb(s, 0.33, 0.66, 12.69, 0.4, sub, 15, False, "8C8C8C", name="Subtitle"); return s
def pic(s, f, x, y):
    w, h = FR[f]; path = f"fig/{f}.png"; px = Image.open(path).size; assert abs(px[0] / w - 600) < 2 and abs(px[1] / h - 600) < 2, (f, px); s.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h)); return w, h
def table(s, rows, widths, x, y, rh, sz=12, hl=(), bold_cells=()):
    T = s.shapes.add_table(len(rows), len(widths), Inches(x), Inches(y), Inches(sum(widths)), Inches(rh * len(rows))); T.name = "Findings"; t = T.table
    for j, w in enumerate(widths): t.columns[j].width = Inches(w)
    for i, r in enumerate(rows):
        t.rows[i].height = Inches(rh)
        for j, v in enumerate(r):
            c = t.cell(i, j); c.vertical_anchor = MSO_ANCHOR.MIDDLE; c.margin_top = c.margin_bottom = Inches(0.015); c.margin_left = c.margin_right = Inches(0.05); c.fill.solid(); c.fill.fore_color.rgb = RGBColor.from_string("E3E3E3" if i == 0 else ("FCE9A8" if i in hl else "FFFFFF"))
            for k, line in enumerate(str(v).split("\n")):
                par = c.text_frame.paragraphs[0] if k == 0 else c.text_frame.add_paragraph(); run(par, line, sz, i == 0 or j == 0 or (i, j) in bold_cells)
NB = json.load(open("fig/norm_numbers.json")); import numpy as np; f2 = lambda x: f"{np.floor(x * 100 + 1e-9) / 100:.2f}"
A = pd.read_csv("data/a_known_complexes_v5.tsv", sep="\t", keep_default_na=False); PM = pd.read_csv("../../scripts/complex_catalog/mats/pct_all.tsv", sep="\t", index_col=0); PM.index = PM.index.str.strip().str.upper(); PM.columns = PM.columns.str.strip().str.upper(); pm = lambda a, b: float(PM.loc[a, b])
K = [("EZH2–MTF2", pm("EZH2", "MTF2"), "AF3 none; STRING physical", "MTF2 CpG (literature)", "Bridged by SUZ12"), ("MBD3–RBBP4/7", np.mean([pm("MBD3", "RBBP4"), pm("MBD3", "RBBP7")]), "Boltz none (trimmed); STRING physical", "–", "Mean of 2 pairs"),
     ("RBBP4–ZNF512B", pm("RBBP4", "ZNF512B"), "Boltz weak (trimmed); STRING physical", "–", "NAR 2024"), ("RBBP4–SMARCA1", pm("RBBP4", "SMARCA1"), "Boltz none (trimmed); STRING physical", "–", "NURF-type"), ("HP1–PCGF6", pm("HP1", "PCGF6"), "Boltz weak (trimmed); STRING physical", "–", "PRC1.6"),
     ("ADNP–HP1", pm("ADNP", "HP1"), "Boltz weak (trimmed); STRING physical", "–", "ChAHP; Nature 2018"), ("CHAMP1–HP1", pm("CHAMP1", "HP1"), "Boltz weak (trimmed); STRING physical", "–", ""), ("WIZ–ZNF644", pm("WIZ", "ZNF644"), "Boltz none (trimmed); STRING physical", "–", "G9a/GLP; both low library"),
     ("DNMT1–HP1/EZH2", np.mean([pm("DNMT1", "HP1"), pm("DNMT1", "EZH2")]), "Boltz none (trimmed); STRING physical", "–", "Mean of 2 pairs"), ("MEF2A–MEF2D", pm("MEF2A", "MEF2D"), "Boltz weak (trimmed); STRING physical", "JASPAR both", "Paralog dimer"), ("RELA–ZBTB7A", pm("RELA", "ZBTB7A"), "Boltz none (trimmed); STRING physical", "JASPAR both", ""),
     ("CTCF–STAG2", pm("CTCF", "STAG2"), "Boltz none (trimmed)", "JASPAR CTCF", "CTCF function caveat"), ("MAZ–ZNF143", pm("MAZ", "ZNF143"), "Boltz none (trimmed)", "JASPAR both", "")]
C_ = [("FOXK1–MLX", pm("FOXK1", "MLX"), "Boltz none (trimmed)", "CAP-SELEX call; JASPAR both", "No composite in table"), ("MLX–MSX2", pm("MLX", "MSX2"), "Boltz none (trimmed)", "CAP-SELEX call; JASPAR both", "No composite in table"), ("HOXD10–SREBF2", pm("HOXD10", "SREBF2"), "Boltz none (trimmed)", "CAP-SELEX call; JASPAR both", "No composite in table")]
N_ = [("MLX–SREBF2", pm("MLX", "SREBF2"), "AF3 strong (5/5 seeds); Boltz full strong", "JASPAR E-box (both); SRE", "bHLH dimer caveat"), ("HOXC4–RBBP4", pm("HOXC4", "RBBP4"), "AF3 strong (3/5 seeds); Boltz full none", "JASPAR HOXC4", "RBBP4 hub partner")]
rows = [["Pair", "Category", "Structural support", "Correlation percentile", "Motif", "Note"]]; bold = set()
for cat, G in (("Known", K), ("CAP-SELEX candidate", C_), ("Novel", N_)):
    for r in sorted(G, key=lambda x: -x[1]):
        rows.append([r[0], cat, r[2], f2(r[1]), r[3], r[4]])
        if r[1] >= .90: bold.add(len(rows) - 1)
s2 = new("Cobinding Complex Analysis- all complexes", "13 known, 3 CAP-SELEX candidates, 2 novel. Bold = correlation percentile ≥ 0.90")
table(s2, rows, [1.8, 1.95, 3.55, 1.0, 2.35, 2.25], 0.2, 1.05, 0.27, 12, bold_cells={(i, 3) for i in bold})
# known example
s4 = new("PRC2 core EZH2–MTF2- layer 1: structure" , "A TF has different sites: the MTF2 EH domain binds CpG DNA, its C-terminal region binds PRC2")
if PRC2:
    Sc = pd.read_csv("img/af3_prc2_scores.tsv", sep="\t"); T_ = Sc[Sc.job == "EZH2_SUZ12_MTF2"].set_index("pair"); D_ = Sc[Sc.job == "EZH2_MTF2"].iloc[0]; pic(s4, "struct_PRC2", 0.25, 1.1); pic(s4, "sitemap_PRC2", 7.0, 1.1); pic(s4, "pae_PRC2", 7.0, 3.95)
    tb(s4, 9.8, 3.95, 3.4, 2.4, [("EZH2–MTF2 alone:", f"ipTM {D_.iptm_top:.2f}, ipSAE {D_.ipsae_max:.2f}, {D_.seeds_strong}/5 seeds"), ("In trimer, MTF2–SUZ12:", f"ipTM {T_.loc['B-C', 'iptm_top']:.2f}, ipSAE {T_.loc['B-C', 'ipsae_max']:.2f}, {T_.loc['B-C', 'seeds_strong']}/5"), ("In trimer, EZH2–SUZ12:", f"ipTM {T_.loc['A-B', 'iptm_top']:.2f}, ipSAE {T_.loc['A-B', 'ipsae_max']:.2f}, {T_.loc['A-B', 'seeds_strong']}/5"), ("In trimer, EZH2–MTF2:", f"ipTM {T_.loc['A-C', 'iptm_top']:.2f}, ipSAE {T_.loc['A-C', 'ipsae_max']:.2f}, {T_.loc['A-C', 'seeds_strong']}/5")], 13, name="Scores")
else:
    pic(s4, "sitemap_PRC2_placeholder", 0.4, 1.3); tb(s4, 0.45, 5.4, 12.4, 0.5, "AlphaFold 3 model running: EZH2–MTF2 and EZH2–SUZ12–MTF2. The render and scores go here.", 15, True, "B2182B", name="Placeholder")
tb(s4, 0.3, 6.4, 12.7, 0.9, "Purple = DNA-binding site (EH, Li et al. 2017; InterPro IPR063986). Black ticks = interface residues within 5 Å of another chain.", 12, False, GR, name="Footnote") if PRC2 else tb(s4, 0.3, 6.1, 12.7, 0.9, "Purple = DNA-binding site (EH, Li et al. 2017; InterPro IPR063986). Domains from UniProt and InterPro.", 12, False, GR, name="Footnote")
def corr(t, sub, key, fig):
    s = new(t, sub); pic(s, fig, 0.15, 1.05); n_ = NB[key]; lab_ = {"all": "All elements", "noquies": "No quiescent", "tss": "TSS", "enhancer": "Enhancer", "repressive": "Repressive", "polycomb": "Polycomb only"}
    nc = NB[key + "_corrected"]
    table(s, [["ChromHMM elements", "Corrected percentile", "Within local cluster"]] + [[lab_[k], f2(nc[k][0]), f2(nc[k][1])] for k in lab_], [2.6, 1.6, 1.6], 7.3, 1.55, 0.42, 14)
    tb(s, 7.25, 5.0, 5.9, 2.0, ["Corrected = accessibility-corrected: each element's signal over its 79-TF mean, then Pearson, SpQN, percentile; elements with ≥ 20 reads. The matrix shown is before this correction.", "Within local cluster = rank among pairs in the pair's cluster on that matrix (dendrogram cut at 4 clusters).", "Repressive = ReprPC, ReprPCWk, Het, ZNF/Rpts. Polycomb = ReprPC, ReprPCWk."], 12, False, GR, name="Footnote"); return s
s5 = corr("PRC2 core EZH2–MTF2- layer 2: correlation", "Our normalized ChromHMM matrix (reads per cell + SpQN, percentile); PRC2 members outlined", "ezh2-mtf2", "normmat_EZH2_MTF2")
s6 = new("PRC2 core EZH2–MTF2- layer 3: motif", "Neither has a sequence motif; MTF2 reads CpG. Our co-bound elements are CpG-rich, but no more than matched elements")
pic(s6, "cpg_bars", 0.25, 1.2)
tb(s6, 9.2, 1.25, 3.95, 3.5, [("Literature:", "MTF2 EH domain binds unmethylated CpG (Li et al. 2017). EZH2 has no motif; JASPAR has neither."), ("Elements:", "7,574 both, 26,884 one; matched on GC and ATAC.")], 14, name="Summary")
tb(s6, 0.3, 5.0, 12.7, 1.0, [("Both vs matched:", "CpG o/e ratio 0.99 (95% CI 0.97–1.01), inside the range of 40 non-partner control pairs (0.98–1.02)."), ("Reading:", "no CpG enrichment beyond GC and accessibility at this depth.")], 15, name="Result")
# novel example
s7 = new("Novel MLX–SREBF2- layer 1: structure", "AlphaFold 3 strong on 5 of 5 seeds; the contacts sit on the dimerisation helices, and the basic DNA-contact region stays free")
pic(s7, "struct_MLX_SREBF2", 0.25, 1.1); pic(s7, "sitemap_MLX_SREBF2", 7.0, 1.1); pic(s7, "pae_MLX_SREBF2", 7.0, 3.95)
tb(s7, 9.8, 3.95, 3.4, 2.3, [("AF3:", "ipTM 0.45, ipSAE 0.57"), ("Boltz-2:", "strong, trimmed and full length"), ("Interface:", "IFACE")], 14, name="Scores")
tb(s7, 0.3, 6.4, 12.7, 0.9, "Purple = basic region, DNA contact (first 13 residues of the bHLH domain, inferred). Black ticks = interface. MLX ZIP helix taken from the model; SCAP-binding region approximate. Full-length SREBF2 includes membrane segments that the nuclear form lacks.", 12, False, GR, name="Footnote")
s8 = corr("Novel MLX–SREBF2- layer 2: correlation", "Same normalized matrix: percentile 0.84 before correction, 0.71 after", "mlx-srebf2", "normmat_MLX_SREBF2")
s9 = new("Novel MLX–SREBF2- layer 3: motif", "Both are bHLH-ZIP E-box binders; SREBF2 also reads the SRE. Composite sites are too rare to test at our depth")
for f, x in (("logo_mlx", 0.25), ("logo_srebf2_ebox", 4.6), ("logo_srebf2_sre", 8.95)): pic(s9, f, x, 1.15)
pic(s9, "halfsite", 0.25, 3.1)
tb(s9, 7.9, 3.15, 5.2, 1.8, [("Elements with both:", "15 at FIMO p ≤ 1e-5"), ("Co-binding at them:", "ratio 0.80 (95% CI 0.37–1.21)"), ("Spacing:", "none preferred at FDR 10%")], 14, name="Summary")
tb(s9, 0.3, 5.3, 12.7, 1.2, ["CAP-SELEX: pair tested, not called. Our composite test is not detectable at this depth, which does not rule the pair out.", "bHLH proteins dimerise readily, so a strong model may reflect the fold rather than a specific partner."], 13, False, GR, name="Footnote")
NEW = 7; hic = hic_old - len(old) + NEW; m184 = next(i for i, s in enumerate(slides, 1) if ttl(s).startswith("Cobinding Analysis")) - len(old) + NEW; tb(s2, 0.2, 6.62, 12.9, 0.8, [f"Correlation percentile: normalized ChromHMM matrix, all elements, SpQN (slide {m184}). AF3 where run, otherwise Boltz; – = no motif. Mean of 2 pairs is one number per complex.", f"Hi-C clique triangle: slide {hic}."], 12, False, GR, name="Footnote")
import json as J_; r_ = J_.load(open("img/MLX_SREBF2.sites.json")); cl = lambda v, d: sum(d[0] <= x <= d[1] for x in v); ctx = lambda v, a, b: f"{cl(v, a)} of {len(v)}"; hz = cl(r_["A"], (142, 220)) ; hzb = cl(r_["B"], (343, 401)); ba = cl(r_["A"], (129, 141)); bb = cl(r_["B"], (330, 342)); txt = f"HLH-ZIP {hz} of {len(r_['A'])} on MLX, {hzb} of {len(r_['B'])} on SREBF2; basic {ba} and {bb}"
for sh in s7.shapes:
    if sh.has_text_frame:
        for par in sh.text_frame.paragraphs:
            for rr in par.runs:
                if " IFACE" in rr.text: rr.text = " " + txt
print(txt)
Ls = p.slides._sldIdLst; ids = list(Ls); added = ids[n0:]
for e in old: Ls.remove(e); p.part.drop_rel(e.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"))
for k, e in enumerate(added): Ls.remove(e); Ls.insert(1 + k, e)
assert not any(w_ in sh.text_frame.text for s_ in (s2, s4, s5, s6, s7, s8, s9) for sh in s_.shapes if sh.has_text_frame for w_ in ("R1", "Network edge", "edge rule", "co-binding edge")), "edge wording left on a slide"
p.save(OUT); print("slides", n0, "->", len(p.slides), "Hi-C now", hic, "section ends at", ttl(p.slides[8])[:30])
