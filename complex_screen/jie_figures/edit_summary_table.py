"""Slide 1 table: correlation first, highest percentile across the five element sets, pairs below 0.60 dropped, more novel candidates; slide 5 matrix on the enhancer set. usage: edit_summary_table.py in.pptx out.pptx"""
import copy, sys
import numpy as np, pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"; fl = lambda v: f"{np.floor(v * 100 + 1e-9) / 100:.2f}"
C = pd.read_csv("data/summary_current_values.tsv", sep="\t").set_index("pair"); N = pd.read_csv("data/summary_novel_all.tsv", sep="\t", keep_default_na=False)
NOMOTIF = {"RBBP4", "EZH2", "XPA", "DR1", "HP1"}; NOJASPAR = {"THAP6", "ZNF746", "WIZ", "ZNF367", "ZNF581"}; SHARED = {("ZNF143", "ZNF282"): "No shared motif"}
def motif(a, b):
    if (a, b) in SHARED: return SHARED[(a, b)]
    if a in NOMOTIF or b in NOMOTIF: return "n/a (no DNA motif)"
    if a in NOJASPAR or b in NOJASPAR: return "n/a (no JASPAR motif)"
    raise ValueError((a, b))
p = Presentation(sys.argv[1]); s = p.slides[0]; G = next(x for x in s.shapes if x.has_table); assert G.name == "Findings"; tb = G.table._tbl; trs = list(tb.tr_lst)
def settext(tc, text, bold=None):
    ps = tc.findall(f".//{A}p")
    for q in ps[1:]: q.getparent().remove(q)
    rs = ps[0].findall(f"{A}r")
    for r in rs[1:]: ps[0].remove(r)
    rs[0].find(f"{A}t").text = text
    if bold is not None: rs[0].find(f"{A}rPr").set("b", "1" if bold else "0")
def text(tc): return "".join(t.text or "" for t in tc.iter(f"{A}t"))
def order(tr, note, note_bold=False):
    t = tr.findall(f"{A}tc"); assert len(t) == 5; nt = copy.deepcopy(t[1]); settext(nt, note, note_bold)
    for x in t: tr.remove(x)
    ins = tr.index(tr.find(f"{A}extLst")) if tr.find(f"{A}extLst") is not None else len(tr)
    for k, x in enumerate((t[0], t[1], t[3], t[2], t[4], nt)): tr.insert(ins + k, x)
def clean(st):
    st = st.replace("; STRING physical", "").replace("STRING physical", "").strip().rstrip(";").strip(); return st or "–"
hdr = trs[0]; settext(hdr.findall(f"{A}tc")[2], "Structural support (AF3, Boltz)"); order(hdr, "Note", True); rows = {text(t.findall(f"{A}tc")[0]): t for t in trs[1:]}; proto = copy.deepcopy(rows["MLX–SREBF2"]); dropped = []; keep = []
for nm, tr in rows.items():
    v = C.loc[nm, "value"]
    if v < .60: dropped.append(f"{nm} ({C.loc[nm, 'category']}): {fl(v)} ({C.loc[nm, 'set']})"); tb.remove(tr); continue
    t = tr.findall(f"{A}tc"); settext(t[2], clean(text(t[2]))); settext(t[3], f"{fl(v)} ({C.loc[nm, 'set']})", float(fl(v)) >= .90); note = "Not partner-specific" if nm == "MLX–SREBF2" else "–"
    if nm == "MLX–SREBF2": settext(t[2], "AF3 strong 5/5, on E-box DNA"); settext(t[4], "Shared E-box (q = 9 × 10⁻⁷)")
    order(tr, note); keep.append((nm, tr, C.loc[nm, "category"], v))
PURE = [f"{r.a}–{r.b}" for r in N.head(9).itertuples()]; N["strong_last"] = (N.tier != "strong").astype(int); N = N.sort_values(["strong_last", "value"], ascending=[True, False]); cnt = {"MLX": 1, "SREBF2": 1}; pick = []
for r in N.itertuples():
    if cnt.get(r.a, 0) >= 2 or cnt.get(r.b, 0) >= 2: continue
    cnt[r.a] = cnt.get(r.a, 0) + 1; cnt[r.b] = cnt.get(r.b, 0) + 1; pick.append(r.Index)
    if len(pick) == 9: break
new = []; TOP = N.loc[pick]
for r in TOP.itertuples():
    tr = copy.deepcopy(proto); t = tr.findall(f"{A}tc"); nm = f"{r.a}–{r.b}"; settext(t[0], nm); settext(t[1], "Novel"); settext(t[2], r.structure, False); settext(t[3], f"{fl(r.value)} ({r.set})", float(fl(r.value)) >= .90); settext(t[4], motif(r.a, r.b), False); order(tr, "Promiscuous partner" if r.promiscuous else "–"); new.append((nm, tr, "Novel", r.value))
CAT = {"Known": 0, "CAP-SELEX candidate": 1, "Novel": 2}; allr = sorted(keep + new, key=lambda x: (CAT[x[2]], 0 if x[2] != "Novel" else (0 if x[0] == "MLX–SREBF2" or x[0] in {f"{r.a}–{r.b}" for r in TOP[TOP.tier == "strong"].itertuples()} else 1), -x[3]))
for x in list(tb.tr_lst)[1:]: tb.remove(x)
for x in allr: tb.append(x[1])
ROW = int(Inches(.25))
for tr in tb.tr_lst: tr.set("h", str(ROW))
for e in tb.iter(f"{A}tcPr"): e.set("marT", "4572"); e.set("marB", "4572")
grid = tb.tblGrid; gc = list(grid); nc = copy.deepcopy(gc[1])
for e in nc.iter():
    if e.tag.endswith("colId"): e.set("val", "20099")
for x in gc: grid.remove(x)
W = (1.7, 1.8, 1.95, 3.0, 2.6, 1.77)
for x, w in zip((gc[0], gc[1], gc[3], gc[2], gc[4], nc), W): x.set("w", str(int(Inches(w)))); grid.append(x)
# +1 = MLX-SREBF2: the novel row carried over from the curated table (CUR). It is excluded from N because every CUR pair is excluded, but it is novel: no STRING physical >= .7, CAP-SELEX tested-not-called, AF3 strong 5/5. Checked 2026-11-10.
G.top = Inches(1.03); G.width = sum(int(Inches(w)) for w in W); G.height = ROW * len(tb.tr_lst); nk = sum(x[2] == "Known" for x in allr); nl = sum(x[2].startswith("CAP") for x in allr); total = len(N) + 1
sub = next(x for x in s.shapes if x.name == "Subtitle"); r = sub.text_frame.paragraphs[0].runs; r[0].text = f"{nk} known, {nl} CAP-SELEX and {total} novel candidates at percentile ≥ 0.60, in three-layer complex analysis (ChromHMM element, Structural, Motif)"
for x in r[1:]: x.text = ""
# +1 = MLX-SREBF2: the novel row carried over from the curated table (CUR). It is excluded from N because every CUR pair is excluded, but it is novel: no STRING physical >= .7, CAP-SELEX tested-not-called, AF3 strong 5/5. Checked 2026-11-10.
ft = s.shapes.add_textbox(Inches(.21), Inches(7.17), Inches(11.9), Inches(.28)); ft.name = "Footnote"; f = ft.text_frame; f.word_wrap = True; f.margin_top = f.margin_bottom = 0; q = f.paragraphs[0]; rr = q.add_run(); rr.text = f"{total} novel candidates pass; {len(TOP) + 1} shown, strong before weak, at most 2 per TF. Value = highest percentile across five element sets; bold ≥ 0.90."; rr.font.size = Pt(12); rr.font.name = "Cambria"; rr.font.color.rgb = RGBColor.from_string("1A1A1A")
L = [f"{r.a}–{r.b}: {fl(r.value)} ({r.set}); {r.structure}" + ("; promiscuous partner" if r.promiscuous else "") for r in N.itertuples()]
s.notes_slide.notes_text_frame.text = "\n".join(["Correlation value: highest percentile across all elements, no quiescent, TSS, enhancer and repressive. For a group row, the best pair in the group (MBD3–RBBP4/7: " + C.loc["MBD3–RBBP4/7", "best_of"] + "; DNMT1–HP1/EZH2: " + C.loc["DNMT1–HP1/EZH2", "best_of"] + ").", "Novel candidate: not known and not a CAP-SELEX call, value ≥ 0.60, predicted interface (AF3 strong or weak where run, otherwise Boltz full-length, otherwise Boltz screen). NEUROD4 pairs and STAG2-wrap pairs left out; pairs with a curated physical interaction counted as known. CBX1/3/5 = HP1.", f"Dropped below 0.60 ({len(dropped)}): " + "; ".join(dropped) + ".", f"Pure correlation ranking, for comparison: " + ", ".join(PURE) + ", plus MLX–SREBF2.", f"All {total} novel candidates, strong first, then by correlation: MLX–SREBF2: {fl(C.loc['MLX–SREBF2', 'value'])} (enhancer); AF3 strong 5/5."] + L)
# slide 5
s5 = next(x for x in p.slides if any(sh.has_text_frame and "MLX–SREBF2- layer 2" in sh.text_frame.text for sh in x.shapes)); pc = next(x for x in s5.shapes if x.name == "Matrix"); f_ = "fig/heatmap_enh_MLX_SREBF2_seq.png"; assert Image.open(f_).size == (4200, 3750); old = pc._element.blipFill.find(f"{A}blip"); rid_old = old.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"); _, rid = s5.part.get_or_add_image_part(f_); old.set("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed", rid); xml = s5.part._element.xml
if f'"{rid_old}"' not in xml: s5.part.rels.pop(rid_old)
def one(name, frag, tx):
    sh = next(x for x in s5.shapes if x.name == name and frag in x.text_frame.text); r = sh.text_frame.paragraphs[0].runs; k = next(i for i, x in enumerate(r) if frag in x.text); r[k].text = tx
    for x in r[k + 1:]: x.text = ""
one("Subtitle", "Percentile", "MLX–SREBF2 co-varies most on enhancers (0.78); 0.71 on all elements. Matrix: enhancer elements.")
one("Conclusion", "MLX–SREBF2 is moderate", "MLX–SREBF2 is highest on enhancers (0.78), moderate on all elements (0.71) and low on repressive (0.44).")
print("rows", len(tb.tr_lst) - 1, "known", nk, "cap", nl, "novel shown", sum(x[2] == "Novel" for x in allr), "total novel", total, "dropped", dropped); p.save(sys.argv[2]); print("ok", len(list(p.slides)))
