# m34: plain-overlap version of Figure 5 slides (no expected / fold). Replace pictures c, h, i, j; update subtitles and supplement bullets. usage: m34_deck_simple.py <in> <out>
import sys, zipfile, collections, re
from pptx import Presentation
IN, OUT = sys.argv[1], sys.argv[2]; PL = "/work/users/d/y/dyy12/XuLab/regulatome/m34_fig5_panels_v3/plots/slide"; MK = "FIG5V3"
SUB = {"Fig5c_ours_vs_bulk": "% of Perturb-seq genes found, ours vs bulk ChIP; bulk lists cover more genes (NRF1: 54% vs 14%); some references are not HEK293T",
       "Fig5h_threshold_sweep": "% of Perturb-seq genes found in our Regulome as the knockdown threshold rises (top-25% sgRNA cells)",
       "Fig5i_high_confidence_per_tf": "Per TF, % of its Perturb-seq genes (|z| ≥ 2) found in our Regulome; grey = the TF's Regulome list size",
       "Fig5j_binary_overlap": "11,337 TF-gene pairs in both: 17.7% of Perturb-seq genes, 9.6% of our links"}
BUL = ["• ML model, TF split (63–76 TFs): finds responders mainly from gene-level features, not from binding",
       "• Pipeline audit: every step from peaks to ABC to genes to Perturb-seq checked; no bug found",
       "• Perturb-seq limits: known partners do not look alike; STAG2 knockdown 18% with no responders",
       "• NRF1 note: our peaks miss TFAM, CYCS and other classic targets that ENCODE NRF1 hits"]
p = Presentation(IN); n0 = len(p.slides); done = set()
for s in p.slides:
    for pic in [sh for sh in s.shapes if sh.name.startswith(f"{MK}-Fig5") and sh.shape_type == 13]:
        key = pic.name[len(MK) + 1:]
        if key not in SUB: continue
        l, t, w, h = pic.left, pic.top, pic.width, pic.height; el = pic._element; parent = el.getparent(); idx = list(parent).index(el)
        new = s.shapes.add_picture(f"{PL}/{key}.png", l, t, width=w, height=h); new.name = pic.name; parent.remove(el); parent.remove(new._element); parent.insert(idx, new._element)
        for sh in s.shapes:
            if sh.name == f"{MK}-subtitle" and sh.has_text_frame:
                r = sh.text_frame.paragraphs[0].runs; r[0].text = SUB[key]
                for x in r[1:]: x.text = ""
        done.add(key)
    for sh in s.shapes:
        if sh.name == f"{MK}-bullets" and sh.has_text_frame:
            for para, b in zip(sh.text_frame.paragraphs, BUL):
                para.runs[0].text = b
                for x in para.runs[1:]: x.text = ""
            done.add("bullets")
assert done == set(SUB) | {"bullets"}, done
bad = [(i, sh.name, sh.text_frame.text[:90]) for i, s in enumerate(p.slides, 1) if any(x.name.startswith(MK) for x in s.shapes) for sh in s.shapes if sh.has_text_frame and re.search(r"expected|fold|chance|×|both halves|reproducib|Tier", sh.text_frame.text, re.I)]
print("flagged text left:", bad); assert not bad
p.part.rename_slide_parts([e.rId for e in p.slides._sldIdLst])
for i, s in enumerate(p.slides, 1):
    if s.has_notes_slide: s.notes_slide.part.partname = s.notes_slide.part.partname.__class__(f"/ppt/notesSlides/notesSlide{i}.xml")
assert len(p.slides) == n0; p.save(OUT); q = Presentation(OUT); print("reopened", len(q.slides))
for i in range(12):
    s = q.slides[i]; t = next((sh.text_frame.text for sh in s.shapes if sh.name == f"{MK}-title"), "(divider)"); u = next((sh.text_frame.text for sh in s.shapes if sh.name == f"{MK}-subtitle" and sh.has_text_frame), ""); print(i + 1, t, "|", u[:110])
c = collections.Counter(zipfile.ZipFile(OUT).namelist()); d = [k for k, v in c.items() if v > 1]; print("duplicates", d); assert not d
