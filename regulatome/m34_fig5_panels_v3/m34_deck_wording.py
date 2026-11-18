# m34: plain wording on Figure 5 slides (no recovery/precision/support/expected/fold). Replace pictures a, c, d, f, h, i, j; set subtitles and two titles. usage: m34_deck_wording.py <in> <out>
import sys, zipfile, collections, re
from pptx import Presentation
IN, OUT = sys.argv[1], sys.argv[2]; PL = "/work/users/d/y/dyy12/XuLab/regulatome/m34_fig5_panels_v3/plots/slide"; MK = "FIG5V3"
SUB = {"Fig5a_layers": "Our HITAG binding linked to genes by ABC; Perturb-seq (top-25% sgRNA cells) adds gene changes and direction",
       "Fig5c_ours_vs_bulk": "% of Perturb-seq genes found, ours vs bulk ChIP; bulk lists cover more genes (NRF1: 54% vs 14%); some references are not HEK293T",
       "Fig5d_NRF1": "1,610 of our genes; 355 in both, 243 go down on knockdown; PGAM1 promoter as an example",
       "Fig5f_activator_repressor": "Direction of each TF's genes in both, in HEK293T; darker = |z| ≥ 4",
       "Fig5h_threshold_sweep": "Share of Perturb-seq genes found in our Regulome as the knockdown threshold rises",
       "Fig5i_high_confidence_per_tf": "Per TF, share of Perturb-seq genes (|z| ≥ 2) found in our Regulome; grey = our genes per TF",
       "Fig5j_binary_overlap": "11,337 TF-gene pairs in both: 17.7% of Perturb-seq genes, 9.6% of our genes"}
TIT = {"Fig5h_threshold_sweep": "Regulome- Figure 5h · Perturb-seq genes found, by knockdown threshold", "Fig5i_high_confidence_per_tf": "Regulome- Figure 5i · Perturb-seq genes found, per TF"}
def settext(sh, text):
    r = sh.text_frame.paragraphs[0].runs; r[0].text = text
    for x in r[1:]: x.text = ""
p = Presentation(IN); n0 = len(p.slides); done = set()
for s in p.slides:
    for pic in [sh for sh in s.shapes if sh.name.startswith(f"{MK}-Fig5") and sh.shape_type == 13]:
        key = pic.name[len(MK) + 1:]
        if key not in SUB: continue
        l, t, w, h = pic.left, pic.top, pic.width, pic.height; el = pic._element; parent = el.getparent(); idx = list(parent).index(el)
        new = s.shapes.add_picture(f"{PL}/{key}.png", l, t, width=w, height=h); new.name = pic.name; parent.remove(el); parent.remove(new._element); parent.insert(idx, new._element)
        for sh in s.shapes:
            if sh.name == f"{MK}-subtitle" and sh.has_text_frame: settext(sh, SUB[key])
            if sh.name == f"{MK}-title" and sh.has_text_frame and key in TIT: settext(sh, TIT[key])
        done.add(key)
assert done == set(SUB), set(SUB) - done
bad = [(i, sh.name, sh.text_frame.text[:90]) for i, s in enumerate(p.slides, 1) if any(x.name.startswith(MK) for x in s.shapes) for sh in s.shapes if sh.has_text_frame and re.search(r"recovery|precision|support|expected|fold|chance|×|both halves|reproducib|Tier", sh.text_frame.text, re.I)]
print("flagged text left:", bad); assert not bad
p.part.rename_slide_parts([e.rId for e in p.slides._sldIdLst])
for i, s in enumerate(p.slides, 1):
    if s.has_notes_slide: s.notes_slide.part.partname = s.notes_slide.part.partname.__class__(f"/ppt/notesSlides/notesSlide{i}.xml")
assert len(p.slides) == n0; p.save(OUT); q = Presentation(OUT); print("reopened", len(q.slides))
for i in range(12):
    s = q.slides[i]; t = next((sh.text_frame.text for sh in s.shapes if sh.name == f"{MK}-title"), "(divider)"); u = next((sh.text_frame.text for sh in s.shapes if sh.name == f"{MK}-subtitle" and sh.has_text_frame), ""); print(i + 1, t, "|", u[:100])
c = collections.Counter(zipfile.ZipFile(OUT).namelist()); d = [k for k, v in c.items() if v > 1]; print("duplicates", d); assert not d
