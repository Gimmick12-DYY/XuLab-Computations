# m34: replace the Figure 5g picture (final Figure 4 pairs) and its subtitle in place. usage: m34_deck_g.py <in> <out>
import sys, zipfile, collections, re
from pptx import Presentation
IN, OUT = sys.argv[1], sys.argv[2]; PL = "/work/users/d/y/dyy12/XuLab/regulatome/m34_fig5_panels_v3/plots/slide"; MK = "FIG5V3"
SUB = {"Fig5g_pairs": "Shared targets of the Figure 4 pairs and their knockdown response (RBBP4: no usable knockdown)"}
p = Presentation(IN); n0 = len(p.slides); done = set()
for s in p.slides:
    pics = [sh for sh in s.shapes if sh.name.startswith(f"{MK}-Fig5") and sh.shape_type == 13]
    for pic in pics:
        key = pic.name[len(MK) + 1:]
        if key not in SUB: continue
        l, t, w, h = pic.left, pic.top, pic.width, pic.height; el = pic._element; idx = list(el.getparent()).index(el)
        new = s.shapes.add_picture(f"{PL}/{key}.png", l, t, width=w, height=h); new.name = pic.name
        parent = el.getparent(); parent.remove(el); parent.remove(new._element); parent.insert(idx, new._element)
        for sh in s.shapes:
            if sh.name == f"{MK}-subtitle" and sh.has_text_frame:
                r = sh.text_frame.paragraphs[0].runs; r[0].text = SUB[key]
                for x in r[1:]: x.text = ""
        done.add(key)
assert done == set(SUB), set(SUB) - done
bad = []
for i, s in enumerate(p.slides, 1):
    if any(sh.name.startswith(MK) for sh in s.shapes):
        for sh in s.shapes:
            if sh.has_text_frame and re.search(r"both halves|reproducib|Tier", sh.text_frame.text, re.I): bad.append((i, sh.name, sh.text_frame.text[:80]))
print("forbidden words left:", bad); assert not bad
p.part.rename_slide_parts([el.rId for el in p.slides._sldIdLst])
for i, s in enumerate(p.slides, 1):
    if s.has_notes_slide: s.notes_slide.part.partname = s.notes_slide.part.partname.__class__(f"/ppt/notesSlides/notesSlide{i}.xml")
assert len(p.slides) == n0; p.save(OUT)
q = Presentation(OUT); print("reopened", len(q.slides))
for i in range(11):
    s = q.slides[i]; t = next((sh.text_frame.text for sh in s.shapes if sh.name == f"{MK}-title"), "(divider)"); u = next((sh.text_frame.text for sh in s.shapes if sh.name == f"{MK}-subtitle" and sh.has_text_frame), "")
    print(i + 1, t, "|", u[:100])
c = collections.Counter(zipfile.ZipFile(OUT).namelist()); d = [k for k, v in c.items() if v > 1]; print("duplicates", d); assert not d
