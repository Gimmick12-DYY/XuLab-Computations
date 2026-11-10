"""Fig 3 v7: collect the final-size panel SVGs into panels/ and the supplementary files into supplementary/ (Mac paths).
  a  panels_src/a_module_edges.svg   (a_module_edges_v7.py, per-edge null)        b  panels_src/b_class_bars.svg (b_class_bars_v7.py)
  c  panels_src/c_se_browser.svg     (c_se_browser_v7.py)                          legend for c-e: panels_src/cde_legend.svg (cde_legend_v7.py)
  d  approved rosette results_3d_model/se_sedb/figure_rosette_own_beads/se_rosette.svg (deck slide 22), edited here: slide title,
     right-hand legend and footnote removed (shared legend + figure_legend_notes.md instead), element labels enlarged to 6 pt at the
     placed size (12 -> 14.6 px in the 0.41x-scaled drawing); the 3D render and leader lines are untouched
  e  panels_src/e_se_flower.svg      (flower_v7.py)
Supplementary: RBBP4 network (v6 panel, slide 169), pairs loop support (approved D1, v6 wording), block-shuffle module-edge null,
approved SE edge-count table (browser_side_final.svg, slide 21).
  python prep_panels_v7.py"""
import re, shutil
from pathlib import Path
H = Path(__file__).resolve().parent; P = H / "panels"; SRC = H / "panels_src"; SUP = H / "supplementary"; P.mkdir(exist_ok=True); SUP.mkdir(exist_ok=True)
SE = Path("/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model/se_sedb"); V6 = Path("/Users/dyy/Desktop/HEK293T-ATGTAG/fig3_v6")
LABEL_PX = 14.6


def groups(s):
    for m in re.finditer(r'<g id="((?:text|line2d|patch)_\d+)">', s):
        i = m.start(); depth = 0
        for t in re.finditer(r'<g\b|</g>', s[i:]):
            depth += 1 if t.group(0) == '<g' else -1
            if depth == 0: yield i, i + t.end(), m.group(1); break


def xs_of(blk):
    use = [float(v) for v in re.findall(r'<use\b[^>]*\bx="([-\d.]+)"', blk)]
    if use: return use   # markers: their <defs> path is in local units, so judge by where the marker is placed
    xs = [float(v) for v in re.findall(r'\bx="([-\d.]+)"', blk)] + [float(v) for v in re.findall(r'translate\(([-\d.]+)', blk)]
    for d in re.findall(r'\bd="([^"]+)"', blk):
        nums = [float(v) for v in re.findall(r'-?\d+\.?\d*', d)]; xs += nums[0::2]
    return xs


s = (SE / "figure_rosette_own_beads/se_rosette.svg").read_text(); out, last, dropped = [], 0, 0
for a, b, gid in list(groups(s)):
    blk = s[a:b]; xs = xs_of(blk)
    if gid == "text_1" or (xs and min(xs) >= 640):   # title, and everything right of x = 640 pt (the legend column and footnote)
        out.append(s[last:a]); last = b; dropped += 1
out.append(s[last:]); s = "".join(out)
s = re.sub(r"(font: (?:italic |700 )?)12px", lambda m: f"{m.group(1)}{LABEL_PX}px", s)
(P / "d_se_rosette.svg").write_text(s); print("rosette: dropped", dropped, "groups; live text left:", s.count("<text"))
for st in ("a_module_edges", "b_class_bars", "c_se_browser", "cde_legend", "e_se_flower"): shutil.copy(SRC / f"{st}.svg", P / f"{st}.svg")
shutil.copy(V6 / "panels/a_network.svg", SUP / "S_rbbp4_network.svg")
for e in ("svg", "pdf", "png"): shutil.copy(V6 / f"supplementary/S_pairs_loop_support.{e}", SUP / f"S_pairs_loop_support.{e}")
for e in ("svg", "pdf", "png"): shutil.copy(SE / f"figure_gold/browser_side_final.{e}", SUP / f"S_se_edge_table.{e}")
for f in sorted(P.glob("*.svg")):
    t = re.sub(r"<!--.*?-->", "", f.read_text(), flags=re.S)
    print(f.name, "| 'clique' in text:", "clique" in " ".join(re.findall(r"<text\b[^>]*>(.*?)</text>", t, flags=re.S)).lower())
