"""Fig 3 v6: make the per-panel SVGs in panels/ from the APPROVED deck figures (Mac paths). Only text is edited here:
  a  network   <- Longleaf hic/work_wt293t/loops/union_5k_10k_0.7/RBBP4_sweep_k50_i10000_s555_pad30kb/publication_plots_v2/
                  tf_peak_network_cliques_loops.svg (slide 169; copied to src/). Glyph-path text -> live Arial text; title dropped;
                  legend dropped here and redrawn at readable size by build_fig3_v6.py with the v6 wording (same handles).
  b  class bars <- panels_src/e_class_bars.svg (e_class_bars_v6.py: approved figC1_class_bars_paletteC, wording only)
  c             <- panels_src/b2_module_loop_coverage.svg (b_hic_support_v6.py: approved D2 + module-edge null panel at its right)
  supplementary/S_pairs_loop_support.* <- panels_src/b1_pairs_loop_support.* (approved D1, v6 wording; out of Fig 3)
  d1 / d2       <- results_3d_model/se_sedb/figure_gold/browser_final.svg (title dropped) + browser_side_final.svg (unchanged)
  e1            <- results_3d_model/se_sedb/figure_rosette_own_beads/se_rosette.svg (title dropped)
  e2            <- panels_src/flower_v6.svg (flower_v6.py: approved flower, petals outward)
  python prep_panels_v6.py"""
import re, shutil
from pathlib import Path
from svgtext import glyph_text_to_live, set_live_text
H = Path(__file__).resolve().parent; P = H / "panels"; P.mkdir(exist_ok=True); SRC = H / "panels_src"
SE = Path("/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model/se_sedb")


def drop_group(s, gid):
    i = s.index(f'<g id="{gid}">'); depth = 0
    for t in re.finditer(r'<g\b|</g>', s[i:]):
        depth += 1 if t.group(0) == '<g' else -1
        if depth == 0: return s[:i] + s[i + t.end():]


s = (H / "src/approved_tf_peak_network_cliques_loops.svg").read_text()
s = drop_group(s, "legend_1")
s = glyph_text_to_live(s, drop=("RBBP4 co-binding clusters",))
s = s.replace("fill: #f7f7f5", "fill: #ffffff")   # slide background tint -> white page
assert "clique" not in re.sub(r"<!--.*?-->", "", s, flags=re.S).lower().replace("cliques_loops", "")
(P / "a_network.svg").write_text(s)
shutil.copy(SRC / "e_class_bars.svg", P / "b_class_bars.svg")
for f in P.glob("c1_*.svg"): f.unlink()   # v6 (user 2026-11-09): pairs chart moved to supplementary/
SUP = H / "supplementary"; SUP.mkdir(exist_ok=True)
for e in ("svg", "pdf", "png"): shutil.copy(SRC / f"b1_pairs_loop_support.{e}", SUP / f"S_pairs_loop_support.{e}")
for e in ("svg", "pdf", "png", "tsv"):   # stricter whole-module block-shuffle version of panel c (NULL_MODE=block)
    if (SRC / f"S_module_edge_block_null.{e}").exists(): shutil.copy(SRC / f"S_module_edge_block_null.{e}", SUP / f"S_module_edge_block_null.{e}")
for f in P.glob("c2_*.svg"): f.unlink()
shutil.copy(SRC / "b2_module_loop_coverage.svg", P / "c_module_loop_support.svg")
(P / "d1_se_browser.svg").write_text(set_live_text((SE / "figure_gold/browser_final.svg").read_text(), drop=("TF binding and Hi-C at the super-enhancer*",)))
shutil.copy(SE / "figure_gold/browser_side_final.svg", P / "d2_se_edge_table.svg")
(P / "e1_se_rosette.svg").write_text(set_live_text((SE / "figure_rosette_own_beads/se_rosette.svg").read_text(), drop=("Most-closed structure*",)))
shutil.copy(SRC / "flower_v6.svg", P / "e2_se_flower.svg")
for f in sorted(P.glob("*.svg")):
    t = re.sub(r"<!--.*?-->", "", f.read_text(), flags=re.S)
    txt = " ".join(re.sub(r"<[^>]+>", "", m) for m in re.findall(r"<text\b[^>]*>.*?</text>", t, flags=re.S))
    print(f.name, "live text:", t.count("<text"), "| 'clique' in text:", "clique" in txt.lower())
