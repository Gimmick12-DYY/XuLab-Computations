"""Collect the panels of the new Fig 3 plan (user, 2026-11-11) into XuLab/figures/Figure3/, one file set per panel; no layout.
Each panel: source SVG -> slide-style titles dropped, 'clique(s)' -> 'module(s)' in figure text, matplotlib CSS font shorthand
rewritten as explicit Arial attributes (Illustrator-safe; glyph-outline text converted to live Arial first), cropped to its content
(viewBox) -> <letter>_<name>.svg, .pdf and 300 dpi .png via rsvg-convert (Arial from /work/users/d/y/dyy12/fonts/arial).
Sources and commits are listed in README.md.   Runs on Longleaf: python collect_figure3.py"""
import os, re, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
X = Path("/work/users/d/y/dyy12/XuLab"); F = X / "figures"; OUT = F / "Figure3"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(OUT)); from svgtext import glyph_text_to_live, set_live_text, _groups  # noqa: E402
P2 = X / "hic/work_wt293t/loops/union_5k_10k_0.7/RBBP4_sweep_k50_i10000_s555_pad30kb/publication_plots_v2"; V7 = F / "fig3_v7"
FC = OUT / "fonts.conf"
FC.write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig><dir>/work/users/d/y/dyy12/fonts/arial</dir>'
              '<include ignore_missing="yes">/etc/fonts/fonts.conf</include><cachedir>' + str(OUT / ".fccache") + '</cachedir></fontconfig>')
ENV = dict(os.environ, FONTCONFIG_FILE=str(FC))
PS = {("normal", "normal"): "ArialMT", ("bold", "normal"): "Arial-BoldMT", ("normal", "italic"): "Arial-ItalicMT", ("bold", "italic"): "Arial-BoldItalicMT"}


def illustrator_text(svg):   # as build_fig3_v7.py (commit 715c1a5)
    def one(m):
        tag = m.group(0); sm = re.search(r'style="([^"]*)"', tag)
        if not sm: return tag
        keep, size, wt, it = [], None, "normal", "normal"
        for d in [d.strip() for d in sm.group(1).split(";") if d.strip()]:
            k, _, v = d.partition(":"); k, v = k.strip(), v.strip()
            if k == "font":
                mm_ = re.match(r"(?:(italic|oblique)\s+)?(?:(bold|[1-9]00)\s+)?([\d.]+)px", v)
                if mm_: it = "italic" if mm_.group(1) else it; wt = "bold" if mm_.group(2) in ("bold", "700", "800", "900") else wt; size = mm_.group(3)
            elif k == "font-size": size = v.replace("px", "")
            elif k == "font-weight": wt = "bold" if v in ("bold", "700", "800", "900") else "normal"
            elif k == "font-style": it = "italic" if v in ("italic", "oblique") else "normal"
            elif k == "font-family": pass
            else: keep.append(f"{k}: {v}")
        if size is None: return tag
        tag = tag.replace(sm.group(0), f'style="{"; ".join(keep)}"' if keep else "")
        return (tag[:-1].rstrip() + f' font-family="Arial, {PS[(wt, it)]}, Helvetica, sans-serif" font-size="{float(size):.3f}"'
                + (' font-weight="bold"' if wt == "bold" else "") + (' font-style="italic"' if it == "italic" else "") + ">")
    return re.sub(r"<text\b[^>]*>", one, svg)


def module_words(s):   # wording rules: never "clique"; "coordinated binding", never "co-binding" (user 2026-11-11); comments/ids untouched
    f = lambda m: m.group(1) + m.group(2).replace("Cliques", "Modules").replace("cliques", "modules").replace("Clique", "Module").replace("clique", "module").replace("cobinding module", "higher-order coordinated binding module").replace("co-binding", "coordinated binding").replace("Co-binding", "Coordinated binding").replace("cobinding", "coordinated binding") + m.group(3)
    return re.sub(r"(<text\b[^>]*>)(.*?)(</text>)", f, s, flags=re.S)


def crop(svg_path, scale=1.0):
    s = svg_path.read_text(); vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', s).group(1).split()]
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "p.png"; subprocess.run(["rsvg-convert", "-z", "1.5", "-b", "white", "-o", str(png), str(svg_path)], check=True, env=ENV)
        a = np.asarray(Image.open(png).convert("L")).astype(int)
    ys, xs = np.where(a < 248); k = a.shape[1] / vb[2]; pad = 2.0
    x0, x1 = max(xs.min() / k - pad, 0), min(xs.max() / k + pad, vb[2]); y0, y1 = max(ys.min() / k - pad, 0), min(ys.max() / k + pad, vb[3])
    box = (vb[0] + x0, vb[1] + y0, x1 - x0, y1 - y0)
    m = re.search(r"<svg\b[^>]*>", s); head = m.group(0)
    head = re.sub(r'\swidth="[^"]*"', "", head); head = re.sub(r'\sheight="[^"]*"', "", head); head = re.sub(r'\sviewBox="[^"]*"', "", head)
    head = head[:-1] + f' width="{box[2] * scale:.2f}pt" height="{box[3] * scale:.2f}pt" viewBox="{box[0]:.3f} {box[1]:.3f} {box[2]:.3f} {box[3]:.3f}">'
    svg_path.write_text(s[:m.start()] + head + s[m.end():]); return tuple(v * scale for v in box)


def emit(name, s, scale=1.0):   # scale: print size of panels drawn on a larger canvas with fonts preset for that size (v7)
    p = OUT / f"{name}.svg"; p.write_text(illustrator_text(module_words(s))); box = crop(p, scale)
    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(OUT / f"{name}.pdf"), str(p)], check=True, env=ENV)
    subprocess.run(["rsvg-convert", "-d", "300", "-p", "300", "-b", "white", "-o", str(OUT / f"{name}.png"), str(p)], check=True, env=ENV)
    t = p.read_text(); print(f"{name}: {box[2] / 72 * 25.4:.0f} x {box[3] / 72 * 25.4:.0f} mm | live text {t.count('<text')} | 'clique' in text:",
                            "clique" in " ".join(re.findall(r"<text\b[^>]*>(.*?)</text>", t, flags=re.S)).lower(), "| font shorthand:", t.count("font: "))


def drop_text_groups(s, needles):   # remove whole glyph-text groups whose <!-- comment --> contains any needle
    out, last = [], 0
    for a, b, gid in list(_groups(s)):
        cm = re.search(r"<!--(.*?)-->", s[a:b], re.S)
        if cm and any(n in cm.group(1) for n in needles): out.append(s[last:a]); out.append(f'<g id="{gid}"/>'); last = b
    out.append(s[last:]); return "".join(out)


# b: RBBP4 network (deck slide 169) and its Hi-C loop circos (slide 170)
s = (P2 / "tf_peak_network_cliques_loops.svg").read_text().replace("fill: #f7f7f5", "fill: #ffffff")
emit("b1_rbbp4_network", glyph_text_to_live(s, drop=("RBBP4 co-binding clusters",)))
s = (P2 / "RBBP4_clique_loop_edges_circos_tf_top.svg").read_text()
emit("b2_rbbp4_hic_circos", glyph_text_to_live(drop_text_groups(s, ("Genome-wide RBBP4",)), drop=("Genome-wide RBBP4",)))
# c: RBBP4 genome-wide chromosome plot, module edges above / Peakachu loops below (slide 172)
for v in ("plain", "highlight"):
    s = (P2 / f"RBBP4_clique_edges_above_loops_below_chromosomes.{v}.svg").read_text()
    emit(f"c_rbbp4_chromosomes_{v}", glyph_text_to_live(s, drop=("Genome-wide RBBP4", "Pairs above chromosomes")))
# d, e, g, h, i: Fig 3 v7 panels (final-size renders, commits 9766509 / 1a3c955 / 715c1a5)
for name, src, sc in (("d_module_edges_hic_support", "a_module_edges", 1.0), ("e_module_genomic_class", "b_class_bars", 1.0), ("g_se_browser", "c_se_browser", 0.739),
                      ("h_se_3d_structure", "d_se_rosette", 0.421), ("i_se_flower", "e_se_flower", 0.367)):
    emit(name, (V7 / "panels" / f"{src}.svg").read_text(), sc)
s = (V7 / "panels/cde_legend.svg").read_text().replace("3D model (d):", "3D model (h):").replace("small spheres in d)", "small spheres in h)")
emit("ghi_shared_legend", s)
# f: chord of the most widely shared module edges (deck slide 72; hub_figures/basic/clique_edge_overlap/fig_chord_tf_top_shared_edges.svg)
s = (OUT / "src/fig_chord_tf_top_shared_edges.svg").read_text()
emit("f_top_shared_edges_chord", set_live_text(s, drop=("TFs (left) and the 10 most widely shared*", "a ribbon = that TF has this exact edge*")))
# j: gene-set dot plot (commit 14d6893)
emit("j_module_genes_go", (V7 / "panels_extra/fig3_geneset_dotplot.svg").read_text())
