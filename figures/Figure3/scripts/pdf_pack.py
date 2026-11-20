"""Figure 3 editable PDF pack (user 2026-11-18: final panels as editable PDFs for Illustrator).
Each approved SVG (live Arial text, vector shapes) is printed to PDF by headless Google Chrome at its exact page size: text stays real text with
embedded TrueType Arial, shapes stay vector; images already embedded in an SVG (d heatmap cells, g browser tracks, h 3D render) stay raster.
Checks per PDF (pypdf): page size, embedded TrueType fonts, extractable text. Writes Figure3/Figure3_PDF_pack/ + pack_check.tsv.
  python Figure3/scripts/pdf_pack.py   (Mac)"""
import re, subprocess, tempfile
from pathlib import Path
import pandas as pd
from pypdf import PdfReader
H = Path(__file__).resolve().parents[1]; OUT = H / "Figure3_PDF_pack"; (OUT / "supplementary").mkdir(parents=True, exist_ok=True)
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PANELS = ["a_coordinated_binding_schematic", "b1_rbbp4_network", "b2_rbbp4_hic_circos", "b_rbbp4_characterization", "c_rbbp4_chromosomes",
          "d_module_edges_hic_support", "e_module_genomic_class", "f_top_shared_edges_chord", "g_se_browser", "h_se_3d_structure", "i_se_flower",
          "ghi_shared_legend", "j_go_specific_vs_conserved_summary", "hic_ets2_locus", "supplementary/go_per_tf_great_terms", "supplementary/j_gene_set_dotplot"]
SRC = {"supplementary/j_gene_set_dotplot": "j_gene_set_dotplot"}
def size_pt(svg):
    m = re.search(r'<svg[^>]*?\swidth="([\d.]+)(pt|mm|px)?"[^>]*?\sheight="([\d.]+)(pt|mm|px)?"', svg, re.S)
    f = {"pt": 1, None: 0.75, "px": 0.75, "mm": 72 / 25.4}
    return float(m.group(1)) * f[m.group(2)], float(m.group(3)) * f[m.group(4)]
rows = []
for p in PANELS:
    src = H / f"{SRC.get(p, p)}.svg"; svg = src.read_text(); w, h = size_pt(svg)
    body = re.sub(r"^<\?xml[^>]*>\s*", "", svg); body = re.sub(r"<!DOCTYPE[^>]*>\s*", "", body)
    html = (f"<!doctype html><html><head><meta charset='utf-8'><style>@page{{size:{w:.3f}pt {h:.3f}pt;margin:0}}html,body{{margin:0;padding:0}}"
            f"svg{{display:block;width:{w:.3f}pt;height:{h:.3f}pt}}svg text,svg tspan{{font-family:Arial,Helvetica,sans-serif}}</style></head><body>{body}</body></html>")
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, dir=OUT) as t: t.write(html); tp = t.name
    pdf = OUT / f"{p}.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--print-to-pdf-no-header", f"--print-to-pdf={pdf}", f"file://{tp}"],
                   check=True, capture_output=True, timeout=300)
    Path(tp).unlink()
    r = PdfReader(str(pdf)); pg = r.pages[0]; fonts = set(); tt = 0
    res = pg.get("/Resources") or {}
    for k, f in (res.get("/Font") or {}).items():
        f = f.get_object(); fonts.add(str(f.get("/BaseFont"))); d = f.get("/FontDescriptor")
        if d is None and "/DescendantFonts" in f: d = f["/DescendantFonts"][0].get_object().get("/FontDescriptor")
        if d is not None and "/FontFile2" in d.get_object(): tt += 1
    txt = pg.extract_text() or ""
    rows.append(dict(panel=p, pages=len(r.pages), w_pt=round(float(pg.mediabox.width), 1), h_pt=round(float(pg.mediabox.height), 1), svg_w_pt=round(w, 1), svg_h_pt=round(h, 1),
                     fonts=";".join(sorted(fonts)), truetype_embedded=tt, text_chars=len(txt.strip()), raster_images_in_svg=svg.count("<image"), kb=round(pdf.stat().st_size / 1024)))
C = pd.DataFrame(rows); C.to_csv(OUT / "pack_check.tsv", sep="\t", index=False); print(C.drop(columns=["svg_w_pt", "svg_h_pt"]).to_string(index=False))
