"""Figure 3 top-row RBBP4 slot (2026-11-20): square candidates B (b2_rbbp4_hic_circos) and C (b1_rbbp4_network) fitted unchanged to a
60 x 60 mm square (vector, centred, aspect kept), A's editable PDF, and top-row mock-ups (a | candidate | network overview | Hi-C/Cicero locus)
composed from the final PNGs at one height. Headless Chrome prints SVG -> PDF (live Arial text) and screenshots SVG -> PNG.
  python Figure3/scripts/rbbp4_square_slot.py   (Mac)"""
import re, subprocess, tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
H = Path(__file__).resolve().parents[1]; O = H / "slot_mockups"; O.mkdir(exist_ok=True)
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"; SQ = 60 * 72 / 25.4     # 60 mm in pt
def html(svg, w, h):
    body = re.sub(r"^<\?xml[^>]*>\s*", "", svg); body = re.sub(r"<!DOCTYPE[^>]*>\s*", "", body)
    return (f"<!doctype html><html><head><meta charset='utf-8'><style>@page{{size:{w:.3f}pt {h:.3f}pt;margin:0}}html,body{{margin:0;padding:0;background:white}}"
            f"body>svg{{display:block;width:{w:.3f}pt;height:{h:.3f}pt}}svg text,svg tspan{{font-family:Arial,Helvetica,sans-serif}}</style></head><body>{body}</body></html>")
def run(svgtext, stem, w=SQ, h=SQ, png=True):
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, dir=O) as t: t.write(html(svgtext, w, h)); tp = t.name
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={O / (stem + '.pdf')}", f"file://{tp}"], check=True, capture_output=True, timeout=300)
    if png:
        px = round(w / 72 * 300); py = round(h / 72 * 300)
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--force-device-scale-factor={300 / 96}", f"--window-size={round(w / 72 * 96)},{round(h / 72 * 96)}",
                        f"--screenshot={O / (stem + '.png')}", f"file://{tp}"], check=True, capture_output=True, timeout=300)
    Path(tp).unlink()
def fit_square(src):   # wrap an SVG unchanged inside a square SVG (aspect kept, centred)
    s = (H / f"{src}.svg").read_text(); s = re.sub(r"^<\?xml[^>]*>\s*", "", s); s = re.sub(r"<!DOCTYPE[^>]*>\s*", "", s)
    s = re.sub(r'(<svg\b[^>]*?)\swidth="[^"]*"', r'\1 width="100%"', s, count=1); s = re.sub(r'(<svg\b[^>]*?)\sheight="[^"]*"', r'\1 height="100%"', s, count=1)
    if "preserveAspectRatio" not in s[:500]: s = s.replace("<svg", '<svg preserveAspectRatio="xMidYMid meet"', 1)
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{SQ:.3f}pt" height="{SQ:.3f}pt" viewBox="0 0 100 100"><foreignObject x="0" y="0" width="100" height="100"></foreignObject></svg>', s
for tag, src in (("B", "b2_rbbp4_hic_circos"),):
    _, inner = fit_square(src); (O / f"rbbp4_square_{tag}.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{SQ:.3f}pt" height="{SQ:.3f}pt">' +
        inner.replace("<svg", f'<svg x="0" y="0"', 1) + "</svg>")
    run((O / f"rbbp4_square_{tag}.svg").read_text(), f"rbbp4_square_{tag}")
run((O / "rbbp4_square_A.svg").read_text(), "rbbp4_square_A", png=False)          # A: PNG comes from matplotlib, PDF printed from its SVG
# ---- top-row mock-ups at one height ----
HT = 700
def at_h(p): im = Image.open(p).convert("RGB"); return im.resize((round(im.width * HT / im.height), HT))
def placeholder(w, text):
    im = Image.new("RGB", (w, HT), (245, 245, 245)); d = ImageDraw.Draw(im)
    try: f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 34)
    except OSError: f = ImageFont.load_default()
    d.rectangle([2, 2, w - 3, HT - 3], outline=(170, 170, 170), width=3); d.multiline_text((w / 2, HT / 2), text, fill=(120, 120, 120), font=f, anchor="mm", align="center"); return im
for tag in ("A", "B"):                                                          # C (b1 network) skipped: labels ~1 pt at 60 mm
    parts = [at_h(H / "a_coordinated_binding_schematic.png"), at_h(O / f"rbbp4_square_{tag}.png"),
             placeholder(round(HT * 1.4), "network overview\nwith zoom insets\n(file not in Figure3)"), at_h(H / "g_se_browser.png")]
    W = sum(p.width for p in parts) + 30 * (len(parts) + 1); row = Image.new("RGB", (W, HT + 60), "white"); x = 30
    for p in parts: row.paste(p, (x, 30)); x += p.width + 30
    row.save(O / f"row_option_{tag}.png")
print("done")
