"""Fig 3 v7 export check: render Fig3_v7.svg with cairosvg (independent of the rsvg-convert that writes the PDF) and Fig3_v7.pdf with
pdftoppm, both at 150 dpi; write checks/svg_vs_pdf.png (SVG | PDF | difference) and report the mean pixel difference and whether any
ink touches the outer 0.5 mm of the 183 mm artboard.   PYTHONPATH=checks/pylib python checks/svg_vs_pdf_check.py"""
import os, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageChops
import cairosvg
H = Path(__file__).resolve().parents[1]; C = H / "checks"
os.environ["FONTCONFIG_FILE"] = str(H / "fonts.conf")
cairosvg.svg2png(url=str(H / "Fig3_v7.svg"), write_to=str(C / "_svg.png"), dpi=150, background_color="white")
subprocess.run(["pdftoppm", "-r", "150", "-png", "-singlefile", str(H / "Fig3_v7.pdf"), str(C / "_pdf")], check=True)
a = Image.open(C / "_svg.png").convert("RGB"); b = Image.open(C / "_pdf.png").convert("RGB").resize(a.size)
d = ImageChops.difference(a, b).convert("L"); arr = np.asarray(d); print("size", a.size, "mean |diff|", round(arr.mean(), 3), "pixels >64:", round((arr > 64).mean() * 100, 3), "%")
g = np.asarray(a.convert("L")); edge = int(round(0.5 / 25.4 * 150)); print("ink in outer 0.5 mm: left", (g[:, :edge] < 200).any(), "right", (g[:, -edge:] < 200).any(), "top", (g[:edge] < 200).any(), "bottom", (g[-edge:] < 200).any())
W, Hh = a.size; out = Image.new("RGB", (W * 3 + 40, Hh), "white"); out.paste(a, (0, 0)); out.paste(b, (W + 20, 0)); out.paste(Image.eval(d, lambda v: 255 - min(255, v * 4)).convert("RGB"), (2 * W + 40, 0))
out.save(C / "svg_vs_pdf.png"); [p.unlink() for p in (C / "_svg.png", C / "_pdf.png")]
