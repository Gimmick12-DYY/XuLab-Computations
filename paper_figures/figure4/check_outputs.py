"""Check that every figure file in output/ keeps its text editable.

PDF: every font is embedded TrueType Arial (pdffonts, poppler) and the page has extractable text.
SVG: text is stored as <text> elements (no glyph outlines: no <defs> glyph paths, no <use> of glyph ids) and every <text>/<tspan> names Arial
with an explicit font-size (no CSS "font:" shorthand, which Adobe Illustrator misreads).
Writes output/check_outputs.tsv and exits non-zero on any failure.
  python check_outputs.py
"""
import re
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "output"


def pdf_check(p):
    rows = subprocess.run(["pdffonts", str(p)], capture_output=True, text=True, check=True).stdout.splitlines()[2:]
    fonts = [r.split() for r in rows if r.strip()]
    names = sorted({f[0].split("+")[-1] for f in fonts})
    ok_fonts = bool(fonts) and all("TrueType" in " ".join(f) and f[-5] == "yes" for f in fonts) and all(n.startswith("Arial") for n in names)
    txt = subprocess.run(["pdftotext", str(p), "-"], capture_output=True, text=True).stdout
    return ok_fonts and len(txt.strip()) > 0, f"{len(fonts)} fonts ({', '.join(names)}), {len(txt.split())} words"


def svg_check(p):
    s = p.read_text()
    n_text = len(re.findall(r"<text\b", s))
    glyphs = len(re.findall(r'<path id="(?:glyph|[A-Za-z]+-[0-9a-f]+)', s)) + len(re.findall(r'xlink:href="#(?:glyph|DejaVu|Arial)', s))
    shorthand = len(re.findall(r'style="[^"]*\bfont:', s))
    no_arial = sum(1 for t in re.findall(r"<(?:text|tspan)\b[^>]*>", s) if ("font" in t) and 'font-family="Arial' not in t)
    return n_text > 0 and glyphs == 0 and shorthand == 0 and no_arial == 0, f"{n_text} <text>, {glyphs} glyph outlines, {shorthand} CSS shorthand, {no_arial} without Arial"


def main():
    rows, bad = [], 0
    for p in sorted(OUT.rglob("*")):
        if p.suffix not in (".pdf", ".svg"):
            continue
        ok, info = (pdf_check if p.suffix == ".pdf" else svg_check)(p)
        bad += not ok
        rows.append(f"{p.relative_to(OUT)}\t{'ok' if ok else 'FAIL'}\t{info}")
    (OUT / "check_outputs.tsv").write_text("file\tstatus\tdetail\n" + "\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"{len(rows) - bad} of {len(rows)} files pass")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
