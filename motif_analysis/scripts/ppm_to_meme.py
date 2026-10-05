#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# ppm_to_meme.py   (Novel Motif Finding, Step 1 — Codebook DB)
#
# Convert Codebook (MEX) per-motif .ppm/.pcm files into one MEME-format DB for
# TOMTOM re-annotation. Each file: a '>name' header then rows of 4 numbers
# (A C G T), one per position. Names encode the TF as the leading token, e.g.
#   AHCTF1.DBD@PBM.ME@...   -> AHCTF1
#   CTCF.NA@AFS.Lys@...     -> CTCF
# .pcm = counts (row-normalized here); .ppm = probabilities.
#
#   python ppm_to_meme.py --in-dir MEX_top1/ --out codebook_top1.meme
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import re
from pathlib import Path

PREAMBLE = ("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n"
            "Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")


def tf_symbol(name: str) -> str:
    base = name.lstrip(">").strip()
    # leading token before the first '.' is the gene symbol (AHCTF1.DBD@... -> AHCTF1)
    return re.split(r"[.@\s]", base)[0] or base


def read_matrix(path: Path):
    name = path.stem
    rows = []
    with open(path) as fh:
        for ln in fh:
            t = ln.strip()
            if not t:
                continue
            if t.startswith(">"):
                name = t[1:].strip()
                continue
            vals = t.replace(",", " ").split()
            if len(vals) >= 4:
                try:
                    rows.append([float(x) for x in vals[:4]])
                except ValueError:
                    pass
    return name, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, required=True, help="dir of .ppm/.pcm files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--glob", default="*.ppm", help="file glob (default *.ppm; use '*' for ppm+pcm)")
    args = ap.parse_args()

    files = sorted(args.in_dir.rglob(args.glob))
    if not files:
        files = sorted(p for p in args.in_dir.rglob("*") if p.suffix in (".ppm", ".pcm"))
    if not files:
        raise SystemExit(f"no .ppm/.pcm under {args.in_dir}")

    seen: dict[str, int] = {}
    n = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as out:
        out.write(PREAMBLE)
        for f in files:
            name, rows = read_matrix(f)
            if not rows:
                continue
            tf = tf_symbol(name)
            seen[tf] = seen.get(tf, 0) + 1
            mid = tf if seen[tf] == 1 else f"{tf}.{seen[tf]}"
            w = len(rows)
            out.write(f"MOTIF {mid} {name}\n")
            out.write(f"letter-probability matrix: alength= 4 w= {w} nsites= 20\n")
            for r in rows:
                s = sum(r) or 1.0
                p = [max(min(v / s, 1.0), 0.0) for v in r]   # row-normalize (handles .pcm counts)
                out.write("  ".join(f"{v:.6f}" for v in p) + "\n")
            out.write("\n")
            n += 1
    print(f"[ppm2meme] {n} motifs ({len(seen)} TFs) -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
