#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# homer_motifs_to_meme.py
#
# Convert HOMER de novo output (homerMotifs.all.motifs) to MEME format so the
# motifs can be run through TOMTOM against HOCOMOCO/JASPAR. HOMER's own
# "BestGuess" only reports its top match from HOMER's library; TOMTOM against
# the merged database gives a match with a q-value, and -- more usefully here --
# tells us when a de novo motif matches *nothing*, which is what a genuinely
# novel motif for an uncharacterized TF would look like.
#
# HOMER motif block:
#   >CONSENSUS<TAB>name<TAB>logodds<TAB>logP<TAB>...
#   <A> <C> <G> <T>     (one row per position, probabilities)
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import re
from pathlib import Path

HEADER = """MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

"""


def parse_homer(path: Path) -> list[tuple[str, str, list[list[float]]]]:
    motifs: list[tuple[str, str, list[list[float]]]] = []
    name = consensus = None
    rows: list[list[float]] = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if ln.startswith(">"):
                if name is not None and rows:
                    motifs.append((name, consensus or "", rows))
                parts = ln[1:].split("\t")
                consensus = parts[0]
                name = parts[1] if len(parts) > 1 else consensus
                rows = []
            elif ln.strip():
                vals = [float(x) for x in ln.split()]
                if len(vals) == 4:
                    rows.append(vals)
    if name is not None and rows:
        motifs.append((name, consensus or "", rows))
    return motifs


def safe_id(name: str, consensus: str, i: int) -> str:
    """HOMER de novo names are long and full of separators TOMTOM dislikes."""
    m = re.search(r"BestGuess:([^/\t]+)", name)
    guess = m.group(1) if m else ""
    tok = re.sub(r"[^A-Za-z0-9]+", "_", guess)[:24].strip("_")
    return f"denovo{i}_{consensus}" + (f"_{tok}" if tok else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--homer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    motifs = parse_homer(args.homer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(HEADER)
        for i, (name, consensus, rows) in enumerate(motifs, 1):
            fh.write(f"MOTIF {safe_id(name, consensus, i)} {consensus}\n")
            fh.write(f"letter-probability matrix: alength= 4 w= {len(rows)} "
                     f"nsites= 20 E= 0\n")
            for r in rows:
                tot = sum(r) or 1.0
                fh.write(" " + " ".join(f"{v / tot:.6f}" for v in r) + "\n")
            fh.write("\n")
    print(f"[homer2meme] {len(motifs)} motifs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
