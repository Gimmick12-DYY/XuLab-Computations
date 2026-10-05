#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# merge_meme_dbs.py
#
# Merge several MEME-format motif databases into one file for TOMTOM/SEA.
# (MEME 5.5.7 on this cluster ships without meme2meme.)
#
# Every motif is renamed <SOURCE>__<original id> so that a TOMTOM hit says which
# database it came from, and the gene symbol is carried into the alt-name field
# so downstream matching on TF symbol works uniformly across HOCOMOCO's
# 'ANDR.H12CORE.0.P.B' and JASPAR's 'MA0007.3 Ar' conventions.
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


def symbol_of(mid: str, alt: str) -> str:
    """Best-guess gene symbol from either MEME naming convention."""
    for cand in (alt, mid):
        if not cand:
            continue
        tok = cand.split(".")[0].split("::")[0].upper()
        if tok and not re.fullmatch(r"MA\d+", tok):
            return tok
    return (alt or mid).upper()


def parse(path: Path):
    """Yield (motif_id, alt_name, [lines of the motif body])."""
    mid = alt = None
    body: list[str] = []
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("MOTIF"):
                if mid is not None:
                    yield mid, alt, body
                parts = ln.split()
                mid = parts[1]
                alt = parts[2] if len(parts) > 2 else ""
                body = []
            elif mid is not None:
                if ln.startswith(("URL", "MEME version", "ALPHABET", "strands:",
                                  "Background letter")):
                    continue
                body.append(ln.rstrip("\n"))
    if mid is not None:
        yield mid, alt, body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", nargs="+", required=True,
                    help="SOURCE=path/to/db.meme")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    symbols: set[str] = set()
    with open(args.out, "w") as fh:
        fh.write(HEADER)
        for spec in args.db:
            source, _, path = spec.partition("=")
            n = 0
            for mid, alt, body in parse(Path(path)):
                sym = symbol_of(mid, alt)
                symbols.add(sym)
                fh.write(f"MOTIF {source}__{mid} {sym}\n")
                fh.write("\n".join(body).strip("\n") + "\n\n")
                n += 1
            print(f"[merge] {source}: {n} motifs from {path}")
            n_total += n
    print(f"[merge] {n_total} motifs, {len(symbols)} unique symbols -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
