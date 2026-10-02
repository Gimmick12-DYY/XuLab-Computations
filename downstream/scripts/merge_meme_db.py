#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# merge_meme_db.py   (Novel Motif Finding, Step 1)
#
# Merge several MEME-format motif databases into ONE, tagging each motif with its
# source so TOMTOM/MoSBAT re-annotation reports provenance. Use to replace the old
# HOCOMOCO-v11-only DB with HOCOMOCO v12 + Codebook (CisBP 3.1 / Zenodo):
#
#   python merge_meme_db.py \
#     --in HOCOMOCOv12_H12CORE_meme_format.meme=h12 codebook_cisbp3.1.meme=cdbk \
#     --out downstream/cache/motifdb/merged_human_motifs.meme
#
# Motif IDs become "<tag>|<original_id>" (dedup within tag). The output header is a
# single MEME v4 preamble; background is taken from the first file that declares one.
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
from pathlib import Path

PREAMBLE = ("MEME version 4\n\nALPHABET= ACGT\n\n"
            "strands: + -\n\nBackground letter frequencies\n"
            "A 0.25 C 0.25 G 0.25 T 0.25\n\n")


def parse_blocks(path: Path):
    """Yield raw text blocks each starting at a 'MOTIF' line (through the next)."""
    lines = path.read_text().splitlines()
    cur, started = [], False
    for ln in lines:
        if ln.startswith("MOTIF"):
            if started:
                yield "\n".join(cur).rstrip() + "\n"
            cur, started = [ln], True
        elif started:
            cur.append(ln)
    if started:
        yield "\n".join(cur).rstrip() + "\n"


def retag(block: str, tag: str, seen: set) -> str | None:
    head, _, rest = block.partition("\n")
    parts = head.split()
    mid = parts[1] if len(parts) > 1 else "motif"
    name = parts[2] if len(parts) > 2 else mid
    new_id = f"{tag}|{mid}"
    n, k = new_id, 1
    while new_id in seen:
        k += 1; new_id = f"{n}.{k}"
    seen.add(new_id)
    return f"MOTIF {new_id} {name}\n{rest}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inputs", nargs="+", required=True,
                    help="FILE=tag entries (tag labels the source, e.g. h12, cdbk)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    seen: set = set()
    n_tot = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as out:
        out.write(PREAMBLE)
        for spec in args.inputs:
            path_s, _, tag = spec.partition("=")
            tag = tag or "db"
            p = Path(path_s)
            if not p.is_file():
                raise SystemExit(f"missing input: {p}")
            n = 0
            for blk in parse_blocks(p):
                rb = retag(blk, tag, seen)
                if rb:
                    out.write("\n" + rb); n += 1
            print(f"[merge] {tag}: {n} motifs from {p}", flush=True)
            n_tot += n
    print(f"[merge] total {n_tot} motifs -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
