#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Unique peak intervals from a lab Cicero fitConns file, as BED.
# Used to rebuild Cicero on the same sites as TF.RBBP4.fitConns.res.sel
# (current MACS rbbp4.bed has 0 exact coordinate matches).
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import re
from pathlib import Path

_COLON = re.compile(r"(chr[0-9A-Za-z]+):(\d+)-(\d+)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fitconns", type=Path, required=True)
    ap.add_argument("--out-bed", type=Path, required=True)
    args = ap.parse_args()

    peaks: dict[tuple[str, int, int], None] = {}
    with args.fitconns.open() as fh:
        for ln in fh:
            for m in _COLON.finditer(ln):
                peaks[(m.group(1), int(m.group(2)), int(m.group(3)))] = None

    args.out_bed.parent.mkdir(parents=True, exist_ok=True)
    with args.out_bed.open("w") as out:
        for chrom, s, e in sorted(peaks):
            out.write(f"{chrom}\t{s}\t{e}\t{chrom}:{s}-{e}\t0\t.\n")
    print(f"[done] {len(peaks):,} unique peaks -> {args.out_bed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
