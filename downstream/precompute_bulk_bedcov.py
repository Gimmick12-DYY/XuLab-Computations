#!/usr/bin/env python3
"""Precompute genome-wide samtools bedcov for one TF bulk BAM → .npy cache.

Usage:
  python3 downstream/precompute_bulk_bedcov.py --tf ctcf
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

BULK_BAM = {
    "ctcf": REPO / "data" / "ENCFF139DCW.bam",
    "maz": REPO / "data" / "MAZ_HEK293.bam",
    "zic2": REPO / "data" / "ZIC2_HEK293.bam",
    "zbtb7a": REPO / "data" / "ZBTB7A_HEK293.bam",
    "znf777": REPO / "data" / "ZNF777_HEK293.bam",
    "nfya": REPO / "data" / "NFYA_K562.bam",
    "znf282": REPO / "data" / "ZNF282_HEK293.bam",
}

SAMTOOLS = "samtools"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", required=True, choices=sorted(BULK_BAM))
    ap.add_argument(
        "--regions-bed",
        type=Path,
        default=ROOT / "plots" / "tf_bulk_pearson" / "_cache" / "regions_1kb.bed",
    )
    ap.add_argument(
        "--out-npy",
        type=Path,
        default=None,
        help="Default: .../_cache/bulk_bedcov_<tf>.npy",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = args.out_npy or (
        ROOT / "plots" / "tf_bulk_pearson" / "_cache" / f"bulk_bedcov_{args.tf}.npy"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and not args.force:
        arr = np.load(out)
        print(f"[skip] {out} exists (n={arr.size})")
        return 0

    bam = BULK_BAM[args.tf]
    if not bam.is_file():
        raise SystemExit(f"missing BAM {bam}")
    if not args.regions_bed.is_file():
        raise SystemExit(f"missing regions bed {args.regions_bed}")

    # Count regions without loading whole file into memory twice.
    print(f"[count] {args.regions_bed}", flush=True)
    n = 0
    with args.regions_bed.open() as fh:
        for _ in fh:
            n += 1
    print(f"[bedcov] tf={args.tf} bam={bam.name} n={n}", flush=True)

    cov = np.zeros(n, dtype=np.float64)
    proc = subprocess.Popen(
        [SAMTOOLS, "bedcov", "-Q", "0", str(args.regions_bed), str(bam)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1 << 20,
    )
    assert proc.stdout is not None
    for i, line in enumerate(proc.stdout):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 5 and parts[3].isdigit():
            bi = int(parts[3])
        else:
            bi = i
        cov[bi] = float(parts[-1])
        if (i + 1) % 500000 == 0:
            print(f"  ... {i + 1}/{n}", flush=True)
    err = proc.stderr.read() if proc.stderr else ""
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"bedcov failed rc={rc}: {err[:800]}")
    np.save(out, cov)
    nz = float((cov > 0).mean())
    print(
        f"[done] wrote {out}  nz={nz:.3f}  median={np.median(cov):.1f}  max={cov.max():.0f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
