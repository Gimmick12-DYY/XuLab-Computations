#!/usr/bin/env python3
# Pack build_pmat_panel.R outputs into the npz that tf_specific_regions.py reads.
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=Path, required=True)
    ap.add_argument("--out-npz", type=Path, required=True)
    args = ap.parse_args()
    prefix = str(args.prefix)
    tfs = [ln.strip() for ln in open(prefix + ".tfs.txt") if ln.strip()]
    peaks = []
    with open(prefix + ".peaks.bed") as fh:
        for ln in fh:
            p = ln.split()
            peaks.append(f"{p[0]}:{p[1]}-{p[2]}")
    n_peaks, n_tfs = len(peaks), len(tfs)
    raw = np.fromfile(prefix + ".ranknorm.f32", dtype="<f4")
    if raw.size != n_peaks * n_tfs:
        raise SystemExit(f"ranknorm.f32 size {raw.size} != {n_peaks}*{n_tfs}")
    # R wrote each TF column in sequence -> Fortran layout (n_peaks, n_tfs)
    ranknorm = np.ascontiguousarray(raw.reshape((n_peaks, n_tfs), order="F"))
    n_cells = np.zeros(n_tfs, dtype=np.int32)
    meta = Path(prefix + ".meta.csv")
    if meta.is_file():
        import csv
        with open(meta) as fh:
            rows = {r["tf"]: int(r["n_cells"]) for r in csv.DictReader(fh)}
        n_cells = np.array([rows.get(t, 0) for t in tfs], dtype=np.int32)
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        prevalence=ranknorm,
        ranknorm=ranknorm,
        bin_idx=np.arange(n_peaks, dtype=np.int32),
        regions=np.array(peaks, dtype="U32"),
        tfs=np.array(tfs, dtype="U32"),
        n_cells=n_cells,
    )
    print(f"[pack] {n_peaks} peaks x {n_tfs} TFs -> {args.out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
