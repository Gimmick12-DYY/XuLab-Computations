#!/usr/bin/env python3
"""Print one TF id per line: every unified TF with peaks, plus any fitConns TFs."""
from __future__ import annotations

from pathlib import Path

XULAB = Path("/work/users/d/y/dyy12/XuLab")
UNIFIED = XULAB / "unified" / "work"
PEAKS_IMP = XULAB / "tf_complex" / "work" / "peaks_imputed"
PEAKS_RAW = XULAB / "tf_complex" / "work" / "peaks"
DATA = XULAB / "data"


def main() -> int:
    tfs: set[str] = set()
    if UNIFIED.is_dir():
        for d in UNIFIED.iterdir():
            if not d.is_dir():
                continue
            if not ((d / "impute" / "matrix_csr.npz").is_file()
                    or (d / "mm" / "matrix.mtx.gz").is_file()):
                continue
            tf = d.name
            if (PEAKS_IMP / f"{tf}.bed").is_file() or (PEAKS_RAW / f"{tf}.bed").is_file():
                tfs.add(tf.upper())
    for p in DATA.glob("TF.*.fitConns.res.*"):
        # TF.RBBP4.fitConns.res.sel
        parts = p.name.split(".")
        if len(parts) >= 2:
            tfs.add(parts[1].upper())
    for tf in sorted(tfs):
        print(tf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
