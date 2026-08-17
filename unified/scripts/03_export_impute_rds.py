#!/usr/bin/env python3
"""Export unified impute/matrix_csr.npz to a Kangli-style dgCMatrix .rds.

Writes a temporary Matrix Market file, then calls R (Matrix::readMM + saveRDS)
so the object matches CTCF_bin1000_mtx.rds:
  - dgCMatrix / CsparseMatrix
  - rownames = chr:start-end (from regions.tsv)
  - colnames = barcodes (from barcodes.tsv)

Usage:
  python unified/scripts/03_export_impute_rds.py \\
    --impute-dir unified/work/ctcf/impute \\
    --out data/share/CTCF_unified_imputed.rds

  # Or convert the four TFs for Kangli:
  for tf in ctcf maz nfya znf282; do
    python unified/scripts/03_export_impute_rds.py \\
      --impute-dir unified/work/$tf/impute \\
      --out data/share/${tf^^}_unified_imputed.rds
  done
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impute-dir", type=Path, required=True,
                    help="unified/work/<tf>/impute (matrix_csr.npz + regions.tsv + barcodes.tsv)")
    ap.add_argument("--out", type=Path, required=True, help="Output .rds path")
    ap.add_argument("--rscript", default="Rscript", help="Rscript executable")
    args = ap.parse_args()

    imp = args.impute_dir
    csr_path = imp / "matrix_csr.npz"
    reg_path = imp / "regions.tsv"
    bc_path = imp / "barcodes.tsv"
    for p in (csr_path, reg_path, bc_path):
        if not p.is_file():
            sys.exit(f"missing {p}")

    print(f"[export] loading {csr_path}", flush=True)
    mat = sp.load_npz(csr_path).tocsr()
    regions = [ln.strip() for ln in reg_path.read_text().splitlines() if ln.strip()]
    barcodes = [ln.strip() for ln in bc_path.read_text().splitlines() if ln.strip()]
    if mat.shape[0] != len(regions):
        sys.exit(f"row mismatch: matrix {mat.shape[0]} vs regions {len(regions)}")
    if mat.shape[1] != len(barcodes):
        sys.exit(f"col mismatch: matrix {mat.shape[1]} vs barcodes {len(barcodes)}")
    print(f"[export] shape={mat.shape} nnz={mat.nnz}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="impute_rds_") as tmp:
        tmp_p = Path(tmp)
        mtx = tmp_p / "matrix.mtx"
        regs = tmp_p / "regions.tsv"
        bcs = tmp_p / "barcodes.tsv"
        print(f"[export] writing Matrix Market -> {mtx}", flush=True)
        # Coordinate form is what readMM expects; scipy mmwrite handles CSR.
        sio.mmwrite(str(mtx), mat.astype(np.float64))
        regs.write_text("\n".join(regions) + "\n")
        bcs.write_text("\n".join(barcodes) + "\n")

        r_code = f"""
suppressPackageStartupMessages(library(Matrix))
m <- Matrix::readMM('{mtx}')
m <- as(m, 'CsparseMatrix')
rownames(m) <- readLines('{regs}')
colnames(m) <- readLines('{bcs}')
message(sprintf('[export] R: %d x %d dgCMatrix, nnz=%d', nrow(m), ncol(m), length(m@x)))
saveRDS(m, '{args.out.resolve()}')
message('[export] Wrote {args.out.resolve()}')
"""
        print(f"[export] assembling dgCMatrix via R -> {args.out}", flush=True)
        proc = subprocess.run(
            [args.rscript, "-e", r_code],
            check=False,
            capture_output=True,
            text=True,
        )
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            sys.exit(f"Rscript failed with exit {proc.returncode}")

    print("[export] done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
