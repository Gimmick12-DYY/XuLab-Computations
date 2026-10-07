#!/usr/bin/env python3
"""Rebuild the RBBP4 cell-by-peak matrix from the peak BED and SnapATAC dataset."""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.io
import scipy.sparse as sp
import snapatac2 as snap


def write_lines_gz(path: Path, values) -> None:
    with gzip.open(path, "wt") as out:
        for value in values:
            out.write(f"{value}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True,
                    help="RBBP4 AnnDataSet (.h5ads)")
    ap.add_argument("--anndata-dir", type=Path, required=True,
                    help="directory containing the moved component .h5ad files")
    ap.add_argument("--peaks", type=Path, required=True,
                    help="BED/narrowPeak file; first three columns define peaks")
    ap.add_argument("--out-h5ad", type=Path, required=True)
    ap.add_argument("--out-mm-dir", type=Path, required=True)
    ap.add_argument("--reference-h5ad", type=Path, default=None,
                    help="optional saved Wang peak matrix for exact validation")
    args = ap.parse_args()

    args.out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    args.out_mm_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pmat] SnapATAC2 {snap.__version__}", flush=True)
    print(f"[pmat] dataset={args.dataset}", flush=True)
    print(f"[pmat] peaks={args.peaks}", flush=True)
    dataset = snap.read_dataset(
        str(args.dataset),
        adata_files_update=str(args.anndata_dir),
        mode="r+",
    )
    try:
        peak_mat = snap.pp.make_peak_matrix(
            dataset,
            peak_file=str(args.peaks),
            file=str(args.out_h5ad),
        )
        # Materialize only the sparse peak matrix (~1M nnz for RBBP4).
        mem = peak_mat.to_memory() if hasattr(peak_mat, "to_memory") else ad.read_h5ad(args.out_h5ad)
    finally:
        dataset.close()

    # scipy.io.mmwrite labels uint matrices as "unsigned-integer", which
    # Matrix::readMM does not support. Counts are binary/small, so int32 is safe.
    x = sp.csr_matrix(mem.X).astype(np.int32)
    print(f"[pmat] rebuilt cells={x.shape[0]:,} peaks={x.shape[1]:,} nnz={x.nnz:,}",
          flush=True)

    # Cicero expects peaks x cells.
    scipy.io.mmwrite(args.out_mm_dir / "matrix.mtx", x.T.tocoo())
    with open(args.out_mm_dir / "matrix.mtx", "rb") as src, \
            gzip.open(args.out_mm_dir / "matrix.mtx.gz", "wb") as dst:
        while chunk := src.read(1024 * 1024):
            dst.write(chunk)
    (args.out_mm_dir / "matrix.mtx").unlink()
    write_lines_gz(args.out_mm_dir / "regions.tsv.gz", mem.var_names)
    write_lines_gz(args.out_mm_dir / "barcodes.tsv.gz", mem.obs_names)

    if args.reference_h5ad is not None:
        ref = ad.read_h5ad(args.reference_h5ad)
        y = sp.csr_matrix(ref.X)
        same_shape = x.shape == y.shape
        same_obs = np.array_equal(np.asarray(mem.obs_names), np.asarray(ref.obs_names))
        same_var = np.array_equal(np.asarray(mem.var_names), np.asarray(ref.var_names))
        same_x = same_shape and (x != y).nnz == 0
        report = (
            f"shape_rebuilt={x.shape[0]}x{x.shape[1]}\n"
            f"shape_reference={y.shape[0]}x{y.shape[1]}\n"
            f"nnz_rebuilt={x.nnz}\n"
            f"nnz_reference={y.nnz}\n"
            f"same_obs={same_obs}\n"
            f"same_var={same_var}\n"
            f"same_matrix={same_x}\n"
        )
        report_path = args.out_h5ad.with_suffix(".compare_to_reference.txt")
        report_path.write_text(report)
        print("[pmat] reference comparison:\n" + report.rstrip(), flush=True)
        if not (same_obs and same_var and same_x):
            raise SystemExit("rebuilt peak matrix does not exactly match reference")

    print(f"[pmat] wrote {args.out_h5ad} and {args.out_mm_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
