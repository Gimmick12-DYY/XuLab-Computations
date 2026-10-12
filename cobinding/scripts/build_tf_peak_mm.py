#!/usr/bin/env python3
"""Build one TF's Cicero matrix from its TAGATG narrowPeak and its Cell1000 cells.

The fragment dataset is the shared TF1000cells AnnDataSet. Cells are the
rows of Final_DNA_cells_TF1000cells_qc20_ForSubset.csv for this TF. Peak
intervals are the TF's q005 narrowPeak. The written 10x-style matrix is
peaks x cells, which is what 02_build_cds.R reads.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import shutil
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
    ap.add_argument("--tf", required=True)
    ap.add_argument("--cells-csv", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True,
                    help="TF1000cells _dataset.h5ads (read-only source)")
    ap.add_argument("--anndata-dir", type=Path, required=True)
    ap.add_argument("--peaks", type=Path, required=True)
    ap.add_argument("--out-h5ad", type=Path, required=True)
    ap.add_argument("--out-mm-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True,
                    help="scratch for the dataset copy and the per-TF fragment file")
    args = ap.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    args.out_mm_dir.mkdir(parents=True, exist_ok=True)

    want = set()
    with args.cells_csv.open() as handle:
        for row in csv.DictReader(handle):
            if row["TF"] == args.tf:
                want.add(f"{row['DNA_id']}:{row['cell']}")
    if not want:
        raise SystemExit(f"no cells for {args.tf} in {args.cells_csv}")
    print(f"[pmat] {args.tf} cells in table {len(want):,}", flush=True)

    local_ds = args.work_dir / "_dataset.h5ads"
    if not local_ds.exists():
        shutil.copyfile(args.dataset, local_ds)
    print(f"[pmat] SnapATAC2 {snap.__version__}", flush=True)
    dataset = snap.read_dataset(
        str(local_ds),
        adata_files_update=str(args.anndata_dir),
        mode="r+",
    )
    try:
        names = list(dataset.obs_names)
        keys = [args.tf if name in want else None for name in names]
        matched = sum(key is not None for key in keys)
        print(f"[pmat] matched {matched:,} / {len(want):,} cells in the fragment dataset",
              flush=True)
        if matched == 0:
            raise SystemExit(f"no fragment cells matched {args.tf}")
        frag_dir = args.work_dir / "fragments"
        frag_dir.mkdir(parents=True, exist_ok=True)
        parts = dataset.split_obs_by(keys, out_dir=str(frag_dir))
        tf_adata = parts[args.tf]
        print(f"[pmat] split object {type(tf_adata)} n_obs={getattr(tf_adata, 'n_obs', '?')}",
              flush=True)
        peak_mat = snap.pp.make_peak_matrix(
            tf_adata,
            peak_file=str(args.peaks),
            file=str(args.out_h5ad),
        )
        mem = peak_mat.to_memory() if hasattr(peak_mat, "to_memory") else ad.read_h5ad(args.out_h5ad)
    finally:
        dataset.close()

    x = sp.csr_matrix(mem.X).astype(np.int32)
    print(f"[pmat] cells={x.shape[0]:,} peaks={x.shape[1]:,} nnz={x.nnz:,}", flush=True)
    if x.shape[1] == 0 or x.nnz == 0:
        raise SystemExit(f"{args.tf} peak matrix is empty")

    scipy.io.mmwrite(args.out_mm_dir / "matrix.mtx", x.T.tocoo())
    with open(args.out_mm_dir / "matrix.mtx", "rb") as src, \
            gzip.open(args.out_mm_dir / "matrix.mtx.gz", "wb") as dst:
        while chunk := src.read(1024 * 1024):
            dst.write(chunk)
    (args.out_mm_dir / "matrix.mtx").unlink()
    write_lines_gz(args.out_mm_dir / "regions.tsv.gz", mem.var_names)
    write_lines_gz(args.out_mm_dir / "barcodes.tsv.gz", mem.obs_names)

    # The fragment extract is only an input to the peak matrix.
    frag_dir = args.work_dir / "fragments"
    if frag_dir.exists():
        shutil.rmtree(frag_dir)
    print(f"[pmat] wrote {args.out_mm_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
