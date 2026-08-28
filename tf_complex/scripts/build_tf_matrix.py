#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# build_tf_matrix.py
#
# Build a bins x TFs RAW-binding pseudobulk matrix for TF co-occupancy analysis.
#
# RAW (not imputed): per-TF pseudobulk = sum across cells of each TF's mm matrix
# (unified/work/<tf>/mm/matrix.mtx.gz). The imputed matrices are open-chromatin
# masked, which would force every TF into the same open bins and inflate
# co-occupancy -- so co-occupancy MUST use raw. All TFs share the 1 kb bin universe.
#
# Output (--out-dir):
#   tf_matrix.npz   scipy CSC, bins x TFs (raw per-bin cell-sum)
#   tfs.txt         TF column order
#   regions.tsv     the shared bin universe (chr:start-end per row)
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


def discover_tfs(work_root: Path) -> list[str]:
    return sorted(d.name for d in work_root.iterdir()
                  if d.is_dir() and (d / "mm" / "matrix.mtx.gz").is_file())


def read_regions(path: Path) -> list[str]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        return [ln.strip() for ln in f if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-root", type=Path,
                    default=Path(__file__).resolve().parents[2] / "unified" / "work")
    ap.add_argument("--tfs", nargs="*", default=None,
                    help="lowercase TF keys; default = every <tf>/mm/matrix.mtx.gz")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    tfs = args.tfs or discover_tfs(args.work_root)
    if not tfs:
        raise SystemExit(f"no <tf>/mm/matrix.mtx.gz under {args.work_root}")

    cols: list[sp.csc_matrix] = []
    used: list[str] = []
    regions: list[str] | None = None
    n_bins = 0
    for tf in tfs:
        mm = args.work_root / tf / "mm"
        mtx = mm / "matrix.mtx.gz"
        if not mtx.is_file():
            print(f"[skip] {tf}: no {mtx}", file=sys.stderr)
            continue
        mat = sio.mmread(str(mtx)).tocsr()
        pb = np.asarray(mat.sum(axis=1)).ravel().astype(np.float32)   # per-bin cell-sum
        if regions is None:
            regions = read_regions(mm / "regions.tsv.gz")
            n_bins = len(regions)
            if pb.shape[0] != n_bins:
                raise SystemExit(f"{tf}: matrix bins {pb.shape[0]} != regions {n_bins}")
        elif pb.shape[0] != n_bins:
            raise SystemExit(f"{tf}: bins {pb.shape[0]} != universe {n_bins}")
        cols.append(sp.csc_matrix(pb.reshape(-1, 1)))
        used.append(tf)
        print(f"[{len(used)}] {tf}: nonzero_bins={int((pb > 0).sum()):,} "
              f"total={pb.sum():.0f}", flush=True)

    if not used:
        raise SystemExit("no TF matrices loaded")
    M = sp.hstack(cols).tocsc()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(args.out_dir / "tf_matrix.npz", M)
    (args.out_dir / "tfs.txt").write_text("\n".join(used) + "\n")
    (args.out_dir / "regions.tsv").write_text("\n".join(regions) + "\n")
    print(f"\n[done] matrix {M.shape[0]:,} bins x {M.shape[1]} TFs "
          f"(nnz={M.nnz:,}) -> {args.out_dir/'tf_matrix.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
