#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# export_tf_cicero_mm.py
#
# Build a peak x cell Matrix Market for Cicero from a TF's raw HiTAG mm and
# called peaks. Each peak is one Cicero site (chr_start_end); its counts are
# the sum of overlapping 1 kb bins. This is the missing first step the
# cobinding parser never had — it only read precomputed TF.*.fitConns files.
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from build_pairs_from_matrix import (  # noqa: E402
    _COORD,
    chrom_index,
    load_peaks,
    read_lines,
)


def overlapping_bins(chrom, s, e, idx) -> list[int]:
    if chrom not in idx:
        return []
    ids, st, en = idx[chrom]
    j = int(np.searchsorted(st, e, side="left"))
    hit = []
    k = j - 1
    while k >= 0 and en[k] > s:
        if min(e, int(en[k])) - max(s, int(st[k])) > 0:
            hit.append(int(ids[k]))
        k -= 1
    k = j
    while k < len(st) and st[k] < e:
        if min(e, int(en[k])) - max(s, int(st[k])) > 0:
            hit.append(int(ids[k]))
        k += 1
    return hit


def resolve_regions(mm_dir: Path, unified_tf: Path, fallback: Path) -> Path:
    for p in (mm_dir / "regions.tsv.gz", mm_dir / "regions.tsv",
              unified_tf / "impute" / "regions.tsv", fallback):
        if p.is_file():
            return p
    raise SystemExit(f"no regions.tsv for {mm_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--peaks", type=Path, required=True)
    ap.add_argument("--mm-dir", type=Path, required=True,
                    help="unified/work/<tf>/mm (raw HiTAG)")
    ap.add_argument("--out-mm", type=Path, required=True,
                    help="Cicero work_dir/mm (matrix.mtx.gz + regions + barcodes)")
    ap.add_argument("--regions-fallback", type=Path, default=None)
    ap.add_argument("--max-peaks", type=int, default=20_000)
    args = ap.parse_args()

    peaks = load_peaks(args.peaks, args.max_peaks)
    mtx_path = args.mm_dir / "matrix.mtx.gz"
    if not mtx_path.is_file():
        raise SystemExit(f"missing {mtx_path}")
    unified_tf = args.mm_dir.parent
    fb = args.regions_fallback or Path("/work/users/d/y/dyy12/XuLab/unified/work/ctcf/mm/regions.tsv.gz")
    reg_path = resolve_regions(args.mm_dir, unified_tf, fb)
    print(f"[{args.tf}] peaks={len(peaks)} mm={mtx_path} regions={reg_path}", flush=True)

    names = read_lines(reg_path)
    n = len(names)
    chroms = np.empty(n, dtype=object)
    starts = np.zeros(n, dtype=np.int64)
    ends = np.zeros(n, dtype=np.int64)
    for i, nm in enumerate(names):
        m = _COORD.match(nm.strip())
        if not m:
            raise SystemExit(f"bad region {nm!r}")
        chroms[i], starts[i], ends[i] = m.group(1), int(m.group(2)), int(m.group(3))

    print(f"[{args.tf}] reading {mtx_path} ...", flush=True)
    X = sio.mmread(str(mtx_path)).tocsr()
    if X.shape[0] != len(names):
        raise SystemExit(f"matrix rows {X.shape[0]} != regions {len(names)}")
    idx = chrom_index(chroms, starts, ends)

    bar_path = args.mm_dir / "barcodes.tsv.gz"
    if not bar_path.is_file():
        bar_path = args.mm_dir / "barcodes.tsv"
    barcodes = read_lines(bar_path)

    data, indices, indptr = [], [], [0]
    kept_names = []
    n_empty = 0
    for chrom, s, e, _sc, name in peaks:
        bins = overlapping_bins(chrom, s, e, idx)
        if not bins:
            n_empty += 1
            continue
        vec = X[bins].sum(axis=0)
        vec = np.asarray(vec).ravel()
        nz = np.flatnonzero(vec)
        if nz.size == 0:
            n_empty += 1
            continue
        data.append(vec[nz].astype(np.float32, copy=False))
        indices.append(nz.astype(np.int32, copy=False))
        indptr.append(indptr[-1] + nz.size)
        kept_names.append(name)
    if not kept_names:
        raise SystemExit("no peaks overlap the raw matrix")
    data = np.concatenate(data)
    indices = np.concatenate(indices)
    indptr = np.asarray(indptr, dtype=np.int32)
    P = sp.csr_matrix((data, indices, indptr), shape=(len(kept_names), X.shape[1]))
    keep_cells = np.asarray(P.sum(axis=0)).ravel() > 0
    n_drop_cells = int((~keep_cells).sum())
    if n_drop_cells:
        P = P[:, keep_cells]
        barcodes = [b for b, k in zip(barcodes, keep_cells) if k]
    print(f"[{args.tf}] Cicero mm: {P.shape[0]} peaks x {P.shape[1]} cells  "
          f"nnz={P.nnz:,}  dropped_empty_peaks={n_empty}  "
          f"dropped_empty_cells={n_drop_cells}", flush=True)

    args.out_mm.mkdir(parents=True, exist_ok=True)
    tmp = args.out_mm / "matrix.mtx"
    sio.mmwrite(str(tmp), P)
    with open(tmp, "rb") as src, gzip.open(args.out_mm / "matrix.mtx.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    tmp.unlink()
    with gzip.open(args.out_mm / "regions.tsv.gz", "wt") as fh:
        fh.write("\n".join(kept_names) + "\n")
    with gzip.open(args.out_mm / "barcodes.tsv.gz", "wt") as fh:
        fh.write("\n".join(barcodes) + "\n")
    print(f"[done] wrote {args.out_mm}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
