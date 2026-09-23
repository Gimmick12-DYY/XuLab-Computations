#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Replace the ATAC-union open-chromatin mask with the consensus-ChIP ATAC CPM
# 5% cutoff: imputed-only entries are kept only in 1 kb bins with mean ATAC
# CPM > 0.078 (5th percentile of the 38,102 CTCF consensus peaks).
#
# Source matrix: unified/work/ctcf/impute/matrix_csr.npz.pre_openmask
# Output:        unified/work/ctcf/impute_atac_cpm_drop5/   (does not touch impute/)
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
DT_PY = Path(
    "/nas/longleaf/rhel9/apps/deeptools/3.5.6/miniconda3/envs/deeptools/bin/python"
)


def _ensure():
    try:
        import pyBigWig  # noqa: F401
        return
    except ImportError:
        pass
    import os
    if DT_PY.is_file() and Path(sys.executable).resolve() != DT_PY.resolve():
        os.execv(str(DT_PY), [str(DT_PY), *sys.argv])
    raise SystemExit("pyBigWig required")


_ensure()
import pyBigWig  # noqa: E402


def read_lines(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip()]


def parse_regions(names):
    n = len(names)
    chroms = np.empty(n, dtype=object)
    starts = np.zeros(n, dtype=np.int64)
    ends = np.zeros(n, dtype=np.int64)
    for i, name in enumerate(names):
        c, rest = name.split(":", 1)
        a, b = rest.split("-")
        chroms[i] = c
        starts[i] = int(a)
        ends[i] = int(b)
    return chroms, starts, ends


def score_bw(bw_path: Path, chroms, starts, ends):
    bw = pyBigWig.open(str(bw_path))
    chrom_lens = bw.chroms() or {}
    out = np.zeros(len(chroms), dtype=np.float64)
    for c in dict.fromkeys(chroms.tolist()):
        idx = np.flatnonzero(chroms == c)
        if c not in chrom_lens:
            continue
        clen = int(chrom_lens[c])
        n_full = clen // 1000
        if n_full > 0:
            vals = bw.stats(c, 0, n_full * 1000, nBins=n_full, type="mean")
            arr = np.array([0.0 if v is None else max(float(v), 0.0) for v in vals],
                           dtype=np.float64)
        else:
            arr = np.zeros(0, dtype=np.float64)
        s = starts[idx]
        e = ends[idx]
        k = s // 1000
        full = (s % 1000 == 0) & (e == np.minimum(s + 1000, clen)) & (k >= 0) & (k < n_full)
        out[idx[full]] = arr[k[full]]
        for j in idx[~full]:
            a, b = int(starts[j]), int(ends[j])
            b = min(b, clen)
            if b <= a:
                continue
            v = bw.stats(c, a, b, type="mean")
            if v and v[0] is not None:
                out[j] = max(float(v[0]), 0.0)
    bw.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=float, default=0.0781542,
                    help="ATAC CPM floor (default = 5th pct of CTCF consensus peaks)")
    ap.add_argument(
        "--pre-openmask", type=Path,
        default=ROOT / "unified/work/ctcf/impute/matrix_csr.npz.pre_openmask",
    )
    ap.add_argument("--mm", type=Path, default=ROOT / "unified/work/ctcf/mm")
    ap.add_argument("--impute-dir", type=Path, default=ROOT / "unified/work/ctcf/impute")
    ap.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "unified/work/ctcf/impute_atac_cpm_drop5",
    )
    args = ap.parse_args()
    tracks = [
        ROOT / "downstream/tracks/atac_bulk_sources/2_GSE283384_pooled_CPM.bw",
        ROOT / "downstream/tracks/atac_bulk_sources/3_GSE152177_pooled_CPM.bw",
    ]
    regions_path = args.impute_dir / "regions.tsv"
    if not regions_path.is_file():
        regions_path = args.mm / "regions.tsv.gz"
    regions = read_lines(regions_path)
    chroms, starts, ends = parse_regions(regions)
    print(f"[regions] n={len(regions):,}", flush=True)

    cache = args.out_dir / "bin_atac_mean_cpm.npy"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if cache.is_file():
        cpm = np.load(cache)
        print(f"[cpm] loaded cache {cache}", flush=True)
    else:
        stack = []
        for p in tracks:
            print(f"[cpm] scoring {p.name}", flush=True)
            stack.append(score_bw(p, chroms, starts, ends))
        cpm = np.mean(np.vstack(stack), axis=0)
        np.save(cache, cpm)

    keep = cpm > args.cutoff
    print(
        f"[mask] cutoff={args.cutoff:.6g} CPM  keep={int(keep.sum()):,}/"
        f"{keep.size:,} bins ({100 * keep.mean():.2f}%)  "
        f"median_bin_cpm={np.median(cpm):.4g}",
        flush=True,
    )

    print(f"[load] {args.pre_openmask}", flush=True)
    out = sp.load_npz(args.pre_openmask).tocsr()
    raw = sio.mmread(str(args.mm / "matrix.mtx.gz")).tocsr()
    raw_bin = (raw != 0).astype(np.float32).tocsr()
    if out.shape != raw.shape:
        raise SystemExit(f"shape mismatch {out.shape} vs {raw.shape}")

    out_coo = out.tocoo()
    raw_at = np.asarray(raw_bin[out_coo.row, out_coo.col]).ravel()
    is_new = raw_at == 0
    keep_entries = is_new & keep[out_coo.row]
    pred_masked = sp.coo_matrix(
        (np.ones(int(keep_entries.sum()), dtype=np.float32),
         (out_coo.row[keep_entries], out_coo.col[keep_entries])),
        shape=out.shape,
    ).tocsr()
    new_out = raw_bin.maximum(pred_masked).tocsr()
    new_out.eliminate_zeros()
    print(
        f"[nnz] pre_openmask={out.nnz:,}  imputed_only={int(is_new.sum()):,}  "
        f"imputed_only_kept={pred_masked.nnz:,}  "
        f"output={new_out.nnz:,}  raw={raw_bin.nnz:,}  "
        f"gain={new_out.nnz - raw_bin.nnz:,}",
        flush=True,
    )

    sp.save_npz(args.out_dir / "matrix_csr.npz", new_out)
    for name in ("regions.tsv", "barcodes.tsv"):
        src = args.impute_dir / name
        if src.is_file():
            shutil.copy2(src, args.out_dir / name)
    meta = {}
    mp = args.impute_dir / "meta.json"
    if mp.is_file():
        meta = json.loads(mp.read_text())
    meta.update({
        "mask": "atac_cpm_drop5_not_open_chromatin",
        "atac_cpm_cutoff": float(args.cutoff),
        "atac_cpm_n_keep_bins": int(keep.sum()),
        "source_csr": str(args.pre_openmask),
        "open_chromatin_bed": None,
        "open_chromatin_applied_posthoc": False,
        "predicted_nnz_pre_mask": int(is_new.sum()),
        "predicted_nnz": int(pred_masked.nnz),
        "output_nnz": int(new_out.nnz),
        "raw_nnz": int(raw_bin.nnz),
        "gain_vs_raw": int(new_out.nnz) - int(raw_bin.nnz),
        "shape": [int(new_out.shape[0]), int(new_out.shape[1])],
    })
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[out] {args.out_dir / 'matrix_csr.npz'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
