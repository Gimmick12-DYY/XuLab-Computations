#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Quantitative overlay check: mean bulk bigWig signal in the matching peak BED
# vs the same intervals shifted +50 kb (same chrom, clipped). A liftOver /
# assembly mismatch drops the peak/shift ratio toward 1.
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DT_PY = Path(
    "/nas/longleaf/rhel9/apps/deeptools/3.5.6/miniconda3/envs/deeptools/bin/python"
)


def _ensure_pybigwig():
    try:
        import pyBigWig  # noqa: F401
        return
    except ImportError:
        pass
    if DT_PY.is_file() and Path(sys.executable).resolve() != DT_PY.resolve():
        os.execv(str(DT_PY), [str(DT_PY), *sys.argv])
    raise SystemExit("pyBigWig required")


_ensure_pybigwig()
import numpy as np  # noqa: E402
import pyBigWig  # noqa: E402

SHIFT = 50_000


def load_bed(path: Path, limit: int | None = None):
    rows = []
    with path.open() as fh:
        for ln in fh:
            if not ln.strip() or ln.startswith(("#", "track", "browser")):
                continue
            p = ln.split("\t")
            if len(p) < 3:
                continue
            rows.append((p[0], int(p[1]), int(p[2])))
            if limit and len(rows) >= limit:
                break
    return rows


def mean_over(bw, chroms, rows, shift=0):
    vals = np.full(len(rows), np.nan)
    for i, (c, s, e) in enumerate(rows):
        if c not in chroms:
            continue
        clen = chroms[c]
        s2, e2 = s + shift, e + shift
        if shift and e2 > clen:
            s2, e2 = s - shift, e - shift
        s2 = max(0, s2)
        e2 = min(clen, e2)
        if e2 <= s2:
            continue
        v = bw.stats(c, s2, e2, type="mean")
        if v and v[0] is not None:
            vals[i] = float(v[0])
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-dir", type=Path,
                    default=ROOT / "downstream" / "tracks" / "bulk_chipseq")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.track_dir / "peak_vs_shift_summary.tsv")

    rows_out = []
    for bw_path in sorted(args.track_dir.glob("*_bulk_CPM.bw")):
        tf = bw_path.name.replace("_bulk_CPM.bw", "")
        peaks = args.track_dir / f"{tf}_peaks.bed"
        if not peaks.exists():
            print(f"[skip] {tf} no peaks link", flush=True)
            continue
        real = peaks.resolve() if peaks.is_symlink() else peaks
        if not real.is_file():
            print(f"[skip] {tf} peaks missing", flush=True)
            continue
        bw = pyBigWig.open(str(bw_path))
        chroms = bw.chroms() or {}
        chr1 = chroms.get("chr1")
        bed = load_bed(real)
        pv = mean_over(bw, chroms, bed, 0)
        sv = mean_over(bw, chroms, bed, SHIFT)
        bw.close()
        ok = np.isfinite(pv) & np.isfinite(sv)
        med_p = float(np.nanmedian(pv))
        med_s = float(np.nanmedian(sv))
        ratio = med_p / med_s if med_s > 0 else float("nan")
        n_ok = int(ok.sum())
        print(
            f"{tf:8s}  n={len(bed):6d}  scored={n_ok:6d}  "
            f"chr1={chr1}  peak_med={med_p:.4g}  "
            f"+{SHIFT//1000}kb_med={med_s:.4g}  ratio={ratio:.2f}",
            flush=True,
        )
        rows_out.append(
            f"{tf}\t{bw_path}\t{real}\t{len(bed)}\t{n_ok}\t{chr1}\t"
            f"{med_p:.6g}\t{med_s:.6g}\t{ratio:.4f}\n"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write("tf\tbigwig\tpeaks\tn_peaks\tn_scored\tchr1_len\t"
                 "median_in_peaks\tmedian_shifted_50kb\tratio_peak_over_shift\n")
        fh.writelines(rows_out)
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
