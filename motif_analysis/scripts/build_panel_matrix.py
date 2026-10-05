#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# build_panel_matrix.py
#
# Build the per-bin x per-TF prevalence matrix for the whole panel, once.
#
# Why this exists: motif discovery on the imputed calls keeps returning generic
# promoter grammar (Maz/SP/NFY/CTCF) for every TF. The reason is structural, not
# a parameter choice:
#   - the raw input is *_bin1000_mtx.rds, i.e. 1 kb bins. There are no fragments,
#     so MACS3 "summits" sit at the bin midpoint and carry no positional
#     information -- centrality-based discovery (CentriMo etc.) cannot work here.
#   - imputed signal is masked to the HEK293T ATAC union (~105k open bins), so
#     every TF's peaks are drawn from the SAME universe of open, GC-rich,
#     promoter-heavy 1 kb windows. Against a genome or OCR background what wins
#     is "open chromatin", not "this TF".
#
# The one axis that still separates TFs is the panel itself: 78 TFs profiled on
# an identical bin grid in the same cell type. A bin where TF t is high and the
# other 77 are low is TF-specific by construction, and a background drawn from
# bins that are equally open but not bound by t is matched for accessibility,
# GC, CpG and chromatin state without any explicit matching. build_panel_matrix
# materializes the matrix that makes that comparison possible.
#
# Per TF: per-bin pseudobulk (peak_coverage.load_per_bin_signal) / n_cells ->
# prevalence in (0, 1]. Columns are additionally rank-normalized to a common
# uniform marginal so per-TF depth and breadth cannot masquerade as specificity.
#
# Output (--out-npz):
#   prevalence  float32 (n_kept, n_tfs)   fraction of cells with signal
#   ranknorm    float32 (n_kept, n_tfs)   per-column percentile rank of nonzeros
#   bin_idx     int32   (n_kept,)         row index into the full bin universe
#   regions     str     (n_kept,)         'chr1:1000-2000'
#   tfs         str     (n_tfs,)
#   n_cells     int32   (n_tfs,)
# Bins that are zero in every TF are dropped.
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_DOWN = Path(__file__).resolve().parent.parent
if str(_DOWN) not in sys.path:
    sys.path.insert(0, str(_DOWN))
from peak_coverage import load_per_bin_signal  # noqa: E402


def rank_normalize(col: np.ndarray) -> np.ndarray:
    """Percentile-rank the nonzero entries of one track into (0, 1]; zeros stay 0.

    Removes per-TF depth and breadth so that a bin being "high for this TF" means
    high *relative to that TF's own distribution*, which is what the differential
    test needs.
    """
    out = np.zeros_like(col, dtype=np.float32)
    nz = np.flatnonzero(col)
    if nz.size == 0:
        return out
    order = np.argsort(col[nz], kind="stable")
    ranks = np.empty(nz.size, dtype=np.float64)
    ranks[order] = np.arange(1, nz.size + 1, dtype=np.float64)
    out[nz] = (ranks / nz.size).astype(np.float32)
    return out


def discover_tfs(work_root: Path) -> list[str]:
    return sorted(
        d.name for d in work_root.iterdir()
        if d.is_dir() and (d / "impute" / "matrix_csr.npz").is_file()
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-root", type=Path,
                    default=_DOWN.parent / "unified" / "work")
    ap.add_argument("--tfs", nargs="*", default=None,
                    help="lowercase TF keys; default = every <tf>/impute/matrix_csr.npz")
    ap.add_argument("--out-npz", type=Path, required=True)
    ap.add_argument("--min-tfs", type=int, default=10)
    args = ap.parse_args()

    tfs = args.tfs or discover_tfs(args.work_root)
    if not tfs:
        raise SystemExit(f"no TF impute dirs under {args.work_root}")

    prevalence: np.ndarray | None = None
    regions: list[str] | None = None
    n_cells: list[int] = []
    used: list[str] = []

    for tf in tfs:
        d = args.work_root / tf / "impute"
        if not (d / "matrix_csr.npz").is_file():
            print(f"[skip] {tf}: no matrix_csr.npz", flush=True)
            continue
        signal, regs, _kind, cells = load_per_bin_signal(d)
        if prevalence is None:
            regions = regs
            prevalence = np.zeros((len(regs), len(tfs)), dtype=np.float32)
        elif len(regs) != len(regions or []):
            raise SystemExit(f"{tf}: bins {len(regs)} != {len(regions or [])} (universe mismatch)")
        prevalence[:, len(used)] = (signal / max(cells, 1)).astype(np.float32)
        used.append(tf)
        n_cells.append(int(cells))
        print(f"[{len(used)}/{len(tfs)}] {tf}: n_cells={cells} "
              f"nonzero_bins={int(np.count_nonzero(signal))}", flush=True)

    if len(used) < args.min_tfs:
        raise SystemExit(f"only {len(used)} TFs (< --min-tfs {args.min_tfs})")
    assert prevalence is not None and regions is not None
    prevalence = prevalence[:, :len(used)]

    keep = np.flatnonzero(prevalence.any(axis=1))
    prevalence = np.ascontiguousarray(prevalence[keep])
    kept_regions = np.array([regions[i] for i in keep], dtype=object)
    print(f"\n[panel] {len(used)} TFs x {keep.size} bins with any signal "
          f"(of {len(regions)})", flush=True)

    ranknorm = np.zeros_like(prevalence)
    for j in range(prevalence.shape[1]):
        ranknorm[:, j] = rank_normalize(prevalence[:, j])

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        prevalence=prevalence,
        ranknorm=ranknorm,
        bin_idx=keep.astype(np.int32),
        regions=kept_regions.astype("U32"),
        tfs=np.array(used, dtype="U32"),
        n_cells=np.array(n_cells, dtype=np.int32),
    )
    meta = {
        "n_tfs": len(used),
        "n_bins_kept": int(keep.size),
        "n_bins_universe": len(regions),
        "tfs": used,
        "work_root": str(args.work_root),
        "built_by": "build_panel_matrix.py",
    }
    Path(str(args.out_npz) + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[panel] -> {args.out_npz}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
