#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# build_consensus_accessibility.py
#
# Build a consensus accessibility profile across TFs, for consensus-subtracted
# ("TF-specific") peak calling. Imputed peaks are dominated by a shared
# accessibility landscape (all TFs' pseudobulks look alike, 75-91% peak overlap),
# so their motif enrichment returns generic GC/CpG motifs. Subtracting this
# consensus from a TF's own signal isolates the fraction that is enriched ABOVE
# generic accessibility -- the part that can carry the TF's own motif.
#
# Per TF: per-bin pseudobulk (sum across cells, via peak_coverage.load_per_bin_signal)
#   -> normalize (default 'rank': percentile-rank each track's nonzero bins to a
#      common (0,1] marginal, removing per-TF depth/breadth differences)
#   -> accumulate. The MEAN across TFs is the consensus.
#
# call_imputed_peaks.py --consensus-npy <this> then uses, per bin,
#   residual_X = max(0, normalize(signal_X) - consensus)
# as the signal fed to MACS3 (same --consensus-normalize mode).
#
# Output: <out-npy> (float64, length = n_bins) + <out-npy>.meta.json
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_DOWN = Path(__file__).resolve().parent
if str(_DOWN) not in sys.path:
    sys.path.insert(0, str(_DOWN))
from peak_coverage import load_per_bin_signal, normalize_per_bin_signal  # noqa: E402


def discover_tfs(work_root: Path) -> list[str]:
    return sorted(
        d.name for d in work_root.iterdir()
        if d.is_dir() and (d / "impute" / "matrix_csr.npz").is_file()
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-root", type=Path, default=_DOWN.parent / "unified" / "work")
    ap.add_argument("--tfs", nargs="*", default=None,
                    help="lowercase TF keys; default = every <tf>/impute/matrix_csr.npz")
    ap.add_argument("--normalize", choices=["rank", "none"], default="rank",
                    help="per-track normalization before averaging (default rank)")
    ap.add_argument("--out-npy", type=Path, required=True)
    ap.add_argument("--min-tfs", type=int, default=5,
                    help="require at least this many TFs for a meaningful consensus")
    args = ap.parse_args()

    tfs = args.tfs or discover_tfs(args.work_root)
    if not tfs:
        raise SystemExit(f"no TF impute dirs under {args.work_root}")

    acc: np.ndarray | None = None
    n_bins = 0
    used: list[str] = []
    for tf in tfs:
        d = args.work_root / tf / "impute"
        if not (d / "matrix_csr.npz").is_file():
            print(f"[skip] {tf}: no matrix_csr.npz", flush=True)
            continue
        signal, _regions, _kind, n_cells = load_per_bin_signal(d)
        s = normalize_per_bin_signal(signal, args.normalize)
        if acc is None:
            n_bins = int(s.shape[0])
            acc = np.zeros(n_bins, dtype=np.float64)
        elif s.shape[0] != n_bins:
            raise SystemExit(f"{tf}: bins {s.shape[0]} != {n_bins} (universe mismatch)")
        acc += s
        used.append(tf)
        print(f"[{len(used)}] {tf}: n_cells={n_cells} nonzero={int(np.count_nonzero(signal))}", flush=True)

    if len(used) < args.min_tfs:
        raise SystemExit(f"only {len(used)} TFs (< --min-tfs {args.min_tfs}); consensus not robust")
    assert acc is not None
    consensus = acc / float(len(used))

    args.out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_npy, consensus)
    meta = {
        "normalize": args.normalize,
        "n_tfs": len(used),
        "n_bins": n_bins,
        "tfs": used,
        "work_root": str(args.work_root),
        "built_by": "build_consensus_accessibility.py",
    }
    Path(str(args.out_npy) + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\n[consensus] {len(used)} TFs, {n_bins} bins, normalize={args.normalize}")
    print(f"[consensus] mean={consensus.mean():.4g} nonzero={int(np.count_nonzero(consensus))} "
          f"-> {args.out_npy}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
