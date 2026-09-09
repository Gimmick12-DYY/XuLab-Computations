#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# build_chromhmm_matrix.py
#
# Per-TF RPKM in each ChromHMM-18 state, for the "TF chromHMM18 rpkm ratio" analysis.
# UNIT = a ChromHMM state (18 states, aggregated over ALL their genomic segments).
# For each TF, sum its raw pseudobulk reads (unified/work/<tf>/mm) falling in each
# state, then:
#     RPKM(state) = reads_in_state / (state_total_bp / 1e3) / (TF_total_reads / 1e6)
# RPKM normalizes by state LENGTH (removes the size confound) AND per-TF library depth
# (removes the cell-count/complexity confound) -- both at once, unlike CPM.
#
# Output (--out-dir):
#   chromhmm_matrix.npz   (states x TFs) RPKM, CSC float
#   states.tsv            state  total_bp   (unit order)
#   tfs.txt               TF column order
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_DOWN = Path(__file__).resolve().parents[2] / "downstream"
if str(_DOWN) not in sys.path:
    sys.path.insert(0, str(_DOWN))
from peak_coverage import load_per_bin_signal  # noqa: E402  (mm + impute)


def load_chromhmm(path):
    """BED (chrom start end state[...]) -> per-chrom sorted segments + state total bp.

    State label = 4th column (accepts '1_TssA', 'TssA', 'E1', etc. -- used verbatim)."""
    import gzip
    opener = gzip.open if str(path).endswith(".gz") else open
    by_c = {}
    state_bp = OrderedDict()
    for ln in opener(path, "rt"):
        if not ln.strip() or ln.startswith(("#", "track", "browser")):
            continue
        p = ln.split("\t")
        if len(p) < 4:
            continue
        c, s, e, st = p[0], int(p[1]), int(p[2]), p[3].strip()
        by_c.setdefault(c, []).append((s, e, st))
        state_bp[st] = state_bp.get(st, 0) + (e - s)
    idx = {}
    for c, segs in by_c.items():
        segs.sort()
        idx[c] = (np.array([x[0] for x in segs]), np.array([x[1] for x in segs]),
                  [x[2] for x in segs])
    return idx, state_bp


def parse_region(r):
    c, rest = r.split(":"); return c, int(rest.split("-")[0])


def discover_tfs(work_root, sub, matfile):
    return sorted(d.name for d in work_root.iterdir()
                  if d.is_dir() and (d / sub / matfile).is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-root", type=Path,
                    default=Path(__file__).resolve().parents[2] / "unified" / "work")
    ap.add_argument("--chromhmm-bed", type=Path, required=True,
                    help="ChromHMM-18 segmentation BED for HEK293T (chrom start end state)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--source", choices=["raw", "imputed"], default="raw")
    ap.add_argument("--tfs", nargs="*", default=None)
    args = ap.parse_args()

    sub = "mm" if args.source == "raw" else "impute"
    seg_idx, state_bp = load_chromhmm(args.chromhmm_bed)
    states = list(state_bp.keys())
    sid = {s: i for i, s in enumerate(states)}
    bp = np.array([state_bp[s] for s in states], dtype=np.float64)
    print(f"[chromhmm] {len(states)} states, {int(bp.sum()):,} bp total", flush=True)

    tfs = args.tfs or discover_tfs(args.work_root, sub, "matrix.mtx.gz" if sub == "mm" else "matrix_csr.npz")
    if not tfs:
        raise SystemExit(f"no TFs with {sub} matrix under {args.work_root}")

    cols, used = [], []
    for tf in tfs:
        d = args.work_root / tf / sub
        if not (d / "matrix.mtx.gz").is_file() and not (d / "matrix_csr.npz").is_file():
            print(f"[skip] {tf}: no {sub} matrix", file=sys.stderr); continue
        signal, regions, kind, ncells = load_per_bin_signal(d)
        reads = np.zeros(len(states), dtype=np.float64)
        for i, r in enumerate(regions):
            v = signal[i]
            if v == 0:
                continue
            c, s = parse_region(r)
            if c not in seg_idx:
                continue
            starts, ends, labs = seg_idx[c]
            k = np.searchsorted(starts, s, side="right") - 1     # segment whose start <= bin start
            if k >= 0 and s < ends[k]:
                reads[sid[labs[k]]] += v
        libM = reads.sum() / 1e6
        rpkm = reads / (bp / 1e3) / (libM if libM > 0 else 1.0)  # RPKM per state
        cols.append(sp.csc_matrix(rpkm.reshape(-1, 1))); used.append(tf)
        print(f"[{len(used)}] {tf}: {reads.sum():.0f} reads in states", flush=True)

    if not used:
        raise SystemExit("no TF matrices loaded")
    M = sp.hstack(cols).tocsc()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(args.out_dir / "chromhmm_matrix.npz", M)
    (args.out_dir / "tfs.txt").write_text("\n".join(used) + "\n")
    with (args.out_dir / "states.tsv").open("w") as f:
        for s in states:
            f.write(f"{s}\t{state_bp[s]}\n")
    print(f"[done] {M.shape[0]} states x {M.shape[1]} TFs -> {args.out_dir/'chromhmm_matrix.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
