#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# build_compartment_matrix.py
#
# Compartment-unit TF read matrix for the A/B co-binding correlation. The UNIT is
# an A/B compartment DOMAIN (a contiguous A or B segment from the Hi-C calls). For
# each TF we sum its raw pseudobulk reads (unified/work/<tf>/mm) into each domain.
#
#   output = (domains x TFs) read-count matrix  ->  correlate TFs ACROSS domains
#   (A-A = over A domains, B-B = over B domains) in compartment_correlation.py.
#
# Output (--out-dir):
#   compartment_matrix.npz   scipy CSC float, domains x TFs (raw read counts)
#   domains.tsv              chrom start end compartment bp   (unit order)
#   tfs.txt                  TF column order
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_DOWN = Path(__file__).resolve().parents[2] / "downstream"
if str(_DOWN) not in sys.path:
    sys.path.insert(0, str(_DOWN))
from peak_coverage import load_per_bin_signal  # noqa: E402  (handles mm + impute)


def load_domains(path, unit="bin"):
    """Read A/B compartment units. unit='bin' (default): use each BED row as-is -- the
    25 kb compartment bins from .AB.bed (uniform size -> no domain-size confound). unit=
    'domain': merge contiguous same-label bins into variable-length domains (Mb-scale)."""
    raw = []
    for ln in open(path):
        p = ln.rstrip("\n").split("\t")
        if len(p) < 4:
            continue
        raw.append([p[0], int(p[1]), int(p[2]), p[3].strip()])
    raw.sort(key=lambda x: (x[0], x[1]))
    if unit == "bin":
        return raw                                # each 25 kb bin is a unit
    dom = []
    for c, s, e, lab in raw:                       # merge contiguous same-label -> domains
        if dom and dom[-1][0] == c and dom[-1][3] == lab and s <= dom[-1][2]:
            dom[-1][2] = max(dom[-1][2], e)
        else:
            dom.append([c, s, e, lab])
    return dom


def domain_index(domains):
    """Per-chrom sorted (starts, ends, gidx) for fast bin->domain lookup."""
    by_c = {}
    for gi, (c, s, e, lab) in enumerate(domains):
        by_c.setdefault(c, []).append((s, e, gi))
    idx = {}
    for c, lst in by_c.items():
        lst.sort()
        idx[c] = (np.array([x[0] for x in lst]), np.array([x[1] for x in lst]),
                  np.array([x[2] for x in lst]))
    return idx


def parse_region(r):
    c, rest = r.split(":"); s = int(rest.split("-")[0]); return c, s


def discover_tfs(work_root, sub, matfile):
    return sorted(d.name for d in work_root.iterdir()
                  if d.is_dir() and (d / sub / matfile).is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-root", type=Path,
                    default=Path(__file__).resolve().parents[2] / "unified" / "work")
    ap.add_argument("--domains-bed", type=Path, required=True,
                    help="compartment units: .AB.bed (25 kb bins, for --unit bin) or .domains.bed")
    ap.add_argument("--unit", choices=["bin", "domain"], default="bin",
                    help="bin (default) = 25 kb compartment bins (uniform size, no size confound); "
                         "domain = merged contiguous A/B segments (Mb-scale)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--source", choices=["raw", "imputed"], default="raw")
    ap.add_argument("--tfs", nargs="*", default=None)
    args = ap.parse_args()

    sub = "mm" if args.source == "raw" else "impute"
    domains = load_domains(args.domains_bed, args.unit)
    didx = domain_index(domains)
    nD = len(domains)
    print(f"[domains] {nD:,} units "
          f"(A={sum(d[3]=='A' for d in domains):,} B={sum(d[3]=='B' for d in domains):,})", flush=True)

    tfs = args.tfs or discover_tfs(args.work_root, sub, "matrix.mtx.gz" if sub == "mm" else "matrix_csr.npz")
    if not tfs:
        raise SystemExit(f"no TFs with {sub} matrix under {args.work_root}")

    cols, used = [], []
    for tf in tfs:
        d = args.work_root / tf / sub
        if not (d / "matrix.mtx.gz").is_file() and not (d / "matrix_csr.npz").is_file():
            print(f"[skip] {tf}: no {sub} matrix", file=sys.stderr); continue
        signal, regions, kind, ncells = load_per_bin_signal(d)
        acc = np.zeros(nD, dtype=np.float64)
        for i, r in enumerate(regions):
            v = signal[i]
            if v == 0:
                continue
            c, s = parse_region(r)
            if c not in didx:
                continue
            starts, ends, gidx = didx[c]
            k = np.searchsorted(starts, s, side="right") - 1     # domain whose start <= bin start
            if k >= 0 and s < ends[k]:
                acc[gidx[k]] += v
        cols.append(sp.csc_matrix(acc.reshape(-1, 1))); used.append(tf)
        print(f"[{len(used)}] {tf}: {int((acc>0).sum())}/{nD} domains with reads, total={acc.sum():.0f}", flush=True)

    if not used:
        raise SystemExit("no TF matrices loaded")
    M = sp.hstack(cols).tocsc()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(args.out_dir / "compartment_matrix.npz", M)
    (args.out_dir / "tfs.txt").write_text("\n".join(used) + "\n")
    with (args.out_dir / "domains.tsv").open("w") as f:
        for c, s, e, lab in domains:
            f.write(f"{c}\t{s}\t{e}\t{lab}\t{e-s}\n")
    print(f"[done] {M.shape[0]:,} domains x {M.shape[1]} TFs -> {args.out_dir/'compartment_matrix.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
