#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# build_compartment_matrix.py
#
# Compartment-unit TF RPKM matrix for the A/B co-binding correlation -- the same
# recipe validated on ChromHMM-18 (build_chromhmm_matrix.py), with the UNIT swapped
# from a chromatin state to an A/B compartment DOMAIN (a contiguous A or B segment
# from the Hi-C calls). For each TF we sum its raw pseudobulk reads
# (unified/work/<tf>/mm) into each domain, then:
#     RPKM(domain)     = reads / (domain_bp / 1e3) / (TF_total_reads / 1e6)
#     per_cell(domain) = reads / (domain_bp / 1e3) / n_cells
# n_cells comes from the TF1000 metadata (one row per cell), not from the read
# total. Both depth terms are one positive number per TF. Pearson correlates
# the shape of each TF's profile, so swapping reads for cells does not change
# the TF x TF correlation; the /kb term does, because domain length varies.
#
#   output = (domains x TFs) RPKM matrix  ->  correlate TFs ACROSS domains
#   (A-A = over A domains, B-B = over B domains) in compartment_rpkm_correlation.py.
#
# Output (--out-dir):
#   compartment_matrix.npz   scipy CSC float, domains x TFs (RPKM, or counts)
#   domains.tsv              chrom start end compartment bp   (unit order)
#   tfs.txt                  TF column order
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import csv
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


def load_cell_counts(path: Path) -> dict[str, int]:
    """Count cells per TF from TF1000cells.meta.csv (column TF), or a TF,count table."""
    with open(path) as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return {}
    hdr = [h.strip() for h in rows[0]]
    if "TF" in hdr:
        i = hdr.index("TF")
        cc: dict[str, int] = {}
        for r in rows[1:]:
            if len(r) > i and r[i].strip():
                tf = r[i].strip().lower()
                cc[tf] = cc.get(tf, 0) + 1
        return cc
    cc = {}
    for r in rows:
        if len(r) >= 2 and r[1].strip().isdigit():
            cc[r[0].strip().lower()] = int(r[1])
    return cc


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
    ap.add_argument("--residualize", action="store_true",
                    help="imputed only: subtract consensus accessibility before RPKM")
    ap.add_argument("--consensus-npy", type=Path, default=None)
    ap.add_argument("--consensus-scale", type=float, default=1.0)
    ap.add_argument("--transform", choices=["rpkm", "per_cell", "count"], default="rpkm",
                    help="rpkm = reads/kb/(total_reads/1e6); "
                         "per_cell = reads/kb/n_cells from --cell-meta; "
                         "count = raw reads")
    ap.add_argument("--cell-meta", type=Path,
                    default=Path(__file__).resolve().parents[2] / "data" / "TF1000cells.meta.csv",
                    help="per-cell metadata; n_cells per TF is the per_cell denominator")
    ap.add_argument("--tfs", nargs="*", default=None)
    args = ap.parse_args()
    cell_counts = load_cell_counts(args.cell_meta) if args.transform == "per_cell" else {}
    if args.transform == "per_cell" and not cell_counts:
        raise SystemExit(f"--transform per_cell needs cell counts in {args.cell_meta}")

    sub = "mm" if args.source == "raw" else "impute"
    if args.residualize:
        if args.source != "imputed":
            raise SystemExit("--residualize requires --source imputed")
        if args.consensus_npy is None or not args.consensus_npy.is_file():
            raise SystemExit("--residualize needs an existing --consensus-npy")
        cons = np.load(args.consensus_npy)
        print(f"[consensus] {cons.shape[0]} bins from {args.consensus_npy}", flush=True)
    else:
        cons = None
    domains = load_domains(args.domains_bed, args.unit)
    didx = domain_index(domains)
    nD = len(domains)
    bp = np.array([e - s for _, s, e, _ in domains], dtype=np.float64)
    bp[bp <= 0] = 1.0
    print(f"[domains] {nD:,} units "
          f"(A={sum(d[3]=='A' for d in domains):,} B={sum(d[3]=='B' for d in domains):,}) "
          f"median {int(np.median(bp)):,} bp; transform={args.transform}", flush=True)

    tfs = args.tfs or discover_tfs(args.work_root, sub, "matrix.mtx.gz" if sub == "mm" else "matrix_csr.npz")
    if not tfs:
        raise SystemExit(f"no TFs with {sub} matrix under {args.work_root}")

    cols, used = [], []
    for tf in tfs:
        d = args.work_root / tf / sub
        if not (d / "matrix.mtx.gz").is_file() and not (d / "matrix_csr.npz").is_file():
            print(f"[skip] {tf}: no {sub} matrix", file=sys.stderr); continue
        signal, regions, kind, ncells = load_per_bin_signal(d)
        if cons is not None:
            if cons.shape[0] != signal.shape[0]:
                raise SystemExit(
                    f"{tf}: consensus bins {cons.shape[0]} != signal {signal.shape[0]}")
            tot = float(signal.sum())
            signal = np.clip(signal - args.consensus_scale * cons * tot, 0.0, None)
            kept = float(signal.sum()) / tot if tot > 0 else 0.0
            print(f"  residual kept {100 * kept:.1f}% of counts", flush=True)
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
        n_hit, total = int((acc > 0).sum()), acc.sum()
        depth_note = ""
        if args.transform == "rpkm":
            libM = total / 1e6
            acc = acc / (bp / 1e3) / (libM if libM > 0 else 1.0)
            depth_note = f" lib_reads={total:.0f}"
        elif args.transform == "per_cell":
            n_meta = cell_counts.get(tf.lower())
            if not n_meta:
                raise SystemExit(f"{tf}: no cell count in {args.cell_meta}")
            acc = acc / (bp / 1e3) / float(n_meta)
            depth_note = f" n_cells={n_meta} (matrix columns={ncells})"
        cols.append(sp.csc_matrix(acc.reshape(-1, 1))); used.append(tf)
        print(f"[{len(used)}] {tf}: {n_hit}/{nD} domains with reads, total={total:.0f}{depth_note}",
              flush=True)

    if not used:
        raise SystemExit("no TF matrices loaded")
    M = sp.hstack(cols).tocsc()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(args.out_dir / "compartment_matrix.npz", M)
    (args.out_dir / "tfs.txt").write_text("\n".join(used) + "\n")
    (args.out_dir / "norm.txt").write_text(args.transform + "\n")
    with (args.out_dir / "domains.tsv").open("w") as f:
        for c, s, e, lab in domains:
            f.write(f"{c}\t{s}\t{e}\t{lab}\t{e-s}\n")
    print(f"[done] {M.shape[0]:,} domains x {M.shape[1]} TFs -> {args.out_dir/'compartment_matrix.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
