#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# build_peak_matrix.py
#
# Union the per-TF peak BEDs (call_raw_peaks.py) into a consensus set of loci and
# build a loci x TFs BINARY occupancy matrix (TF has a peak overlapping the locus).
# This is the field-standard co-occupancy input (Partridge et al. 2020, Nature):
# co-binding is measured over called peaks, not genome-wide signal bins.
#
# Loci = overlapping peaks from ANY TF merged into one interval (bedtools-merge
# style). Each locus records which TFs contributed a peak -> occupancy k.
#
# Output (--out-dir):
#   peak_matrix.npz   scipy CSC bool, loci x TFs
#   loci.bed          consensus loci (chrom start end locus_i occupancy .)
#   tfs.txt           TF column order
#   loci_ab.tsv       per-locus A/B label (if --ab-bed given): locus_i <TAB> A|B|.
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def read_bed(path: Path):
    out = []
    for ln in open(path):
        if not ln.strip() or ln.startswith(("#", "track", "browser")):
            continue
        p = ln.split("\t")
        sig = 0.0
        if len(p) > 4:
            try:
                sig = float(p[4])                # MACS signalValue (peak intensity)
            except ValueError:
                sig = 0.0
        out.append((p[0], int(p[1]), int(p[2]), sig))
    return out


def merge_to_loci(intervals):
    """intervals: list of (chrom, start, end, tf_idx, signal). Returns (loci, members, sig).

    loci: list of (chrom, start, end); members: list of set(tf_idx); sig: list of
    {tf_idx: max signalValue}. Overlapping intervals from ANY TF merge into one locus;
    per (locus, TF) we keep the strongest peak's signalValue (for CPM-normalized signal).
    """
    intervals = sorted(intervals, key=lambda x: (x[0], x[1], x[2]))
    loci, members, sigs = [], [], []
    cc = cs = ce = None
    cur = set(); cursig = {}
    for c, s, e, t, v in intervals:
        if cc == c and s <= ce:              # overlaps current locus
            ce = max(ce, e); cur.add(t); cursig[t] = max(cursig.get(t, 0.0), v)
        else:
            if cc is not None:
                loci.append((cc, cs, ce)); members.append(cur); sigs.append(cursig)
            cc, cs, ce, cur, cursig = c, s, e, {t}, {t: v}
    if cc is not None:
        loci.append((cc, cs, ce)); members.append(cur); sigs.append(cursig)
    return loci, members, sigs


def compartment_of(ab_bed, loci):
    lab, widths = {}, {}
    for ln in open(ab_bed):
        p = ln.split("\t")
        if len(p) < 4:
            continue
        w = int(p[2]) - int(p[1]); widths[w] = widths.get(w, 0) + 1
    res = max(widths, key=widths.get)
    for ln in open(ab_bed):
        p = ln.split("\t")
        if len(p) < 4:
            continue
        lab[(p[0], int(p[1]) // res)] = p[3].strip()
    # majority label over the res-bins a locus spans
    out = []
    for c, s, e in loci:
        counts = {}
        for b in range(s // res, e // res + 1):
            v = lab.get((c, b))
            if v:
                counts[v] = counts.get(v, 0) + 1
        out.append(max(counts, key=counts.get) if counts else ".")
    return out


def discover_tf_beds(peaks_dir: Path):
    return sorted((f.stem, f) for f in peaks_dir.glob("*.bed"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--peaks-dir", type=Path, required=True, help="dir of per-TF <tf>.bed")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ab-bed", type=Path, default=None, help="A/B calls (hic step 05) for per-locus label")
    ap.add_argument("--tfs", nargs="*", default=None)
    args = ap.parse_args()

    beds = discover_tf_beds(args.peaks_dir)
    if args.tfs:
        keep = set(args.tfs)
        beds = [(t, f) for t, f in beds if t in keep]
    if not beds:
        raise SystemExit(f"no <tf>.bed under {args.peaks_dir}")
    tfs = [t for t, _ in beds]

    intervals = []
    for j, (tf, f) in enumerate(beds):
        n = 0
        for c, s, e, v in read_bed(f):
            intervals.append((c, s, e, j, v)); n += 1
        print(f"[{j+1}/{len(beds)}] {tf}: {n} peaks", flush=True)

    loci, members, sigs = merge_to_loci(intervals)
    nL, nT = len(loci), len(tfs)
    ri, ci, sv = [], [], []
    for i, mem in enumerate(members):
        for t in mem:
            ri.append(i); ci.append(t); sv.append(sigs[i].get(t, 0.0))
    B = sp.csc_matrix((np.ones(len(ri), dtype=bool), (ri, ci)), shape=(nL, nT))
    # continuous per-locus signalValue matrix (for CPM-normalized correlation)
    S = sp.csc_matrix((np.asarray(sv, dtype=np.float32), (ri, ci)), shape=(nL, nT))
    occ = np.asarray(B.sum(axis=1)).ravel()
    print(f"[loci] {nL:,} consensus loci x {nT} TFs; occupancy median={int(np.median(occ))} "
          f"max={int(occ.max())} (>=2 TFs: {int((occ>=2).sum()):,})", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(args.out_dir / "peak_matrix.npz", B)
    sp.save_npz(args.out_dir / "signal_matrix.npz", S)
    (args.out_dir / "tfs.txt").write_text("\n".join(tfs) + "\n")
    with (args.out_dir / "loci.bed").open("w") as f:
        for i, (c, s, e) in enumerate(loci):
            f.write(f"{c}\t{s}\t{e}\tlocus{i}\t{int(occ[i])}\t.\n")
    if args.ab_bed:
        labs = compartment_of(args.ab_bed, loci)
        with (args.out_dir / "loci_ab.tsv").open("w") as f:
            for i, v in enumerate(labs):
                f.write(f"locus{i}\t{v}\n")
        nA = sum(v == "A" for v in labs); nB = sum(v == "B" for v in labs)
        print(f"[ab] loci A={nA:,} B={nB:,} unlabeled={nL-nA-nB:,}", flush=True)
    print(f"[done] -> {args.out_dir/'peak_matrix.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
