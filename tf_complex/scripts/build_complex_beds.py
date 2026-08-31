#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# build_complex_beds.py
#
# For each candidate complex (candidate_complexes.tsv), emit the group's CO-BOUND
# loci as a BED (foreground) plus a matched BACKGROUND BED, for the shared-motif
# step (HOMER / AME). Checklist step 4: "are the complex members sharing / using
# the same motif?"
#
# Foreground = consensus loci (build_peak_matrix.py) co-bound by at least
# ceil(--min-frac * n_members) members of the complex (>=2). Background = all other
# consensus loci (TF-bound regions NOT co-bound by this complex). Using the union
# peak set as background -- not random genome -- asks the specific question "what
# motif marks THIS complex's co-bound subset among bound regions", avoiding the
# generic GC/accessibility motifs that a whole-genome background yields
# ([[imputed-peaks-accessibility-wash]]).
#
# Output (--out-dir):
#   <complex_id>.cobound.bed   foreground loci
#   <complex_id>.bg.bed        background loci
#   complex_manifest.tsv       complex_id, n_members, n_cobound, n_bg, members
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def read_loci(loci_bed):
    rows = []
    for ln in open(loci_bed):
        p = ln.split("\t")
        rows.append((p[0], int(p[1]), int(p[2])))
    return rows


def parse_complexes(path):
    out = []
    for i, ln in enumerate(open(path)):
        if i == 0 or not ln.strip():
            continue
        t = ln.rstrip("\n").split("\t")
        out.append((t[0], [m for m in t[3].split(",") if m]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix-dir", type=Path, required=True, help="build_peak_matrix.py output")
    ap.add_argument("--complexes", type=Path, required=True, help="candidate_complexes.tsv")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-frac", type=float, default=0.5,
                    help="a locus is 'co-bound' if >= ceil(min-frac * n_members) members bind it "
                         "(and always >=2). Default 0.5 = at least half the complex.")
    ap.add_argument("--window", type=int, default=200,
                    help="emit FIXED-WIDTH windows of this size centered on each locus midpoint "
                         "(default 200). Equalizes foreground/background length -- co-bound loci "
                         "are intrinsically wider (merged from more peaks), which would otherwise "
                         "bias motif enrichment by length/GC. 0 = keep original locus widths.")
    ap.add_argument("--min-cobound", type=int, default=100,
                    help="warn if a complex has fewer than this many co-bound loci (too few for "
                         "a reliable motif enrichment).")
    args = ap.parse_args()

    B = sp.load_npz(args.matrix_dir / "peak_matrix.npz").tocsc()
    tfs = [t.strip() for t in (args.matrix_dir / "tfs.txt").read_text().split() if t.strip()]
    ix = {t: i for i, t in enumerate(tfs)}
    loci = read_loci(args.matrix_dir / "loci.bed")
    complexes = parse_complexes(args.complexes)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    w = args.window
    def write_bed(path, mask, tag):
        idx = np.flatnonzero(mask)
        with open(path, "w") as f:
            for k, i in enumerate(idx):
                c, s, e = loci[i]
                if w and w > 0:                       # fixed-width window on midpoint
                    mid = (s + e) // 2
                    s, e = max(0, mid - w // 2), mid + w - w // 2
                f.write(f"{c}\t{s}\t{e}\t{tag}_{k}\t0\t.\n")
        return idx.size

    man = args.out_dir / "complex_manifest.tsv"
    with man.open("w") as mf:
        mf.write("complex_id\tn_members\tn_cobound\tn_bg\tmin_members\tmembers\n")
        for cid, members in complexes:
            mem_idx = [ix[m] for m in members if m in ix]
            if len(mem_idx) < 2:
                print(f"[skip] {cid}: <2 members present", flush=True)
                continue
            sub = B[:, mem_idx]
            occ = np.asarray(sub.sum(axis=1)).ravel()
            thr = max(2, math.ceil(args.min_frac * len(mem_idx)))
            fg = occ >= thr
            bg = (~fg) & (np.asarray(B.sum(axis=1)).ravel() > 0)   # bound elsewhere, not by this group
            n_fg = write_bed(args.out_dir / f"{cid}.cobound.bed", fg, cid)
            n_bg = write_bed(args.out_dir / f"{cid}.bg.bed", bg, f"{cid}bg")
            flag = "  <-- FEW LOCI" if n_fg < args.min_cobound else ""
            print(f"[{cid}] members={len(mem_idx)} thr>={thr} cobound={n_fg} bg={n_bg}{flag}", flush=True)
            mf.write(f"{cid}\t{len(mem_idx)}\t{n_fg}\t{n_bg}\t{thr}\t{','.join(members)}\n")
    print(f"[done] -> {args.out_dir} (manifest {man})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
