#!/usr/bin/env python3
"""Build CTCF pos/neg bins from majority-2/3 peaks after dropping the lowest 5% ATAC FC.

Positives = 1 kb bins overlapping those peaks AND a SCREEN cCRE, not blacklist
(same all_bound rule as all_universe).
Negatives = remaining post-blacklist bins with zero bulk CTCF coverage
(reuses bulk_bedcov_ctcf.npy; no new samtools bedcov).

Writes downstream/bins_ctcf_maj2of3_drop5/pos_neg_bins.{tsv,meta.json}
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT = ROOT / "bins_ctcf_maj2of3_drop5"
PEAKS = REPO / "cobinding" / "results" / "CTCF_majority2of3" / "CTCF_majority2of3.atac_drop_q05.bed"
REGIONS = REPO / "unified" / "work" / "ctcf" / "mm" / "regions.tsv.gz"
COV = ROOT / "plots" / "tf_bulk_pearson" / "_cache" / "bulk_bedcov_ctcf.npy"
BL_GZ = ROOT / "cache" / "hg38-blacklist.v2.bed.gz"
if not BL_GZ.is_file():
    BL_GZ = ROOT / "bins_ctcf_all_bound" / "cache" / "hg38-blacklist.v2.bed.gz"
CCRE_293 = ROOT / "cache" / "SCREEN-293-cCRE-ENCFF439DDQ_ENCFF885SUR_ENCFF128UTY.bed"
CCRE_293T = ROOT / "cache" / "SCREEN-293T-cCRE-ENCFF529BOG.bed"
PUBLIC_PEAKS = REPO / "data" / "CTCF_majority2of3_drop5.bed"


def load_bed(path: Path) -> dict[str, list[tuple[int, int]]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    by: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with opener(path, "rt") as fh:
        for ln in fh:
            if not ln.strip() or ln.startswith(("#", "track", "browser")):
                continue
            p = ln.split("\t")
            chrom, s, e = p[0], int(p[1]), int(p[2])
            if e > s:
                by[chrom].append((s, e))
    for chrom in by:
        by[chrom].sort()
    return by


def load_regions(path: Path):
    names, chroms, starts, ends = [], [], [], []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            name = line.strip()
            chrom, rest = name.split(":", 1)
            a, b = rest.split("-", 1)
            names.append(name)
            chroms.append(chrom)
            starts.append(int(a))
            ends.append(int(b))
    return (
        names,
        np.asarray(chroms),
        np.asarray(starts, np.int64),
        np.asarray(ends, np.int64),
    )


def overlap_mask(chroms, starts, ends, iv_by_chrom) -> np.ndarray:
    n = len(chroms)
    mask = np.zeros(n, dtype=bool)
    order = np.argsort(chroms, kind="mergesort")
    i = 0
    while i < n:
        chrom = str(chroms[order[i]])
        j = i + 1
        while j < n and str(chroms[order[j]]) == chrom:
            j += 1
        iv = iv_by_chrom.get(chrom, [])
        if not iv:
            i = j
            continue
        bi = 0
        for k in range(i, j):
            idx = int(order[k])
            b0, b1 = int(starts[idx]), int(ends[idx])
            while bi < len(iv) and iv[bi][1] <= b0:
                bi += 1
            t = bi
            while t < len(iv) and iv[t][0] < b1:
                if iv[t][1] > b0:
                    mask[idx] = True
                    break
                t += 1
        i = j
    return mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--peaks", type=Path, default=PEAKS)
    ap.add_argument(
        "--exclude-peaks",
        type=Path,
        default=None,
        help=(
            "Peak BED to exclude from the negative pool (full majority 2/3). "
            "Dropped ATAC-low peaks stay out of both pos and neg. Default: --peaks."
        ),
    )
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--public-peaks", type=Path, default=PUBLIC_PEAKS)
    ap.add_argument("--note", default=(
        "positives = majority ≥2/3 CTCF peaks after dropping lowest 5% by "
        "OmniATAC FC, intersect SCREEN cCRE, minus blacklist"
    ))
    args = ap.parse_args()
    peaks_path = args.peaks
    exclude_path = args.exclude_peaks if args.exclude_peaks is not None else peaks_path
    out_dir = args.out_dir
    public_peaks = args.public_peaks

    for p in (peaks_path, exclude_path, REGIONS, COV, BL_GZ, CCRE_293, CCRE_293T):
        if not Path(p).is_file():
            raise SystemExit(f"missing {p}")

    print(f"[peaks] {peaks_path}", flush=True)
    public_peaks.parent.mkdir(parents=True, exist_ok=True)
    public_peaks.write_bytes(peaks_path.read_bytes())

    names, chroms, starts, ends = load_regions(REGIONS)
    n = len(names)
    print(f"[regions] {n:,}", flush=True)
    cov = np.load(COV)
    if cov.size != n:
        raise SystemExit(f"bedcov {cov.size} != regions {n}")

    peaks = load_bed(peaks_path)
    n_pk = sum(len(v) for v in peaks.values())
    print(f"[peaks] {n_pk:,} intervals", flush=True)
    exclude = load_bed(exclude_path)
    n_ex = sum(len(v) for v in exclude.values())
    print(f"[exclude-from-neg] {exclude_path}  {n_ex:,} intervals", flush=True)

    ccre = load_bed(CCRE_293)
    for chrom, ivs in load_bed(CCRE_293T).items():
        ccre[chrom].extend(ivs)
        ccre[chrom].sort()
    print("[cCRE] loaded 293+293T", flush=True)

    bl = load_bed(BL_GZ)
    hit_pk = overlap_mask(chroms, starts, ends, peaks)
    hit_full = overlap_mask(chroms, starts, ends, exclude)
    hit_ccre = overlap_mask(chroms, starts, ends, ccre)
    hit_bl = overlap_mask(chroms, starts, ends, bl)
    print(
        f"[overlap] kept_peak={int(hit_pk.sum()):,}  full_peak={int(hit_full.sum()):,}  "
        f"cCRE={int(hit_ccre.sum()):,}  blacklist={int(hit_bl.sum()):,}",
        flush=True,
    )

    is_pos = hit_pk & hit_ccre & (~hit_bl)
    # all_candidates vs the full consensus bound-cCRE set, not the ATAC-filtered
    # positives. Dropped low-ATAC CTCF sites are held out of both pos and neg.
    is_bound_ccre = hit_full & hit_ccre & (~hit_bl)
    pos = np.flatnonzero(is_pos).astype(np.int64)
    naive = (~hit_bl) & (~is_pos) & (cov <= 0)
    cand = (~hit_bl) & (~is_bound_ccre) & (cov <= 0)
    n_held_from_neg = int((naive & is_bound_ccre).sum())
    neg = np.flatnonzero(cand).astype(np.int64)
    print(
        f"[bins] n_pos={pos.size:,}  n_neg={neg.size:,}  "
        f"held_out_of_neg={n_held_from_neg:,}  "
        f"n_bound_cCRE={int(is_bound_ccre.sum()):,}",
        flush=True,
    )

    keep_orig = np.flatnonzero(~hit_bl)
    orig_to_local0 = np.full(n, -1, dtype=np.int64)
    orig_to_local0[keep_orig] = np.arange(keep_orig.size, dtype=np.int64)

    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / "pos_neg_bins.tsv"
    with tsv.open("w") as fh:
        fh.write("bin_idx_orig\tbin_idx_post_blacklist\tlabel\tbin_name\n")
        for i in pos:
            fh.write(f"{i}\t{orig_to_local0[i]}\tpos\t{names[i]}\n")
        for i in neg:
            fh.write(f"{i}\t{orig_to_local0[i]}\tneg\t{names[i]}\n")

    meta = {
        "mm_regions": str(REGIONS),
        "positive_source": "tf_bound_cCRE",
        "positive_mode": "all_bound",
        "negative_mode": "all_candidates",
        "neg_max_coverage": 0,
        "generic_mode": True,
        "tf_peaks_bed": str(public_peaks),
        "n_peaks": n_pk,
        "n_bins_universe": n,
        "n_blacklist": int(hit_bl.sum()),
        "n_post_blacklist": int((~hit_bl).sum()),
        "n_pos": int(pos.size),
        "n_neg": int(neg.size),
        "n_candidate_neg": int(neg.size),
        "exclude_peaks_bed": str(exclude_path),
        "n_exclude_peaks": n_ex,
        "n_bound_ccre_full": int(is_bound_ccre.sum()),
        "n_held_out_of_neg": n_held_from_neg,
        "bulk_bedcov_npy": str(COV),
        "built_by": "build_ctcf_maj2of3_drop5_bins.py",
        "note": args.note,
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": 2026,
    }
    (out_dir / "pos_neg_bins.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[done] {tsv}", flush=True)
    print(f"[done] n_pos={pos.size} n_neg={neg.size}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
