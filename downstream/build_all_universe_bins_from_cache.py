#!/usr/bin/env python3
"""Build all_universe pos/neg bins without re-running samtools bedcov.

Reuses:
  - positives from existing all_bound (or znf282 final) TSVs
  - genome-wide bulk coverage from plots/tf_bulk_pearson/_cache/bulk_bedcov_<tf>.npy
  - hg38 blacklist v2

Negatives = post-blacklist, non-TF-bound-cCRE (≈ not in the all_bound positive
set), zero bulk coverage — i.e. the full all_candidates pool.

Writes:
  downstream/bins_<tf>_all_universe/pos_neg_bins.{tsv,meta.json}
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
CACHE = ROOT / "plots" / "tf_bulk_pearson" / "_cache"
BL_GZ = ROOT / "bins_ctcf_all_bound" / "cache" / "hg38-blacklist.v2.bed.gz"

# Prefer all_bound; znf282 final is all_bound under a different tag.
POS_SRC = {
    "ctcf": ROOT / "bins_ctcf_all_bound" / "pos_neg_bins.tsv",
    "maz": ROOT / "bins_maz_all_bound" / "pos_neg_bins.tsv",
    "zic2": ROOT / "bins_zic2_all_bound" / "pos_neg_bins.tsv",
    "zbtb7a": ROOT / "bins_zbtb7a_all_bound" / "pos_neg_bins.tsv",
    "znf777": ROOT / "bins_znf777_all_bound" / "pos_neg_bins.tsv",
    "nfya": ROOT / "bins_nfya_all_bound" / "pos_neg_bins.tsv",
    "znf282": ROOT / "bins_znf282_final" / "pos_neg_bins.tsv",
}

PANEL = list(POS_SRC)


def load_regions(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    chroms: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    names: list[str] = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            name = line.strip()
            chrom, rest = name.split(":", 1)
            start_s, end_s = rest.split("-", 1)
            chroms.append(chrom)
            starts.append(int(start_s))
            ends.append(int(end_s))
            names.append(name)
    return names, np.asarray(chroms), np.asarray(starts, np.int64), np.asarray(ends, np.int64)


def blacklist_mask(chroms: np.ndarray, starts: np.ndarray, ends: np.ndarray, bl_gz: Path) -> np.ndarray:
    """True for bins overlapping hg38 blacklist (0-based half-open BED)."""
    from collections import defaultdict

    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with gzip.open(bl_gz, "rt") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split("\t")
            intervals[p[0]].append((int(p[1]), int(p[2])))
    for chrom in intervals:
        intervals[chrom].sort()

    n = len(chroms)
    mask = np.zeros(n, dtype=bool)
    # Group bin indices by chrom for efficient sweep.
    order = np.argsort(chroms, kind="mergesort")
    i = 0
    while i < n:
        chrom = chroms[order[i]]
        j = i + 1
        while j < n and chroms[order[j]] == chrom:
            j += 1
        iv = intervals.get(str(chrom), [])
        if not iv:
            i = j
            continue
        # Sweep bins vs sorted blacklist intervals.
        bi = 0
        for k in range(i, j):
            idx = int(order[k])
            b0, b1 = int(starts[idx]), int(ends[idx])
            while bi < len(iv) and iv[bi][1] <= b0:
                bi += 1
            t = bi
            while t < len(iv) and iv[t][0] < b1:
                # overlap if iv.start < bin.end and iv.end > bin.start
                if iv[t][1] > b0:
                    mask[idx] = True
                    break
                t += 1
        i = j
    return mask


def load_pos_orig(tsv: Path) -> np.ndarray:
    pos = []
    with tsv.open() as fh:
        header = fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if parts[2] != "pos":
                continue
            pos.append(int(parts[0]))
    return np.asarray(pos, dtype=np.int64)


def build_one(tf: str) -> None:
    src = POS_SRC[tf]
    if not src.is_file():
        raise SystemExit(f"missing positive source {src}")
    cov_path = CACHE / f"bulk_bedcov_{tf}.npy"
    if not cov_path.is_file():
        raise SystemExit(f"missing {cov_path}")

    regions_gz = REPO / "unified" / "work" / tf / "mm" / "regions.tsv.gz"
    print(f"[{tf}] loading regions...", flush=True)
    names, chroms, starts, ends = load_regions(regions_gz)
    n = len(names)
    cov = np.load(cov_path)
    if cov.size != n:
        raise SystemExit(f"{tf}: bedcov size {cov.size} != regions {n}")

    print(f"[{tf}] blacklist mask...", flush=True)
    bl = blacklist_mask(chroms, starts, ends, BL_GZ)
    print(f"[{tf}] blacklist bins={int(bl.sum())}", flush=True)

    pos = load_pos_orig(src)
    pos_set = set(int(x) for x in pos)
    print(f"[{tf}] positives from {src.name}: n_pos={len(pos)}", flush=True)

    # Candidate negatives: not blacklist, not TF-bound cCRE (= all_bound pos set),
    # zero bulk coverage.
    is_pos = np.zeros(n, dtype=bool)
    is_pos[pos] = True
    cand = (~bl) & (~is_pos) & (cov <= 0)
    neg = np.flatnonzero(cand).astype(np.int64)
    print(
        f"[{tf}] all_candidates negatives: n_neg={neg.size} "
        f"(zero-cov genome={int((cov <= 0).sum())})",
        flush=True,
    )
    if neg.size < len(pos):
        raise SystemExit(f"{tf}: only {neg.size} negatives for {len(pos)} positives")

    # post-blacklist local index (1-based in R, 0-based here for the TSV col)
    keep_orig = np.flatnonzero(~bl)
    orig_to_local0 = np.full(n, -1, dtype=np.int64)
    orig_to_local0[keep_orig] = np.arange(keep_orig.size, dtype=np.int64)

    out_dir = ROOT / f"bins_{tf}_all_universe"
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / "pos_neg_bins.tsv"
    with tsv.open("w") as fh:
        fh.write("bin_idx_orig\tbin_idx_post_blacklist\tlabel\tbin_name\n")
        for i in pos:
            fh.write(f"{i}\t{orig_to_local0[i]}\tpos\t{names[i]}\n")
        for i in neg:
            fh.write(f"{i}\t{orig_to_local0[i]}\tneg\t{names[i]}\n")

    meta = {
        "mm_regions": str(regions_gz),
        "positive_source": "tf_bound_cCRE",
        "positive_mode": "all_bound",
        "negative_mode": "all_candidates",
        "neg_max_coverage": 0,
        "generic_mode": True,
        "n_bins_universe": n,
        "n_blacklist": int(bl.sum()),
        "n_post_blacklist": int((~bl).sum()),
        "n_pos": int(len(pos)),
        "n_neg": int(neg.size),
        "n_candidate_neg": int(neg.size),
        "bulk_bedcov_npy": str(cov_path),
        "pos_source_tsv": str(src),
        "built_by": "build_all_universe_bins_from_cache.py",
        "seed": 2026,
    }
    (out_dir / "pos_neg_bins.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[{tf}] wrote {tsv} and meta", flush=True)


def main() -> int:
    if not BL_GZ.is_file():
        raise SystemExit(f"missing blacklist {BL_GZ}")
    for tf in PANEL:
        build_one(tf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
