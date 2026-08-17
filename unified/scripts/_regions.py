"""Shared region / BED overlap helpers for the unified pipeline."""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np

_REGION_RE = re.compile(r"^([^:]+):(\d+)-(\d+)$")


def parse_region_names(names: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(names)
    chroms = np.empty(n, dtype=object)
    starts = np.zeros(n, dtype=np.int64)
    ends = np.zeros(n, dtype=np.int64)
    for i, name in enumerate(names):
        m = _REGION_RE.match(str(name).strip())
        if not m:
            raise SystemExit(f"Bad region name row {i}: {name!r}")
        chroms[i] = m.group(1)
        starts[i] = int(m.group(2))
        ends[i] = int(m.group(3))
    return chroms, starts, ends


def load_bed_intervals(bed_path: Path) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    opener = gzip.open if str(bed_path).endswith(".gz") else open
    with opener(bed_path, "rt") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith(("#", "track", "browser")):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                out.setdefault(parts[0], []).append((int(parts[1]), int(parts[2])))
            except ValueError:
                continue
    return out


def bins_overlap_bed(
    chroms: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    intervals: dict[str, list[tuple[int, int]]],
) -> np.ndarray:
    """Boolean mask: True iff bin overlaps any BED interval (half-open)."""
    mask = np.zeros(len(chroms), dtype=bool)
    for chrom in np.unique(chroms):
        ivs = intervals.get(str(chrom))
        if not ivs:
            # tolerate chr-less keys
            ivs = intervals.get(str(chrom).replace("chr", ""))
        if not ivs:
            continue
        ivs = sorted(ivs)
        starts_iv = np.fromiter((a for a, _ in ivs), dtype=np.int64, count=len(ivs))
        ends_iv = np.fromiter((b for _, b in ivs), dtype=np.int64, count=len(ivs))
        bin_idx = np.where(chroms == chrom)[0]
        bs = starts[bin_idx]
        be = ends[bin_idx]
        # Interval starts that could overlap bin: start < be
        hi_arr = np.searchsorted(starts_iv, be, side="left")
        local = np.zeros(len(bin_idx), dtype=bool)
        for k, (b_s, hi) in enumerate(zip(bs, hi_arr)):
            if hi == 0:
                continue
            if (ends_iv[:hi] > b_s).any():
                local[k] = True
        mask[bin_idx[local]] = True
    return mask


def open_chromatin_bin_mask(
    regions: list[str],
    bed_path: Path | str | None,
) -> np.ndarray | None:
    """Return open-chromatin boolean mask over ``regions``, or None if disabled."""
    if bed_path is None:
        return None
    if isinstance(bed_path, str) and bed_path.strip().lower() in {"", "null", "none", "off"}:
        return None
    p = Path(bed_path)
    if not p.is_file():
        raise SystemExit(f"open_chromatin_bed not found: {p}")
    chroms, starts, ends = parse_region_names(regions)
    ivs = load_bed_intervals(p)
    return bins_overlap_bed(chroms, starts, ends, ivs)
