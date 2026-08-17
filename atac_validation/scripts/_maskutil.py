"""Shared helpers for the ATAC-seq validation overlap/union scripts.

Reuses the unified pipeline's region/BED overlap logic (`unified/scripts/_regions.py`)
so "open bin" here means exactly what it means in the imputer's open-chromatin mask.
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np

# Make unified/scripts/_regions.py importable regardless of CWD.
_UNIFIED_SCRIPTS = Path(__file__).resolve().parents[2] / "unified" / "scripts"
if str(_UNIFIED_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_UNIFIED_SCRIPTS))

from _regions import (  # noqa: E402
    bins_overlap_bed,
    load_bed_intervals,
    parse_region_names,
)

_HEADER_TOKENS = {"region", "regions", "name", "bin", "peak", "chrom"}


def load_region_names(regions_tsv: str | Path) -> list[str]:
    """Read 'chr:start-end' region names from the first column of a TSV(.gz)."""
    names: list[str] = []
    opener = gzip.open if str(regions_tsv).endswith(".gz") else open
    with opener(regions_tsv, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tok = line.split("\t")[0].split()[0]
            if tok.lower() in _HEADER_TOKENS:
                continue
            names.append(tok)
    if not names:
        raise SystemExit(f"no region names parsed from {regions_tsv}")
    return names


def region_arrays(names: list[str]):
    return parse_region_names(names)


def bin_mask_for_peaks(chroms, starts, ends, peak_path: str | Path) -> np.ndarray:
    """Boolean mask over the bin universe: True iff the bin overlaps a peak."""
    intervals = load_bed_intervals(Path(peak_path))
    return bins_overlap_bed(chroms, starts, ends, intervals)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return inter / union if union else 0.0
