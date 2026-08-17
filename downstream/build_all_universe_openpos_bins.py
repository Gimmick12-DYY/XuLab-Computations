#!/usr/bin/env python3
"""Build all_universe bins with positives restricted to open chromatin.

Definitions:
  positives = all_bound ∩ OmniATAC open (1 kb bins overlapping IDR-optimal peaks)
  negatives = all_candidates unchanged (full zero-bulk non-TF-bound pool)

Reads existing downstream/bins_<tf>_all_universe/pos_neg_bins.tsv and writes:
  downstream/bins_<tf>_all_universe_openpos/pos_neg_bins.{tsv,meta.json}
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "unified" / "scripts"))
from _regions import open_chromatin_bin_mask  # noqa: E402

PANEL = ["ctcf", "maz", "zic2", "zbtb7a", "znf777", "nfya", "znf282"]
MODE = "all_universe_openpos"
OPEN_BED = REPO / "data" / "HEK293T_OmniATAC_idr_optimal.narrowPeak.gz"


def main() -> int:
    if not OPEN_BED.is_file():
        raise SystemExit(f"missing open chromatin BED: {OPEN_BED}")

    with gzip.open(REPO / "unified" / "work" / "ctcf" / "mm" / "regions.tsv.gz", "rt") as fh:
        regions = [ln.strip() for ln in fh]
    open_mask = open_chromatin_bin_mask(regions, OPEN_BED)
    n_open = int(open_mask.sum())
    print(f"open bins: {n_open} / {len(regions)} from {OPEN_BED}")

    for tf in PANEL:
        src = ROOT / f"bins_{tf}_all_universe" / "pos_neg_bins.tsv"
        if not src.is_file():
            raise SystemExit(f"missing {src}")
        out_dir = ROOT / f"bins_{tf}_{MODE}"
        out_dir.mkdir(parents=True, exist_ok=True)

        header = None
        kept_lines: list[str] = []
        n_pos_in = n_neg_in = n_pos = n_neg = 0
        with src.open() as fh:
            header = fh.readline().rstrip("\n")
            cols = header.split("\t")
            i_idx = cols.index("bin_idx_orig")
            i_lab = cols.index("label")
            for ln in fh:
                p = ln.rstrip("\n").split("\t")
                bi = int(p[i_idx])
                lab = p[i_lab].lower()
                if lab.startswith("pos"):
                    n_pos_in += 1
                    if not open_mask[bi]:
                        continue
                    n_pos += 1
                elif lab.startswith("neg"):
                    n_neg_in += 1
                    n_neg += 1
                else:
                    continue
                kept_lines.append(ln if ln.endswith("\n") else ln + "\n")

        out_tsv = out_dir / "pos_neg_bins.tsv"
        with out_tsv.open("w") as fh:
            fh.write(header + "\n")
            fh.writelines(kept_lines)

        meta = {
            "tf": tf,
            "mode": MODE,
            "positive_mode": "all_bound_open",
            "negative_mode": "all_candidates",
            "open_chromatin_bed": str(OPEN_BED),
            "n_open_bins_genome": n_open,
            "n_pos_all_bound": n_pos_in,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_neg_source": n_neg_in,
            "source_bins_tsv": str(src),
        }
        (out_dir / "pos_neg_bins.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(
            f"[{tf}] pos {n_pos_in} -> {n_pos} (open); neg kept {n_neg} "
            f"-> {out_tsv}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
