#!/usr/bin/env python
"""How much does each ATAC mask expand credible (open) positives, per TF?

The `all_universe_openpos` eval restricts TF positives to open chromatin. This
script quantifies the effect of swapping / unioning masks: for each TF's positive
bins it reports the count and fraction that fall in open chromatin under each
mask (OmniATAC alone, each validation set, and the union).

Positives come from the downstream bins TSVs (e.g. bins_<tf>_all_universe/
pos_neg_bins.tsv). Region column ('chr:start-end') and the positive-label column
are auto-detected; override with --region-col / --label-col / --pos-value.

Usage:
  python downstream_open_positive_recovery.py \
    --pos ctcf=/work/.../downstream/bins_ctcf_all_universe/pos_neg_bins.tsv \
    --pos maz=/work/.../downstream/bins_maz_all_universe/pos_neg_bins.tsv \
    --mask omniatac=/work/.../data/HEK293T_OmniATAC_idr_optimal.narrowPeak.gz \
    --mask gse283384=/work/.../data/HEK293T_ATAC_gse283384_wt_MACS2.narrowPeak.gz \
    --mask gse152177=/work/.../data/HEK293T_ATAC_gse152177_wt_MACS2.narrowPeak.gz \
    --mask union=/work/.../data/HEK293T_ATAC_union3.bed.gz \
    --out /work/.../atac_validation/results/open_positive_recovery.tsv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import re
from pathlib import Path

import numpy as np

from _maskutil import bin_mask_for_peaks, region_arrays

_REGION_RE = re.compile(r"^[^:]+:\d+-\d+$")
_LABEL_CANDIDATES = ["label", "y", "class", "is_pos", "positive", "pos", "bound", "target"]
_POS_TRUE = {"1", "1.0", "pos", "positive", "true", "yes", "bound"}


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def load_positive_regions(path: str, region_col: str | None,
                          label_col: str | None, pos_value: str | None) -> list[str]:
    """Return the list of positive region names from a bins TSV."""
    with _open(path) as fh:
        sample = fh.read(4096)
    delim = "\t" if "\t" in sample else ","
    with _open(path) as fh:
        reader = csv.reader(fh, delimiter=delim)
        header = next(reader)
        header = [h.strip() for h in header]

        # region column
        if region_col and region_col in header:
            ridx = header.index(region_col)
        else:
            ridx = None
            # header token match
            for cand in ("region", "regions", "name", "bin", "peak"):
                if cand in [h.lower() for h in header]:
                    ridx = [h.lower() for h in header].index(cand)
                    break

        # label column
        if label_col and label_col in header:
            lidx = header.index(label_col)
        else:
            lidx = None
            lower = [h.lower() for h in header]
            for cand in _LABEL_CANDIDATES:
                if cand in lower:
                    lidx = lower.index(cand)
                    break

        pos_true = {pos_value.lower()} if pos_value else _POS_TRUE

        regions: list[str] = []
        first = True
        for row in reader:
            if not row:
                continue
            # infer region column from data if header didn't name it
            if ridx is None:
                for j, cell in enumerate(row):
                    if _REGION_RE.match(cell.strip()):
                        ridx = j
                        break
                if ridx is None:
                    raise SystemExit(f"could not find a chr:start-end column in {path}")
            reg = row[ridx].strip()
            if not _REGION_RE.match(reg):
                continue
            if lidx is not None:
                if row[lidx].strip().lower() not in pos_true:
                    continue
            regions.append(reg)
            first = False
        if first:
            raise SystemExit(f"no positive regions selected from {path} "
                             f"(check --label-col/--pos-value)")
    return regions


def parse_kv(items: list[str], must_exist: bool) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"expected name=path, got {it!r}")
        name, path = it.split("=", 1)
        if must_exist and not Path(path).is_file():
            raise SystemExit(f"file not found for {name}: {path}")
        out[name] = path
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", action="append", required=True, metavar="tf=bins.tsv")
    ap.add_argument("--mask", action="append", required=True, metavar="name=peaks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--region-col", default=None)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--pos-value", default=None,
                    help="value marking a positive row (default: any of "
                         "1/pos/positive/true/yes/bound)")
    args = ap.parse_args()

    pos_files = parse_kv(args.pos, must_exist=True)
    masks = parse_kv(args.mask, must_exist=True)
    mask_names = list(masks.keys())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for tf, pf in pos_files.items():
        regions = load_positive_regions(pf, args.region_col, args.label_col, args.pos_value)
        chroms, starts, ends = region_arrays(regions)
        n_pos = len(regions)
        rec = {"tf": tf, "n_pos": n_pos}
        for mn in mask_names:
            m = bin_mask_for_peaks(chroms, starts, ends, masks[mn])
            n_open = int(m.sum())
            rec[f"open_{mn}"] = n_open
            rec[f"pct_{mn}"] = round(100.0 * n_open / n_pos, 2) if n_pos else 0.0
        rows.append(rec)
        summary = "  ".join(f"{mn}={rec[f'pct_{mn}']:.1f}%" for mn in mask_names)
        print(f"[recovery] {tf}: n_pos={n_pos:,}  {summary}")

    cols = ["tf", "n_pos"]
    for mn in mask_names:
        cols += [f"open_{mn}", f"pct_{mn}"]
    with open(out, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"[recovery] wrote {out}")


if __name__ == "__main__":
    main()
