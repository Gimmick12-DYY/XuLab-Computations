#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 06_export_bedgraph.py
#
# Pseudobulk the cisTopic imputed CSR across cells into a sorted bedGraph that
# can be converted to bigWig for browser visualisation:
#
#   pseudobulk[r] = sum_c imputed[r, c]      (default; mode = "sum")
#                 = #{c : imputed[r, c] > 0} (mode = "count")
#                 = imputed[r, c0]           (mode = "cell:<barcode>" for a single cell)
#
# bedGraph format (one row per region):
#   chrom \t start \t end \t value
#
# To turn it into a bigWig (UCSC tools):
#   bedGraphToBigWig pseudobulk.bedgraph hg38.chrom.sizes pseudobulk.bw
#
# A small companion hg38.chrom.sizes can be downloaded with:
#   fetchChromSizes hg38 > hg38.chrom.sizes
#
# Optional --groups TSV (columns: barcode, group) lets you emit one bedGraph
# per group (e.g. one per cluster, one per timepoint). Cells absent from the
# groups TSV are dropped from the pseudobulk.
#
# Inputs (under <work>/impute/, written by 05_impute.py):
#   matrix_csr.npz   sparse CSR float32 (n_mm_bins, n_mm_cells)
#   regions.tsv      one chr:start-end per row
#   barcodes.tsv     one barcode per col
#
# Outputs (under <work>/bigwig/):
#   <out>.bedgraph                   genome-wide pseudobulk (default)
#   <out>.<mode>.bedgraph            if --mode count or cell:X is used
#   <out>.<group>.bedgraph           one per group when --groups TSV is given
#   bedgraph_meta.json               summary
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

log = logging.getLogger("06_bedgraph")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_REGION_RE = re.compile(r"^([^:]+):(\d+)-(\d+)$")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _read_lines(path: Path) -> list[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            return [ln.rstrip("\n") for ln in fh if ln.strip()]
    return [ln for ln in path.read_text().splitlines() if ln.strip()]


def _parse_region(name: str) -> tuple[str, int, int]:
    m = _REGION_RE.match(name)
    if not m:
        raise SystemExit(f"Cannot parse region {name!r} as chr:start-end")
    return m.group(1), int(m.group(2)), int(m.group(3))


def _coord_sort_indices(parsed: list[tuple[str, int, int]]) -> np.ndarray:
    """Return the permutation that sorts regions by (chrom_lex, start, end).

    Lexicographic chromosome order is what `bedGraphToBigWig` requires --
    UCSC `hg38.chrom.sizes` is in lex order (chr1, chr10, chr11, ..., chr2,
    chr20, ...). Sorting numerically (chr1, chr2, ..., chr22, chrX) writes a
    bedGraph that bedGraphToBigWig rejects with:
        Error: ... chromosome chr10 found before chrom chr2 in chrom.sizes
    so we match `sort -k1,1 -k2,2n` here instead.
    """
    keys = [(c, s, e) for (c, s, e) in parsed]
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    return np.asarray(order, dtype=np.int64)


def _write_bedgraph(out_path: Path, parsed_regions: list[tuple[str, int, int]],
                    values: np.ndarray, sort_order: np.ndarray) -> int:
    nz = 0
    with open(out_path, "w") as fh:
        for i in sort_order:
            v = float(values[i])
            if v == 0.0 or np.isnan(v):
                continue
            chrom, start, end = parsed_regions[i]
            fh.write(f"{chrom}\t{start}\t{end}\t{v:.6g}\n")
            nz += 1
    return nz


def _load_groups(path: Path, barcodes: list[str]
                  ) -> dict[str, np.ndarray]:
    """Return {group_label: 1-D int64 array of col indices} from a TSV
    with a header row containing at least `barcode` and `group`."""
    bar_to_idx = {b: i for i, b in enumerate(barcodes)}
    groups: dict[str, list[int]] = {}
    n_missing = 0
    n_kept = 0
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        try:
            i_bc = header.index("barcode")
            i_gr = header.index("group")
        except ValueError as e:
            raise SystemExit(f"Bad header in {path}: {e}")
        for ln in fh:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) <= max(i_bc, i_gr):
                continue
            bc = parts[i_bc].strip()
            gr = parts[i_gr].strip()
            if not bc or not gr:
                continue
            idx = bar_to_idx.get(bc)
            if idx is None:
                n_missing += 1
                continue
            groups.setdefault(gr, []).append(idx)
            n_kept += 1
    if n_missing:
        log.warning("  groups TSV: %d barcodes not in barcodes.tsv (ignored).",
                    n_missing)
    log.info("  groups: %d labels, %d barcodes assigned",
             len(groups), n_kept)
    return {g: np.asarray(v, dtype=np.int64) for g, v in groups.items()}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__ or "",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--impute-dir", type=Path, required=True,
                    help="cisTopic <work>/impute/ directory containing "
                         "matrix_csr.npz + regions.tsv + barcodes.tsv.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--out-name", type=str, default="pseudobulk")
    ap.add_argument("--mode", type=str, default="sum",
                    help="Pseudobulk operator: 'sum' (default), 'count' "
                         "(number of cells with value > 0 at the region), or "
                         "'cell:<barcode>' to extract a single cell's track.")
    ap.add_argument("--groups", type=Path, default=None,
                    help="Optional TSV with header 'barcode\\tgroup'. Writes "
                         "one bedGraph per distinct group label.")
    ap.add_argument("--include-zero", action="store_true",
                    help="Also write rows with value == 0 (default skips them; "
                         "bedGraph readers handle implicit zeros fine).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    mat_path  = args.impute_dir / "matrix_csr.npz"
    reg_path  = args.impute_dir / "regions.tsv"
    bar_path  = args.impute_dir / "barcodes.tsv"
    for p in (mat_path, reg_path, bar_path):
        if not p.exists():
            log.error("Missing %s; run 05_impute.py first.", p)
            return 1
    log.info("Loading CSR %s", mat_path)
    mat = sparse.load_npz(mat_path).tocsr()
    regions = _read_lines(reg_path)
    barcodes = _read_lines(bar_path)
    if mat.shape != (len(regions), len(barcodes)):
        log.error("Shape mismatch: matrix %s vs regions/barcodes %d/%d",
                  mat.shape, len(regions), len(barcodes))
        return 1
    log.info("  shape=%s, nnz=%d (density=%.3g)",
             mat.shape, mat.nnz,
             mat.nnz / max(int(mat.shape[0]) * int(mat.shape[1]), 1))

    parsed = [_parse_region(r) for r in regions]
    sort_order = _coord_sort_indices(parsed)

    out_meta: dict = {
        "impute_dir":   str(args.impute_dir),
        "matrix":       str(mat_path),
        "shape":        [int(mat.shape[0]), int(mat.shape[1])],
        "nnz":          int(mat.nnz),
        "mode":         args.mode,
        "outputs":      [],
    }

    # --------------------------------------------------------------------
    # Decide what to pseudobulk and emit.
    # --------------------------------------------------------------------
    todo: list[tuple[str, np.ndarray, str]] = []   # (label, col_indices, file_stem)

    mode = args.mode.strip()
    if args.groups is not None:
        log.info("Reading groups TSV %s", args.groups)
        groups = _load_groups(args.groups, barcodes)
        for g, cols in sorted(groups.items()):
            todo.append((f"group:{g}", cols, f"{args.out_name}.{g}"))
    elif mode.startswith("cell:"):
        bc = mode.split(":", 1)[1].strip()
        try:
            idx = barcodes.index(bc)
        except ValueError:
            log.error("Barcode %r not in barcodes.tsv", bc)
            return 1
        todo.append((f"cell:{bc}",
                     np.asarray([idx], dtype=np.int64),
                     f"{args.out_name}.{bc}"))
    else:
        # Global pseudobulk over all cells.
        todo.append(("all_cells",
                     np.arange(mat.shape[1], dtype=np.int64),
                     args.out_name))

    # --------------------------------------------------------------------
    # For each (label, col_indices, file_stem), compute the pseudobulk vector
    # over regions and write a bedGraph.
    # --------------------------------------------------------------------
    for label, col_idx, stem in todo:
        if col_idx.size == 0:
            log.warning("Empty group %r; skipping.", label)
            continue
        sub = mat[:, col_idx]
        if mode == "count" or (args.groups is not None and mode == "count"):
            # Number of contributing cells with a positive entry per region.
            values = np.asarray((sub > 0).sum(axis=1)).ravel().astype(np.float64)
            op = "count"
        else:
            # Sum across the selected cells.
            values = np.asarray(sub.sum(axis=1)).ravel().astype(np.float64)
            op = "sum" if mode != "count" else "count"

        if not args.include_zero:
            n_nonzero = int((values != 0).sum())
        else:
            n_nonzero = int(values.size)
        log.info("  [%s] n_cells=%d  op=%s  regions with signal=%d / %d",
                 label, col_idx.size, op, n_nonzero, values.size)

        out_path = args.out_dir / f"{stem}.bedgraph"
        n_written = _write_bedgraph(out_path, parsed, values, sort_order)
        log.info("  -> %s (%d rows)", out_path, n_written)

        # Also dump per-region values as a TSV for re-plotting / sanity checks.
        tsv_out = args.out_dir / f"{stem}.values.tsv"
        with open(tsv_out, "w") as fh:
            fh.write("chrom\tstart\tend\tvalue\n")
            for i in sort_order:
                if not args.include_zero and values[i] == 0.0:
                    continue
                chrom, s, e = parsed[i]
                fh.write(f"{chrom}\t{s}\t{e}\t{float(values[i]):.6g}\n")
        out_meta["outputs"].append({
            "label":       label,
            "n_cells":     int(col_idx.size),
            "operator":    op,
            "bedgraph":    str(out_path),
            "values_tsv":  str(tsv_out),
            "n_nonzero":   n_written,
        })

    (args.out_dir / "bedgraph_meta.json").write_text(
        json.dumps(out_meta, indent=2) + "\n"
    )
    log.info("Done.")
    log.info("To convert a bedGraph to bigWig (UCSC tools):")
    log.info("  bedGraphToBigWig <out>.bedgraph hg38.chrom.sizes <out>.bw")
    return 0


if __name__ == "__main__":
    sys.exit(main())
