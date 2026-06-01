#!/usr/bin/env python
# -----------------------------------------------------------------------------
# eval_basic.py
#
# Metadata-aware factual-numbers comparison across imputation pipelines.
#
# Reads one or more `<work>/impute/` directories produced by scBasset /
# scBasset_TF / Borzoi / etc. and the AllTF metadata TSV, then writes two
# tables:
#
#   summary.tsv   one row per run, global counts.
#   per_tf.tsv    one row per (run x TF), counts restricted to that TF's cells.
#
# No fancy metrics. Just countable, directly comparable numbers:
#   nnz_raw, nnz_imputed, nnz_gain, gain_pct_over_raw, new_bins_filled,
#   mean/median per-cell densities (raw + imputed).
#
# Metadata convention (matches TF1000cells.meta):
#   columns include `barcode` (= "<DNA_id>:<cell>") and `TF`.
#   `barcode` is matched against impute/barcodes.tsv. Cells without a
#   matching row in metadata are bucketed as "_unknown".
#
# Usage:
#   python downstream/eval_basic.py \
#     --input scbasset_bin=/work/.../scBasset/work/alltf_bin/impute \
#     --input scbasset_peak=/work/.../scBasset/work/alltf_peak/impute \
#     --input scbasset_tf_bin=/work/.../scBasset_TF/work/alltf_bin/impute \
#     --input scbasset_tf_peak=/work/.../scBasset_TF/work/alltf_peak/impute \
#     --metadata /work/.../data/TF1000cells.meta.tsv \
#     --out-dir /work/.../comparison/alltf
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

log = logging.getLogger("eval_basic")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def read_lines(p: Path) -> list[str]:
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rt") as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip()]


def load_metadata(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return (barcode->TF, barcode->subset). Looks for `barcode`/`TF`/`subset`
    columns by name. Falls back to constructing barcode from DNA_id + cell
    if a `barcode` column is absent."""
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        rows = [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]

    def col(name: str) -> int | None:
        return header.index(name) if name in header else None

    bc_col   = col("barcode")
    tf_col   = col("TF") or col("tf")
    sub_col  = col("subset")
    dna_col  = col("DNA_id")
    cell_col = col("cell")

    if tf_col is None:
        raise SystemExit(f"metadata {path}: no 'TF' column. header={header}")

    bc_to_tf: dict[str, str] = {}
    bc_to_sub: dict[str, str] = {}
    for r in rows:
        # pad short rows
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))
        if bc_col is not None and r[bc_col]:
            bc = r[bc_col]
        elif dna_col is not None and cell_col is not None:
            bc = f"{r[dna_col]}:{r[cell_col]}"
        else:
            continue
        bc_to_tf[bc] = r[tf_col] or "_unset"
        if sub_col is not None:
            bc_to_sub[bc] = r[sub_col]
    return bc_to_tf, bc_to_sub


def find_mm_dir(impute_dir: Path, meta: dict) -> Path | None:
    """Try common locations for the source mm/ matrix."""
    candidates: list[Path] = []
    if "mm_dir" in meta and meta["mm_dir"]:
        candidates.append(Path(meta["mm_dir"]))
    candidates.append(impute_dir.parent / "mm")
    if isinstance(meta.get("sources"), dict):
        for v in meta["sources"].values():
            try:
                p = Path(v)
                if p.is_file():
                    candidates.append(p.parent)
                else:
                    candidates.append(p)
            except Exception:
                pass
    for c in candidates:
        if (c / "matrix.mtx.gz").is_file():
            return c
    return None


def summarize_one(label: str, impute_dir: Path,
                  bc_to_tf: dict[str, str]) -> tuple[dict, list[dict]]:
    log.info("[%s] reading %s", label, impute_dir)
    if not (impute_dir / "matrix_csr.npz").is_file():
        raise SystemExit(f"{label}: missing {impute_dir / 'matrix_csr.npz'}")
    mat_imp = sp.load_npz(impute_dir / "matrix_csr.npz").tocsr().astype(np.float32)
    barcodes = read_lines(impute_dir / "barcodes.tsv")
    meta = json.loads((impute_dir / "meta.json").read_text()) if (impute_dir / "meta.json").is_file() else {}

    n_reg, n_cells = mat_imp.shape
    if len(barcodes) != n_cells:
        log.warning("[%s] barcodes (%d) != matrix cols (%d); using matrix cols",
                    label, len(barcodes), n_cells)
        barcodes = barcodes[:n_cells] + [f"_extra_{i}" for i in range(len(barcodes), n_cells)]

    nnz_imp = int(mat_imp.nnz)

    mm_dir = find_mm_dir(impute_dir, meta)
    if mm_dir is None:
        log.warning("[%s] no mm/ found; raw-derived numbers will be blank", label)
        mat_raw = None
        nnz_raw = 0
        col_raw = None
        new_bins = None
    else:
        log.info("[%s] raw from %s", label, mm_dir)
        mat_raw = sio.mmread(mm_dir / "matrix.mtx.gz").tocsr()
        if mat_raw.shape != mat_imp.shape:
            log.warning("[%s] raw shape %s != imp shape %s; skipping raw metrics",
                        label, mat_raw.shape, mat_imp.shape)
            mat_raw = None
            nnz_raw = 0
            col_raw = None
            new_bins = None
        else:
            nnz_raw = int((mat_raw != 0).sum())
            col_raw = np.asarray((mat_raw != 0).sum(axis=0)).ravel()
            row_imp = np.asarray((mat_imp != 0).sum(axis=1)).ravel()
            row_raw = np.asarray((mat_raw != 0).sum(axis=1)).ravel()
            new_bins = int(((row_raw == 0) & (row_imp > 0)).sum())

    col_imp = np.asarray((mat_imp != 0).sum(axis=0)).ravel()

    summary = {
        "run":                  label,
        "input_kind":           meta.get("input_kind") or meta.get("pipeline") or "?",
        "n_regions":            n_reg,
        "n_cells":              n_cells,
        "nnz_raw":              nnz_raw if mat_raw is not None else "",
        "nnz_imputed":          nnz_imp,
        "nnz_gain":             (nnz_imp - nnz_raw) if mat_raw is not None else "",
        "gain_pct_over_raw":    (round(100.0 * (nnz_imp - nnz_raw) / nnz_raw, 2)
                                 if (mat_raw is not None and nnz_raw > 0) else ""),
        "new_bins_filled":      new_bins if new_bins is not None else "",
        "mean_density_per_cell_raw":      (round(float(col_raw.mean()), 3) if col_raw is not None else ""),
        "median_density_per_cell_raw":    (int(np.median(col_raw)) if col_raw is not None else ""),
        "mean_density_per_cell_imputed":  round(float(col_imp.mean()), 3),
        "median_density_per_cell_imputed": int(np.median(col_imp)),
        "mm_dir":               str(mm_dir) if mm_dir else "",
        "impute_dir":           str(impute_dir),
    }

    # Per-TF stratification
    cell_tf = np.array([bc_to_tf.get(b, "_unknown") for b in barcodes], dtype=object)
    tf_known = int((cell_tf != "_unknown").sum())
    log.info("[%s] %d / %d cells have a TF assignment", label, tf_known, len(barcodes))
    unique_tfs = sorted(set(cell_tf.tolist()))

    per_tf_rows: list[dict] = []
    for tf in unique_tfs:
        cell_mask = cell_tf == tf
        n_tf = int(cell_mask.sum())
        if n_tf == 0:
            continue
        sub_imp = mat_imp[:, cell_mask]
        nnz_imp_tf = int(sub_imp.nnz)
        col_imp_tf = np.asarray((sub_imp != 0).sum(axis=0)).ravel()
        if mat_raw is not None:
            sub_raw = mat_raw[:, cell_mask]
            nnz_raw_tf = int((sub_raw != 0).sum())
            col_raw_tf = np.asarray((sub_raw != 0).sum(axis=0)).ravel()
            mean_raw = round(float(col_raw_tf.mean()), 3)
            median_raw = int(np.median(col_raw_tf))
            new_bins_tf = int((
                (np.asarray((sub_raw != 0).sum(axis=1)).ravel() == 0) &
                (np.asarray((sub_imp != 0).sum(axis=1)).ravel() >  0)
            ).sum())
        else:
            nnz_raw_tf = ""
            mean_raw = ""
            median_raw = ""
            new_bins_tf = ""
        per_tf_rows.append({
            "run":                              label,
            "TF":                               tf,
            "n_cells":                          n_tf,
            "nnz_raw":                          nnz_raw_tf,
            "nnz_imputed":                      nnz_imp_tf,
            "nnz_gain":                         (nnz_imp_tf - nnz_raw_tf)
                                                if isinstance(nnz_raw_tf, int) else "",
            "new_bins_filled":                  new_bins_tf,
            "mean_density_per_cell_raw":        mean_raw,
            "median_density_per_cell_raw":      median_raw,
            "mean_density_per_cell_imputed":    round(float(col_imp_tf.mean()), 3),
            "median_density_per_cell_imputed":  int(np.median(col_imp_tf)),
        })

    return summary, per_tf_rows


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    lines = ["\t".join(cols)]
    for r in rows:
        lines.append("\t".join(str(r.get(c, "")) for c in cols))
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "")
    ap.add_argument("--input", action="append", required=True,
                    metavar="LABEL=PATH",
                    help="Repeatable: label=/path/to/impute_dir")
    ap.add_argument("--metadata", required=True,
                    help="TSV with columns including barcode + TF")
    ap.add_argument("--out-dir", required=True,
                    help="Destination for summary.tsv and per_tf.tsv")
    args = ap.parse_args()

    runs: list[tuple[str, Path]] = []
    for s in args.input:
        if "=" not in s:
            log.error("--input must be label=path; got %r", s); return 1
        label, path = s.split("=", 1)
        runs.append((label, Path(path).expanduser().resolve()))

    metadata_path = Path(args.metadata).expanduser().resolve()
    if not metadata_path.is_file():
        log.error("metadata not found: %s", metadata_path); return 1
    log.info("Loading metadata %s", metadata_path)
    bc_to_tf, _bc_to_sub = load_metadata(metadata_path)
    n_tfs = len(set(bc_to_tf.values()))
    log.info("Metadata: %d barcode->TF rows, %d distinct TFs",
             len(bc_to_tf), n_tfs)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    per_tf_rows:  list[dict] = []
    for label, impute_dir in runs:
        s, pt = summarize_one(label, impute_dir, bc_to_tf)
        summary_rows.append(s)
        per_tf_rows.extend(pt)

    write_tsv(out_dir / "summary.tsv", summary_rows)
    write_tsv(out_dir / "per_tf.tsv",  per_tf_rows)
    log.info("Wrote %s (%d rows)", out_dir / "summary.tsv", len(summary_rows))
    log.info("Wrote %s (%d rows)", out_dir / "per_tf.tsv",  len(per_tf_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
