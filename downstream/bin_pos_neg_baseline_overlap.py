#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# bin_pos_neg_baseline_overlap.py
#
# Simple baseline: how many mm rows (bins) have any non-zero signal in raw vs
# selected imputed CSRs, and how those row sets intersect the positive /
# negative evaluation bins from prepare_pos_neg_bins.R (same schema as
# Bin_posVSneg.ipynb → downstream/bins/pos_neg_bins.tsv).
#
# See ``imputed_vs_raw`` in the JSON for bin-rows where the imputed CSR has a
# stored entry but the raw row was all-zero (additive bin opening on that view).
#
# Row index is bin_idx_orig (0-based into mm/regions.tsv.gz), aligned with
# matrix_csr.npz rows when impute uses align_to_mm.
#
# Usage:
#   python downstream/bin_pos_neg_baseline_overlap.py \
#     --mm-dir /work/.../cistopic_ctcf/mm \
#     --bins-tsv /work/.../downstream/bins/pos_neg_bins.tsv \
#     --input cisTopic=/work/.../cisTopic/.../impute \
#     --input PUscOpen=/work/.../PUscOpen/.../impute \
#     --input cicero=/work/.../Cicero/work/ctcf/impute \
#     --input scbasset=/work/.../scBasset/work/ctcf/impute \
#     --out-json /work/.../downstream/bins/baseline_pos_neg_overlap.json
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import sparse

log = logging.getLogger("bin_pos_neg_baseline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_input_pair(s: str) -> tuple[str, Path]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"Expected LABEL=PATH, got: {s!r}")
    label, path = s.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"Empty label in: {s!r}")
    return label, Path(path)


def load_bins_pos_neg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    pos_idx: list[int] = []
    neg_idx: list[int] = []
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        i_orig = header.index("bin_idx_orig")
        i_lab = header.index("label")
        for ln in fh:
            row = ln.rstrip("\n").split("\t")
            if not row or not row[0]:
                continue
            idx = int(row[i_orig])
            if row[i_lab] == "pos":
                pos_idx.append(idx)
            else:
                neg_idx.append(idx)
    return np.asarray(pos_idx, dtype=np.int64), np.asarray(neg_idx, dtype=np.int64)


def row_has_nnz_csr(mat: sparse.csr_matrix) -> np.ndarray:
    """Boolean (n_rows,) — True if that row has at least one stored entry."""
    mat = mat.tocsr()
    nnz_per_row = np.diff(mat.indptr)
    return nnz_per_row > 0


def row_has_nnz_mm(mm_dir: Path) -> tuple[np.ndarray, int, int]:
    mtx = mm_dir / "matrix.mtx.gz"
    if not mtx.exists():
        raise SystemExit(f"Missing {mtx}")
    log.info("Loading raw MM %s", mtx)
    mat = sio.mmread(str(mtx)).tocsr()
    n = int(mat.shape[0])
    return row_has_nnz_csr(mat), n, int(mat.nnz)


def row_has_nnz_impute_csr(impute_dir: Path, n_expect: int) -> tuple[np.ndarray, int, str]:
    p = impute_dir / "matrix_csr.npz"
    r = impute_dir / "regions.tsv"
    if not p.exists():
        raise SystemExit(f"Missing {p} (need CSR impute dir)")
    if not r.exists():
        raise SystemExit(f"Missing {r}")
    log.info("Loading CSR %s", p)
    mat = sparse.load_npz(p).tocsr()
    if mat.shape[0] != n_expect:
        raise SystemExit(
            f"{impute_dir}: CSR rows {mat.shape[0]} != mm rows {n_expect}"
        )
    return row_has_nnz_csr(mat), int(mat.nnz), str(p.resolve())


def summarize(
    name: str,
    mask: np.ndarray,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    *,
    total_stored_nnz: int | None = None,
    csr_path: str | None = None,
    impute_dir: str | None = None,
) -> dict:
    n_nnz = int(mask.sum())
    n_pos_hit = int(mask[pos_idx].sum())
    n_neg_hit = int(mask[neg_idx].sum())
    n_pos = int(pos_idx.size)
    n_neg = int(neg_idx.size)
    out = {
        "source": name,
        "n_bins_any_signal": n_nnz,
        "n_pos_bins_in_panel": n_pos,
        "n_neg_bins_in_panel": n_neg,
        "n_intersection_pos": n_pos_hit,
        "n_intersection_neg": n_neg_hit,
        "frac_pos_panel_covered": (n_pos_hit / n_pos) if n_pos else None,
        "frac_neg_panel_covered": (n_neg_hit / n_neg) if n_neg else None,
        "frac_nnz_that_are_pos_hits": (n_pos_hit / n_nnz) if n_nnz else None,
        "frac_nnz_that_are_neg_hits": (n_neg_hit / n_nnz) if n_nnz else None,
    }
    if total_stored_nnz is not None:
        out["total_stored_nnz"] = total_stored_nnz
    if csr_path is not None:
        out["csr_path"] = csr_path
    if impute_dir is not None:
        out["impute_dir"] = impute_dir
    return out


def summarize_added_bin_rows(
    label: str,
    mask_raw: np.ndarray,
    mask_imp: np.ndarray,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
) -> dict:
    """Bin rows: imputed CSR has a stored entry in that row, raw row was all-zero."""
    added = mask_imp & (~mask_raw)
    n_add = int(added.sum())
    n_pos = int(pos_idx.size)
    n_neg = int(neg_idx.size)
    n_pos_hit = int(added[pos_idx].sum())
    n_neg_hit = int(added[neg_idx].sum())
    return {
        "imputed_source": label,
        "n_bin_rows_raw_silent_imputed_on": n_add,
        "n_intersection_pos_panel": n_pos_hit,
        "n_intersection_neg_panel": n_neg_hit,
        "frac_pos_panel_among_added": (n_pos_hit / n_add) if n_add else None,
        "frac_neg_panel_among_added": (n_neg_hit / n_add) if n_add else None,
        "frac_pos_panel_that_are_added": (n_pos_hit / n_pos) if n_pos else None,
        "frac_neg_panel_that_are_added": (n_neg_hit / n_neg) if n_neg else None,
    }


def pairwise_row_intersections(
    labels: list[str],
    row_sets: dict[str, np.ndarray],
) -> dict[str, int]:
    out: dict[str, int] = {}
    for a, b in combinations(labels, 2):
        key = f"{a}_{b}"
        out[key] = int(
            np.intersect1d(row_sets[a], row_sets[b], assume_unique=True).size
        )
    if len(labels) >= 3:
        inter = row_sets[labels[0]]
        for lab in labels[1:]:
            inter = np.intersect1d(inter, row_sets[lab], assume_unique=True)
        out["all_imputed"] = int(inter.size)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mm-dir", type=Path, required=True,
                    help="Raw MatrixMarket dir (matrix.mtx.gz + regions.tsv.gz)")
    ap.add_argument("--bins-tsv", type=Path, required=True,
                    help="pos_neg_bins.tsv from prepare_pos_neg_bins.R")
    ap.add_argument("--input", type=parse_input_pair, action="append", default=[],
                    metavar="LABEL=PATH",
                    help="Impute dir with matrix_csr.npz + regions.tsv (repeatable)")
    ap.add_argument("--cis-topic-impute", type=Path, default=None,
                    help="(deprecated) alias for --input cisTopic=PATH")
    ap.add_argument("--puscopen-impute", type=Path, default=None,
                    help="(deprecated) alias for --input PUscOpen=PATH")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="Write summary JSON (default: <bins-dir>/baseline_pos_neg_overlap.json)")
    args = ap.parse_args()

    inputs: list[tuple[str, Path]] = list(args.input)
    if args.cis_topic_impute is not None:
        inputs.append(("cisTopic", args.cis_topic_impute))
    if args.puscopen_impute is not None:
        inputs.append(("PUscOpen", args.puscopen_impute))
    if not inputs:
        ap.error("Provide at least one --input LABEL=PATH (or legacy --cis-topic-impute / --puscopen-impute)")

    pos_idx, neg_idx = load_bins_pos_neg(args.bins_tsv)
    log.info("Loaded %d pos + %d neg bins from %s",
             pos_idx.size, neg_idx.size, args.bins_tsv)

    mask_raw, n_rows, nnz_raw = row_has_nnz_mm(args.mm_dir)
    if pos_idx.size and int(pos_idx.max()) >= n_rows:
        raise SystemExit(f"pos bin_idx max {pos_idx.max()} >= n_rows {n_rows}")
    if neg_idx.size and int(neg_idx.max()) >= n_rows:
        raise SystemExit(f"neg bin_idx max {neg_idx.max()} >= n_rows {n_rows}")

    rows = [
        summarize("raw", mask_raw, pos_idx, neg_idx, total_stored_nnz=nnz_raw),
    ]
    imputed_vs_raw: dict[str, dict] = {}
    row_sets: dict[str, np.ndarray] = {"raw": np.flatnonzero(mask_raw)}
    impute_meta: dict[str, dict] = {}

    for label, imp_dir in inputs:
        mask_imp, nnz_imp, path_imp = row_has_nnz_impute_csr(imp_dir, n_rows)
        rows.append(
            summarize(
                label, mask_imp, pos_idx, neg_idx,
                total_stored_nnz=nnz_imp,
                csr_path=path_imp,
                impute_dir=str(imp_dir.resolve()),
            )
        )
        imputed_vs_raw[label] = summarize_added_bin_rows(
            label, mask_raw, mask_imp, pos_idx, neg_idx,
        )
        row_sets[label] = np.flatnonzero(mask_imp)
        impute_meta[label] = {
            "impute_dir": str(imp_dir.resolve()),
            "csr_path": path_imp,
        }

    imp_labels = [lab for lab, _ in inputs]
    pairwise_raw: dict[str, int] = {}
    for lab in imp_labels:
        pairwise_raw[f"raw_{lab}"] = int(
            np.intersect1d(row_sets["raw"], row_sets[lab], assume_unique=True).size
        )
    pairwise_imp = pairwise_row_intersections(imp_labels, row_sets)

    out = {
        "mm_dir": str(args.mm_dir.resolve()),
        "bins_tsv": str(args.bins_tsv.resolve()),
        "inputs": {lab: str(p.resolve()) for lab, p in inputs},
        "impute_meta": impute_meta,
        "n_mm_rows": n_rows,
        "interpretation": (
            "imputed_vs_raw counts bin-rows with no raw fragment in any cell "
            "but at least one imputed CSR nonzero in that row. "
            "total_stored_nnz is global stored entries in each CSR."
        ),
        "pairwise_nnz_row_intersections": {**pairwise_raw, **pairwise_imp},
        "imputed_vs_raw": imputed_vs_raw,
        "per_source": rows,
    }

    out_path = args.out_json
    if out_path is None:
        out_path = args.bins_tsv.parent / "baseline_pos_neg_overlap.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    log.info("Wrote %s", out_path)

    for r in rows:
        log.info(
            "%s: any-signal bins=%d  stored_nnz=%d  ∩ pos=%d (%.2f%% of pos panel)  "
            "∩ neg=%d (%.2f%% of neg panel)",
            r["source"],
            r["n_bins_any_signal"],
            int(r.get("total_stored_nnz", 0)),
            r["n_intersection_pos"],
            100.0 * (r["frac_pos_panel_covered"] or 0.0),
            r["n_intersection_neg"],
            100.0 * (r["frac_neg_panel_covered"] or 0.0),
        )
    for label in imp_labels:
        d = out["imputed_vs_raw"][label]
        log.info(
            "%s vs raw — bin-rows raw-silent but imputed on: %d  "
            "(∩ pos panel=%d, ∩ neg panel=%d)",
            label,
            d["n_bin_rows_raw_silent_imputed_on"],
            d["n_intersection_pos_panel"],
            d["n_intersection_neg_panel"],
        )
    log.info("Pairwise nnz-row intersections: %s", out["pairwise_nnz_row_intersections"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
