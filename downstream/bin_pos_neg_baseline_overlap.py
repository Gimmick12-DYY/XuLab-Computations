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
#     --cis-topic-impute /work/.../cisTopic/.../impute \
#     --puscopen-impute /work/.../PUscOpen/.../impute \
#     --out-json /work/.../downstream/bins/baseline_pos_neg_overlap.json
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import sparse

log = logging.getLogger("bin_pos_neg_baseline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


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
) -> dict:
    n_nnz = int(mask.sum())
    n_pos_hit = int(mask[pos_idx].sum())
    n_neg_hit = int(mask[neg_idx].sum())
    n_pos = int(pos_idx.size)
    n_neg = int(neg_idx.size)
    return {
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mm-dir", type=Path, required=True,
                    help="Raw MatrixMarket dir (matrix.mtx.gz + regions.tsv.gz)")
    ap.add_argument("--bins-tsv", type=Path, required=True,
                    help="pos_neg_bins.tsv from prepare_pos_neg_bins.R")
    ap.add_argument("--cis-topic-impute", type=Path, required=True,
                    help="cisTopic impute/ with matrix_csr.npz")
    ap.add_argument("--puscopen-impute", type=Path, required=True,
                    help="PUscOpen impute/ with matrix_csr.npz")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="Write machine-readable summary (default: <bins-dir>/baseline_pos_neg_overlap.json)")
    args = ap.parse_args()

    pos_idx, neg_idx = load_bins_pos_neg(args.bins_tsv)
    log.info("Loaded %d pos + %d neg bins from %s",
             pos_idx.size, neg_idx.size, args.bins_tsv)

    mask_raw, n_rows, nnz_raw = row_has_nnz_mm(args.mm_dir)
    if pos_idx.size and int(pos_idx.max()) >= n_rows:
        raise SystemExit(f"pos bin_idx max {pos_idx.max()} >= n_rows {n_rows}")
    if neg_idx.size and int(neg_idx.max()) >= n_rows:
        raise SystemExit(f"neg bin_idx max {neg_idx.max()} >= n_rows {n_rows}")

    mask_ct, nnz_ct, path_ct = row_has_nnz_impute_csr(args.cis_topic_impute, n_rows)
    mask_pu, nnz_pu, path_pu = row_has_nnz_impute_csr(args.puscopen_impute, n_rows)

    rows = [
        summarize("raw", mask_raw, pos_idx, neg_idx),
        summarize("cisTopic", mask_ct, pos_idx, neg_idx),
        summarize("PUscOpen", mask_pu, pos_idx, neg_idx),
    ]
    rows[0]["total_stored_nnz"] = nnz_raw
    rows[1]["total_stored_nnz"] = nnz_ct
    rows[2]["total_stored_nnz"] = nnz_pu

    # Pairwise intersections of bin sets (still mm row indices)
    s_raw = np.flatnonzero(mask_raw)
    s_ct = np.flatnonzero(mask_ct)
    s_pu = np.flatnonzero(mask_pu)
    # Use numpy intersect1d on sorted unique (already unique from flatnonzero)
    n_raw_ct = int(np.intersect1d(s_raw, s_ct, assume_unique=True).size)
    n_raw_pu = int(np.intersect1d(s_raw, s_pu, assume_unique=True).size)
    n_ct_pu = int(np.intersect1d(s_ct, s_pu, assume_unique=True).size)
    n_all_three = int(
        np.intersect1d(
            np.intersect1d(s_raw, s_ct, assume_unique=True),
            s_pu,
            assume_unique=True,
        ).size
    )

    out = {
        "mm_dir": str(args.mm_dir.resolve()),
        "bins_tsv": str(args.bins_tsv.resolve()),
        "cis_topic_impute": str(args.cis_topic_impute.resolve()),
        "cis_topic_csr_path": path_ct,
        "puscopen_impute": str(args.puscopen_impute.resolve()),
        "puscopen_csr_path": path_pu,
        "n_mm_rows": n_rows,
        "interpretation": (
            "imputed_vs_raw counts bin-rows with no raw fragment in any cell "
            "but at least one imputed CSR nonzero in that row. "
            "total_stored_nnz is global stored entries in each CSR."
        ),
        "pairwise_nnz_row_intersections": {
            "raw_cisTopic": n_raw_ct,
            "raw_PUscOpen": n_raw_pu,
            "cisTopic_PUscOpen": n_ct_pu,
            "all_three": n_all_three,
        },
        "imputed_vs_raw": {
            "cisTopic": summarize_added_bin_rows(
                "cisTopic", mask_raw, mask_ct, pos_idx, neg_idx,
            ),
            "PUscOpen": summarize_added_bin_rows(
                "PUscOpen", mask_raw, mask_pu, pos_idx, neg_idx,
            ),
        },
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
    for key in ("cisTopic", "PUscOpen"):
        d = out["imputed_vs_raw"][key]
        log.info(
            "%s vs raw — bin-rows raw-silent but imputed on: %d  "
            "(∩ pos panel=%d, ∩ neg panel=%d)",
            key,
            d["n_bin_rows_raw_silent_imputed_on"],
            d["n_intersection_pos_panel"],
            d["n_intersection_neg_panel"],
        )
    log.info(
        "Pairwise nnz-row intersections: raw∩cisTopic=%d raw∩PUscOpen=%d "
        "cisTopic∩PUscOpen=%d all_three=%d",
        n_raw_ct, n_raw_pu, n_ct_pu, n_all_three,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
