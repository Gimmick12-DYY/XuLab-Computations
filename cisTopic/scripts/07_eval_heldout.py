#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 07_eval_heldout.py
#
# Evaluate the cisTopic predictive distribution P(r|c) = phi @ theta on a set
# of (region, cell) pairs against baselines (random, region marginal, cell
# marginal, marginal-independence). Two modes:
#
#   * default ("reconstruction"):
#         Sample positives from observed nonzero entries in the filtered
#         binary matrix (CistopicObject.fragment_matrix) and an equal number
#         of negatives from zero entries. Score all four baselines + cisTopic
#         on those pairs and report AUROC / AUPRC. This is a goodness-of-fit
#         test on the training data, NOT a strict generalization test -- the
#         LDA model was trained on these entries.
#
#   * strict held-out (two steps):
#         1) Run this script with `--prepare-holdout`. It samples the same
#            balanced pos/neg split, zeros out the positives in the training
#            matrix, and writes a new `mm_holdout/` directory (matrix.mtx.gz
#            + regions/barcodes.tsv.gz) plus `eval/holdout_split.npz`.
#         2) Re-run steps 02-05 against a config whose `paths.mm` points at
#            that `mm_holdout/` folder (the new `obj/`, `models/`, `select/`,
#            `impute/` go under a separate work_dir so you don't overwrite
#            the originals). Then:
#               python 07_eval_heldout.py \
#                   --config <cfg> \
#                   --holdout-split  <orig_work>/eval/holdout_split.npz \
#                   --impute-dir     <holdout_work>/impute
#            The script will load the saved positives/negatives and score
#            them with the held-out model's theta/phi. This is a proper
#            generalization benchmark.
# -----------------------------------------------------------------------------
from __future__ import annotations

import gzip
import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from _cfg import base_parser, load_config, resolve_paths  # noqa: E402

log = logging.getLogger("07_eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# --- helpers -----------------------------------------------------------------
def _get_matrix(cistopic_obj) -> sp.csr_matrix:
    """Return the filtered regions x cells binary matrix used to train LDA."""
    for attr in ("fragment_matrix", "binary_matrix"):
        mat = getattr(cistopic_obj, attr, None)
        if mat is not None:
            return sp.csr_matrix(mat)
    raise AttributeError(
        "CistopicObject has neither .fragment_matrix nor .binary_matrix."
    )


def _sample_pairs(
    mat: sp.csr_matrix, n_samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample `n_samples` positives (observed nonzeros) and `n_samples`
    negatives (sampled zero entries) from `mat`."""
    R, C = mat.shape
    coo = mat.tocoo()
    if coo.nnz < n_samples:
        raise ValueError(
            f"Only {coo.nnz} nonzero entries; asked for {n_samples} positives."
        )

    pos_ix = rng.choice(coo.nnz, size=n_samples, replace=False)
    pos_rows = coo.row[pos_ix].astype(np.int64)
    pos_cols = coo.col[pos_ix].astype(np.int64)

    # The binary matrix is very sparse (>99% zeros) so rejection sampling is
    # effectively one-shot. Use a 64-bit linear index + set lookup for speed.
    nz_lin = coo.row.astype(np.int64) * C + coo.col.astype(np.int64)
    nz_set = set(nz_lin.tolist())
    seen: set[int] = set()

    neg_rows = np.empty(n_samples, dtype=np.int64)
    neg_cols = np.empty(n_samples, dtype=np.int64)
    filled = 0
    while filled < n_samples:
        need = n_samples - filled
        batch = int(need * 1.2) + 16
        r = rng.integers(0, R, size=batch, dtype=np.int64)
        c = rng.integers(0, C, size=batch, dtype=np.int64)
        lin = r * C + c
        for ri, ci, li in zip(r, c, lin):
            if filled >= n_samples:
                break
            li_i = int(li)
            if li_i in nz_set or li_i in seen:
                continue
            seen.add(li_i)
            neg_rows[filled] = ri
            neg_cols[filled] = ci
            filled += 1
    return pos_rows, pos_cols, neg_rows, neg_cols


def _cistopic_scores(
    phi: np.ndarray, theta: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> np.ndarray:
    """Score pairs with P(r|c) = phi[r] @ theta[:, c], vectorised."""
    # phi: (R, K), theta: (K, C). Use einsum to avoid building a (N, N) matrix.
    return np.einsum("ik,ki->i", phi[rows], theta[:, cols]).astype(np.float64)


def _baseline_scores(
    mat: sp.csr_matrix,
    rows: np.ndarray,
    cols: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Four cheap non-topic baselines evaluated on the same pair set."""
    R, C = mat.shape
    region_density = np.asarray(mat.sum(axis=1)).ravel() / float(C)  # p(r=1)
    cell_density   = np.asarray(mat.sum(axis=0)).ravel() / float(R)  # p(c=1)
    return {
        "random":       rng.random(len(rows)),
        "region_freq":  region_density[rows],
        "cell_freq":    cell_density[cols],
        "independence": region_density[rows] * cell_density[cols],
    }


def _metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
    }


def _plot(results: dict[str, dict[str, float]], out_png: Path, title: str) -> None:
    names = list(results.keys())
    auroc = [results[n]["auroc"] for n in names]
    auprc = [results[n]["auprc"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, vals, ylab in ((axes[0], auroc, "AUROC"), (axes[1], auprc, "AUPRC")):
        xs = np.arange(len(names))
        bars = ax.bar(xs, vals, color=["#999"] * (len(names) - 1) + ["#c33"])
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel(ylab)
        ax.set_ylim(0.0, 1.0)
        ax.axhline(0.5 if ylab == "AUROC" else 0.5, color="k", lw=0.5, ls=":")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _write_mm_gz(mat: sp.csr_matrix, path: Path) -> None:
    """Write a CSR matrix to Matrix Market (gzip'd)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix("")  # strip .gz
    sio.mmwrite(str(tmp), mat.tocoo(), field="integer", symmetry="general")
    # sio.mmwrite adds .mtx; gzip it
    with open(str(tmp) + ".mtx", "rb") as fin, gzip.open(str(path), "wb") as fout:
        fout.writelines(fin)
    Path(str(tmp) + ".mtx").unlink(missing_ok=True)


def _write_lines_gz(lines: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(path), "wt") as fh:
        fh.write("\n".join(lines) + "\n")


# --- main --------------------------------------------------------------------
def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--n-samples", type=int, default=100_000,
                    help="Number of positives (= number of negatives) to sample.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--impute-dir", type=str, default=None,
                    help="Override directory containing theta/phi .npy "
                         "(default: paths.impute from config).")
    ap.add_argument("--out-name", type=str, default="heldout_eval",
                    help="Stem for output files under <work_dir>/eval/.")
    ap.add_argument("--prepare-holdout", action="store_true",
                    help="Produce a masked mm_holdout/ directory and save a "
                         "reusable pos/neg split for a strict generalization run.")
    ap.add_argument("--holdout-split", type=str, default=None,
                    help="Path to a previously saved holdout_split.npz; if set, "
                         "skip sampling and score the saved pairs instead.")
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)

    obj_path = Path(cfg["paths"]["obj"]) / "cistopic_obj.pkl"
    if not obj_path.exists():
        log.error("CistopicObject not found: %s", obj_path)
        return 1
    with open(obj_path, "rb") as fh:
        cistopic_obj = pickle.load(fh)

    mat = _get_matrix(cistopic_obj)
    regions = list(cistopic_obj.region_names)
    cells   = list(cistopic_obj.cell_names)
    R, C = mat.shape
    nnz = mat.nnz
    density = nnz / float(R * C)
    log.info("Binary matrix: %d regions x %d cells, nnz=%d (density=%.3g)",
             R, C, nnz, density)

    rng = np.random.default_rng(args.seed)
    eval_dir = Path(cfg["paths"]["work_dir"]) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # -- decide pair set ------------------------------------------------------
    if args.holdout_split:
        split = np.load(args.holdout_split)
        pos_rows, pos_cols = split["pos_rows"], split["pos_cols"]
        neg_rows, neg_cols = split["neg_rows"], split["neg_cols"]
        log.info("Loaded holdout split from %s (n_pos=%d)",
                 args.holdout_split, len(pos_rows))
    else:
        log.info("Sampling %d positives and %d negatives (seed=%d)",
                 args.n_samples, args.n_samples, args.seed)
        pos_rows, pos_cols, neg_rows, neg_cols = _sample_pairs(
            mat, args.n_samples, rng
        )

    # -- prepare-holdout short-circuit ---------------------------------------
    if args.prepare_holdout:
        split_path = eval_dir / "holdout_split.npz"
        np.savez_compressed(
            split_path,
            pos_rows=pos_rows.astype(np.int64),
            pos_cols=pos_cols.astype(np.int64),
            neg_rows=neg_rows.astype(np.int64),
            neg_cols=neg_cols.astype(np.int64),
            n_regions=np.int64(R),
            n_cells=np.int64(C),
            seed=np.int64(args.seed),
        )

        # Zero out the positives in a copy of the training matrix.
        masked = mat.tolil()
        masked[pos_rows, pos_cols] = 0
        masked = masked.tocsr()
        masked.eliminate_zeros()

        mm_dir = Path(cfg["paths"]["work_dir"]) / "mm_holdout"
        _write_mm_gz(masked, mm_dir / "matrix.mtx.gz")
        _write_lines_gz(regions, mm_dir / "regions.tsv.gz")
        _write_lines_gz(cells,   mm_dir / "barcodes.tsv.gz")

        log.info("Wrote masked training matrix -> %s (nnz %d -> %d)",
                 mm_dir / "matrix.mtx.gz", nnz, masked.nnz)
        log.info("Wrote holdout split         -> %s", split_path)
        log.info(
            "Next: point a fresh work_dir's `paths.mm` at %s, re-run 02-05, "
            "then evaluate with:\n"
            "    python 07_eval_heldout.py --config <cfg> "
            "--holdout-split %s --impute-dir <holdout_work>/impute",
            mm_dir, split_path,
        )
        return 0

    # -- load theta/phi -------------------------------------------------------
    impute_dir = Path(args.impute_dir) if args.impute_dir else Path(cfg["paths"]["impute"])
    theta_path = impute_dir / "cell_topic_theta.npy"
    phi_path   = impute_dir / "region_topic_phi.npy"
    if not theta_path.exists() or not phi_path.exists():
        log.error("theta/phi not found in %s (expected cell_topic_theta.npy + "
                  "region_topic_phi.npy). Run 05_impute.py first.", impute_dir)
        return 1
    theta = np.load(theta_path)   # K x C
    phi   = np.load(phi_path)     # R x K
    log.info("Loaded phi %s and theta %s from %s", phi.shape, theta.shape, impute_dir)
    if phi.shape[0] != R or theta.shape[1] != C:
        log.error("Shape mismatch: matrix %s vs phi %s / theta %s. Make sure "
                  "the CistopicObject and theta/phi come from the same run.",
                  (R, C), phi.shape, theta.shape)
        return 1

    # -- score ----------------------------------------------------------------
    y_true = np.concatenate([np.ones(len(pos_rows), dtype=np.int8),
                              np.zeros(len(neg_rows), dtype=np.int8)])
    rows = np.concatenate([pos_rows, neg_rows])
    cols = np.concatenate([pos_cols, neg_cols])

    scorers = {
        "cisTopic (phi@theta)": _cistopic_scores(phi, theta, rows, cols),
        **_baseline_scores(mat, rows, cols, rng),
    }

    # Run cisTopic last in the bar plot by keying it after baselines:
    ordered = ["random", "region_freq", "cell_freq", "independence", "cisTopic (phi@theta)"]
    results = {name: _metrics(y_true, scorers[name]) for name in ordered}

    for name, m in results.items():
        log.info("  %-22s  AUROC=%.4f  AUPRC=%.4f", name, m["auroc"], m["auprc"])

    # -- persist --------------------------------------------------------------
    title = (f"cisTopic held-out reconstruction  (n_pos={len(pos_rows)}, "
             f"K={theta.shape[0]}, seed={args.seed})")

    json_path = eval_dir / f"{args.out_name}.json"
    tsv_path  = eval_dir / f"{args.out_name}.tsv"
    png_path  = eval_dir / f"{args.out_name}.png"

    summary = {
        "n_pos": int(len(pos_rows)),
        "n_neg": int(len(neg_rows)),
        "n_regions": int(R),
        "n_cells": int(C),
        "matrix_density": float(density),
        "K": int(theta.shape[0]),
        "seed": int(args.seed),
        "impute_dir": str(impute_dir),
        "holdout_split": args.holdout_split,
        "results": results,
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    with open(tsv_path, "w") as fh:
        fh.write("scorer\tauroc\tauprc\n")
        for name, m in results.items():
            fh.write(f"{name}\t{m['auroc']:.6f}\t{m['auprc']:.6f}\n")

    _plot(results, png_path, title)

    log.info("Wrote %s / %s / %s", json_path.name, tsv_path.name, png_path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
