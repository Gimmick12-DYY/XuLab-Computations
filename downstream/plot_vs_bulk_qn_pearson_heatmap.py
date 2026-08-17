#!/usr/bin/env python3
"""Raw/imputed × bulk after quantile normalization + Pearson.

Spearman is invariant to per-track monotone transforms (including QN), so we
QN each modality (bins × TFs), then use Pearson.

QN modes (--qn):
  cols     classic column QN (equalize TF marginals)
  rows     row QN (equalize per-bin profiles across TFs)
  both     columns then rows (default; addresses uniform off-diagonal wash)

Feature space (--features): all | ccre | union_peaks (default: union_peaks).

Outputs under downstream/plots/tf_bulk_pearson/:
  pearson_{features}_raw_vs_bulk_qn{,_cols,_rows}{.tsv,.png,_annot.png}
  pearson_{features}_imputed_vs_bulk_qn{,_cols,_rows}{.tsv,.png,_annot.png}

Usage:
  python3 downstream/plot_vs_bulk_qn_pearson_heatmap.py
  python3 downstream/plot_vs_bulk_qn_pearson_heatmap.py --features all --qn cols
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

PANEL = [
    ("CTCF", "ctcf"),
    ("MAZ", "maz"),
    ("ZIC2", "zic2"),
    ("ZBTB7A", "zbtb7a"),
    ("ZNF777", "znf777"),
    ("NFYA", "nfya"),
    ("ZNF282", "znf282"),
]

RSCRIPT = "/nas/longleaf/home/dyy12/.conda/envs/cistopic/bin/Rscript"


def load_pseudobulk(work: Path, kind: str) -> np.ndarray:
    if kind == "raw":
        mat = sio.mmread(str(work / "mm" / "matrix.mtx.gz")).tocsr()
        return np.asarray(mat.sum(axis=1)).ravel().astype(np.float64)
    mat = sp.load_npz(work / "impute" / "matrix_csr.npz").tocsr()
    return np.asarray(mat.mean(axis=1)).ravel().astype(np.float64)


def quantile_normalize(X: np.ndarray) -> np.ndarray:
    """Classic QN on columns of X (n_bins × n_samples)."""
    n, k = X.shape
    order = np.argsort(X, axis=0, kind="mergesort")
    sorted_X = np.take_along_axis(X, order, axis=0)
    mean_sorted = sorted_X.mean(axis=1)
    out = np.empty_like(X, dtype=np.float64)
    ranks = np.empty_like(order)
    for j in range(k):
        ranks[order[:, j], j] = np.arange(n)
    for j in range(k):
        out[:, j] = mean_sorted[ranks[:, j]]
    return out


def apply_qn(X: np.ndarray, mode: str) -> np.ndarray:
    """QN columns, rows, or columns-then-rows of bins×TFs matrix."""
    if mode == "cols":
        return quantile_normalize(X)
    if mode == "rows":
        return quantile_normalize(X.T).T
    if mode == "both":
        return quantile_normalize(quantile_normalize(X).T).T
    raise SystemExit(f"unknown --qn mode {mode!r}")


def pearson_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson corr of columns of A (rows) vs columns of B (cols)."""
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    As = np.sqrt((Ac * Ac).sum(axis=0))
    Bs = np.sqrt((Bc * Bc).sum(axis=0))
    As = np.where(As == 0, np.nan, As)
    Bs = np.where(Bs == 0, np.nan, Bs)
    return (Ac.T @ Bc) / np.outer(As, Bs)


def write_tsv(path: Path, M: np.ndarray, labels: list[str], corner: str) -> None:
    with path.open("w") as f:
        f.write(corner + "\t" + "\t".join(labels) + "\n")
        for i, lab in enumerate(labels):
            f.write(
                lab
                + "\t"
                + "\t".join(f"{M[i, j]:.4f}" for j in range(len(labels)))
                + "\n"
            )


def plot_complexheatmap(
    tsv: Path,
    png: Path,
    title: str,
    *,
    row_title: str,
    col_title: str,
    annot: bool = False,
    legend: str = "Pearson",
) -> None:
    if annot:
        cell_fun = textwrap.dedent(
            """\
            cell_fun = function(j, i, x, y, width, height, fill) {
              grid.text(sprintf("%.2f", mat[i, j]), x, y, gp = gpar(fontsize = 8))
            }
            """
        )
    else:
        cell_fun = "cell_fun = NULL\n"
    r_code = textwrap.dedent(
        f"""\
        suppressPackageStartupMessages({{
          library(ComplexHeatmap); library(circlize); library(grid)
        }})
        tab <- read.delim("{tsv}", check.names = FALSE, row.names = 1)
        mat <- as.matrix(tab)
        col_fun <- colorRamp2(c(-1, 0, 1), c("#313695", "#FFFFBF", "#A50026"))
        {cell_fun}
        png("{png}", width = 1700, height = 1550, res = 220)
        ht <- Heatmap(
          mat, name = "{legend}", col = col_fun,
          cluster_rows = FALSE, cluster_columns = FALSE,
          rect_gp = gpar(col = "white", lwd = 0.5),
          cell_fun = cell_fun,
          column_title = "{title}", column_title_gp = gpar(fontsize = 11),
          row_title = "{row_title}", row_title_gp = gpar(fontsize = 11),
          row_names_side = "left", column_names_rot = 45,
          width = ncol(mat) * unit(7.5, "mm"),
          height = nrow(mat) * unit(7.5, "mm"),
          heatmap_legend_param = list(title = "{legend}", at = c(-1, 0, 1))
        )
        draw(
          ht, column_title = "{col_title}", column_title_side = "bottom",
          column_title_gp = gpar(fontsize = 11),
          padding = unit(c(4, 4, 4, 10), "mm")
        )
        dev.off()
        cat("Wrote", "{png}", "\\n")
        """
    )
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as tmp:
        tmp.write(r_code)
        rpath = tmp.name
    try:
        subprocess.run([RSCRIPT, rpath], check=True)
    finally:
        Path(rpath).unlink(missing_ok=True)


def print_matrix(name: str, R: np.ndarray, labels: list[str]) -> None:
    print(f"{name}:")
    print("       " + "".join(f"{l:>8s}" for l in labels))
    for i, lab in enumerate(labels):
        print(f"{lab:6s}" + "".join(f"{R[i, j]:8.3f}" for j in range(len(labels))))
    print("diag:", {labels[i]: round(float(R[i, i]), 3) for i in range(len(labels))})
    n_best = 0
    for i, rl in enumerate(labels):
        jmax = int(np.nanargmax(R[i, :]))
        ok = labels[jmax] == rl
        n_best += int(ok)
        print(
            f"  row {rl}: best={labels[jmax]} r={R[i, jmax]:.3f} "
            f"diag={R[i, i]:.3f}{' OK' if ok else ''}"
        )
    print(f"  diagonal is row-max in {n_best}/{len(labels)} rows")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-root", type=Path, default=REPO / "unified" / "work")
    ap.add_argument(
        "--out-dir", type=Path, default=ROOT / "plots" / "tf_bulk_pearson"
    )
    ap.add_argument(
        "--features",
        choices=["ccre", "union_peaks", "all"],
        default="union_peaks",
        help="Shared genomic feature space (default: union_peaks).",
    )
    ap.add_argument(
        "--qn",
        choices=["cols", "rows", "both"],
        default="both",
        help="QN axes: cols, rows, or columns-then-rows (default: both).",
    )
    ap.add_argument("--cache-dir", type=Path, default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or (args.out_dir / "_cache")
    cache.mkdir(parents=True, exist_ok=True)

    labels = [p[0] for p in PANEL]
    n_tf = len(PANEL)

    if args.features == "all":
        idx = None
        print("[features] all bins", flush=True)
    else:
        idx_path = cache / f"bin_idx_{args.features}.npy"
        if not idx_path.is_file():
            raise SystemExit(
                f"missing {idx_path}; run plot_vs_bulk_spearman_heatmap.py "
                f"--features {args.features} first to build the index"
            )
        idx = np.load(idx_path)
        print(f"[features] {args.features}: n={idx.size}", flush=True)

    raw_pb = None
    imp_pb = None
    bulk_pb = None
    n_use = None

    for j, (name, key) in enumerate(PANEL):
        print(f"[{name}] loading...", flush=True)
        work = args.work_root / key
        r = load_pseudobulk(work, "raw")
        u = load_pseudobulk(work, "imputed")
        b = np.load(cache / f"bulk_bedcov_{key}.npy")
        if r.size != u.size or r.size != b.size:
            raise SystemExit(f"{name}: length mismatch raw/imp/bulk")
        if idx is not None:
            r, u, b = r[idx], u[idx], b[idx]
        if n_use is None:
            n_use = r.size
            raw_pb = np.zeros((n_use, n_tf), dtype=np.float64)
            imp_pb = np.zeros((n_use, n_tf), dtype=np.float64)
            bulk_pb = np.zeros((n_use, n_tf), dtype=np.float64)
        raw_pb[:, j] = r
        imp_pb[:, j] = u
        bulk_pb[:, j] = b

    assert raw_pb is not None and imp_pb is not None and bulk_pb is not None
    qn_label = {"cols": "QN-cols", "rows": "QN-rows", "both": "QN-cols+rows"}[args.qn]
    print(
        f"[QN] {qn_label} on raw / imputed / bulk (shape={raw_pb.shape})...",
        flush=True,
    )
    raw_qn = apply_qn(raw_pb, args.qn)
    imp_qn = apply_qn(imp_pb, args.qn)
    bulk_qn = apply_qn(bulk_pb, args.qn)
    print(
        "  after QN bulk col means: "
        + ", ".join(f"{labels[j]}={bulk_qn[:, j].mean():.2f}" for j in range(n_tf)),
        flush=True,
    )

    # Tag: pearson_union_peaks_raw_vs_bulk_qn  (both) or ..._qn_cols / _qn_rows
    qn_suffix = "" if args.qn == "both" else f"_{args.qn}"
    pairs = [
        (
            "raw",
            raw_qn,
            "raw",
            "bulk",
            f"pearson_{args.features}_raw_vs_bulk_qn{qn_suffix}",
        ),
        (
            "imputed",
            imp_qn,
            "imputed",
            "bulk",
            f"pearson_{args.features}_imputed_vs_bulk_qn{qn_suffix}",
        ),
    ]
    for pretty, mat, row_lab, col_lab, tag in pairs:
        print(f"[corr] Pearson {pretty} × bulk ({qn_label})...", flush=True)
        R = pearson_matrix(mat, bulk_qn)
        tsv = args.out_dir / f"{tag}.tsv"
        write_tsv(tsv, R, labels, f"{row_lab}/{col_lab} {qn_label}")
        print(f"Wrote {tsv}")
        print_matrix(f"Pearson {pretty} × bulk ({qn_label})", R, labels)
        title = f"Pearson  {pretty} × bulk  ({args.features}, {qn_label})"
        plot_complexheatmap(
            tsv,
            args.out_dir / f"{tag}.png",
            title,
            row_title=row_lab,
            col_title=col_lab,
            annot=False,
        )
        plot_complexheatmap(
            tsv,
            args.out_dir / f"{tag}_annot.png",
            title,
            row_title=row_lab,
            col_title=col_lab,
            annot=True,
        )
        print(f"Wrote {args.out_dir / f'{tag}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
