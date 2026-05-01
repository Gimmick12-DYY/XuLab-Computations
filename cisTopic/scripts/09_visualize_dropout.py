#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 09_visualize_dropout.py
#
# Replicate the cisTopic Supplementary Figure 15-style "drop-out" visualization
# (left + middle panels) using *our* data:
#
#   left  panel : original observed binary matrix          (regions x cells)
#   right panel : cisTopic-imputed P(r|c)                   (regions x cells)
#
# Both panels share the same row order (regions) and column order (cells), so
# every column lines up cell-by-cell across panels.
#
# It also saves quantitative numbers requested by the lab:
#
#   * shape, nnz and density of the *original* matrix.
#   * shape and density of the *imputed* matrix at multiple thresholds, plus a
#     matched-threshold density that matches the original nnz exactly. Density
#     is reported in the *original* (lifted) universe so the two numbers are
#     directly comparable.
#   * per-cell total counts and per-cell number of accessible regions for both
#     matrices, written to a per-cell TSV.
#
# The script is read-only with respect to the rest of the pipeline -- it only
# reads <paths.mm>/matrix.mtx.gz and <paths.impute>/imputed_Prc_hdf5_all.h5
# (plus theta/phi .npy for ordering).
# -----------------------------------------------------------------------------
from __future__ import annotations

import gzip
import json
import logging
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.io as sio  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

from _cfg import base_parser, load_config, resolve_paths  # noqa: E402

log = logging.getLogger("09_viz")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ----------------------------- helpers ---------------------------------------
def _read_lines_gz(path: Path) -> list[str]:
    with gzip.open(path, "rt") as fh:
        return [ln.rstrip("\n") for ln in fh]


def _row_variance_chunked(Prc: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """Per-row variance with controlled peak memory."""
    R = Prc.shape[0]
    out = np.empty(R, dtype=np.float32)
    for s in range(0, R, chunk):
        e = min(s + chunk, R)
        out[s:e] = Prc[s:e].var(axis=1)
    return out


def _load_cell_labels(path: Path, cell_names: list[str]) -> np.ndarray:
    """Load barcode -> label TSV/CSV. First two columns must be (barcode, label)."""
    df = pd.read_csv(path, sep=None, engine="python")
    if df.shape[1] < 2:
        raise ValueError(f"{path} must have at least 2 columns (barcode, label).")
    bc_col, lab_col = df.columns[:2]
    mp = dict(zip(df[bc_col].astype(str), df[lab_col].astype(str)))
    return np.array([mp.get(b, "NA") for b in cell_names])


def _topic_argmax_labels(theta: np.ndarray) -> np.ndarray:
    return np.array([f"Topic{i+1}" for i in theta.argmax(axis=0)])


def _label_palette(labels: np.ndarray):
    uniq = sorted(np.unique(labels))
    base = plt.get_cmap("tab20", max(20, len(uniq)))
    color_map = {l: base(i % base.N) for i, l in enumerate(uniq)}
    rgba = np.array([color_map[l] for l in labels])
    return rgba, color_map


# ----------------------------- main ------------------------------------------
def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--imputed-h5", type=str, default=None,
                    help="Path to imputed HDF5 (default: <impute>/imputed_Prc_hdf5_all.h5).")
    ap.add_argument("--mm-dir", type=str, default=None,
                    help="Directory with original matrix.mtx.gz / regions.tsv.gz / barcodes.tsv.gz.")
    ap.add_argument("--n-regions", type=int, default=10000,
                    help="Number of top-variable regions to display in the heatmap.")
    ap.add_argument("--max-cells", type=int, default=0,
                    help="Subsample to N cells for plotting (0 = all).")
    ap.add_argument("--cell-labels", type=str, default=None,
                    help="Optional TSV mapping barcode -> cell-type label "
                         "(first two columns). Defaults to per-cell topic argmax.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-name", type=str, default="dropout_compare")
    ap.add_argument("--imputed-quantile-cap", type=float, default=0.99,
                    help="Quantile clip for the imputed colour scale (avoids a few "
                         "high-magnitude entries flattening the colour range).")
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    rng = np.random.default_rng(args.seed)

    # -------- load original ---------------------------------------------------
    mm_dir = Path(args.mm_dir) if args.mm_dir else Path(cfg["paths"]["mm"])
    mtx_path = mm_dir / "matrix.mtx.gz"
    reg_path = mm_dir / "regions.tsv.gz"
    bar_path = mm_dir / "barcodes.tsv.gz"
    for p in (mtx_path, reg_path, bar_path):
        if not p.exists():
            log.error("Missing original input: %s", p)
            return 1

    log.info("Reading original matrix: %s", mtx_path)
    orig = sio.mmread(mtx_path).tocsr()
    orig_regs = _read_lines_gz(reg_path)
    orig_bars = _read_lines_gz(bar_path)
    R_o, C_o = orig.shape
    orig_nnz = int(orig.nnz)
    orig_density = orig_nnz / float(R_o * C_o)
    orig_total_counts = float(orig.sum())
    orig_per_cell_count = np.asarray(orig.sum(axis=0)).ravel()  # total fragments per cell
    orig_per_cell_nnz = np.diff(orig.tocsc().indptr)             # nnz per cell
    log.info("Original: %d regions x %d cells | nnz=%d | density=%.3g | total_counts=%.0f",
             R_o, C_o, orig_nnz, orig_density, orig_total_counts)

    # -------- load imputed ----------------------------------------------------
    impute_dir = Path(cfg["paths"]["impute"])
    h5_path = Path(args.imputed_h5) if args.imputed_h5 else impute_dir / "imputed_Prc_hdf5_all.h5"
    if not h5_path.exists():
        log.error("Imputed HDF5 not found: %s (run 05_impute.py with hdf5_all/variable).", h5_path)
        return 1

    log.info("Reading imputed HDF5: %s", h5_path)
    with h5py.File(h5_path, "r") as h5:
        imp_regs = [s.decode() if isinstance(s, bytes) else str(s) for s in h5["regions"][:]]
        imp_bars = [s.decode() if isinstance(s, bytes) else str(s) for s in h5["barcodes"][:]]
        Prc = h5["Prc"][...].astype(np.float32, copy=False)
        scale_factor = float(h5.attrs.get("scale_factor", 1_000_000))
    R_f, C_f = Prc.shape
    log.info("Imputed (filtered): %d regions x %d cells | scale_factor=%g | %.1f GiB",
             R_f, C_f, scale_factor, Prc.nbytes / 1024**3)

    # -------- index alignment -------------------------------------------------
    reg_to_orig = {r: i for i, r in enumerate(orig_regs)}
    bar_to_orig = {b: i for i, b in enumerate(orig_bars)}
    try:
        row_idx_o = np.array([reg_to_orig[r] for r in imp_regs], dtype=np.int64)
        col_idx_o = np.array([bar_to_orig[b] for b in imp_bars], dtype=np.int64)
    except KeyError as e:
        log.error("Imputed name not found in original: %s", e)
        return 1
    log.info("Imputed regions cover %d / %d (%.2f%%) | cells cover %d / %d (%.2f%%)",
             R_f, R_o, 100.0 * R_f / R_o, C_f, C_o, 100.0 * C_f / C_o)

    # -------- imputed density (in lifted 3M x 10K universe) ------------------
    n_total_lifted = float(R_o * C_o)
    thresholds = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    imp_density_at: dict[str, dict] = {}
    log.info("Computing imputed density at %d thresholds...", len(thresholds))
    for t in thresholds:
        nnz_t = int((Prc > t).sum())
        imp_density_at[f"thr_{t:.0e}"] = {
            "threshold": t,
            "nnz_lifted": nnz_t,
            "density_lifted": nnz_t / n_total_lifted,
            "density_filtered": nnz_t / float(R_f * C_f),
        }

    # Matched threshold: pick t* so that nnz(Prc > t*) ≈ orig_nnz, in the lifted universe.
    log.info("Estimating matched threshold to equal original nnz=%d...", orig_nnz)
    sample_size = min(10_000_000, Prc.size)
    flat_sample = Prc.ravel()[rng.choice(Prc.size, size=sample_size, replace=False)]
    target_q = max(0.0, 1.0 - (orig_nnz / float(R_f * C_f)))
    matched_thr = float(np.quantile(flat_sample, target_q))
    matched_nnz = int((Prc > matched_thr).sum())
    log.info("Matched-threshold ≈ %.6g  ->  nnz=%d  (target=%d)", matched_thr, matched_nnz, orig_nnz)

    # -------- per-cell stats for imputed -------------------------------------
    imp_per_cell_sum = Prc.sum(axis=0).astype(np.float64)               # ≈ scale_factor
    imp_per_cell_nnz0 = (Prc > 0).sum(axis=0).astype(np.int64)
    imp_per_cell_nnz_matched = (Prc > matched_thr).sum(axis=0).astype(np.int64)

    imp_pc_sum_by_bar = dict(zip(imp_bars, imp_per_cell_sum.tolist()))
    imp_pc_nnz0_by_bar = dict(zip(imp_bars, imp_per_cell_nnz0.tolist()))
    imp_pc_nnzM_by_bar = dict(zip(imp_bars, imp_per_cell_nnz_matched.tolist()))

    # -------- region selection: top-K most variable in imputed ---------------
    K = min(args.n_regions, R_f)
    log.info("Selecting top-%d variable regions in imputed Prc...", K)
    rvar = _row_variance_chunked(Prc, chunk=20000)
    top_idx = np.argpartition(rvar, -K)[-K:]
    top_idx = top_idx[np.argsort(-rvar[top_idx])]

    # -------- cell labels -----------------------------------------------------
    if args.cell_labels:
        log.info("Loading cell labels from %s", args.cell_labels)
        labels = _load_cell_labels(Path(args.cell_labels), imp_bars)
        label_source = f"file:{args.cell_labels}"
    else:
        log.info("Using topic-argmax for cell labels.")
        theta_path = impute_dir / "cell_topic_theta.npy"
        if not theta_path.exists():
            log.error("Missing theta for label fallback: %s", theta_path)
            return 1
        theta_full = np.load(theta_path)
        labels = _topic_argmax_labels(theta_full)
        label_source = "topic_argmax"
    label_rgba, label_color_map = _label_palette(labels)

    # -------- ordering: by topic argmax for both axes ------------------------
    log.info("Computing topic-driven ordering for regions and cells...")
    phi_path = impute_dir / "region_topic_phi.npy"
    theta_path = impute_dir / "cell_topic_theta.npy"
    if not (phi_path.exists() and theta_path.exists()):
        log.error("Missing phi/theta in %s", impute_dir)
        return 1
    phi = np.load(phi_path)        # R_f x K
    theta = np.load(theta_path)    # K x C_f

    # Cells subsample (for plotting only)
    if args.max_cells and args.max_cells < C_f:
        cell_sel = rng.choice(C_f, size=args.max_cells, replace=False)
    else:
        cell_sel = np.arange(C_f)

    region_dom = phi[top_idx].argmax(axis=1)
    region_score = phi[top_idx, region_dom]
    region_order = np.lexsort((-region_score, region_dom))
    cell_dom = theta[:, cell_sel].argmax(axis=0)
    cell_score = theta[cell_dom, cell_sel]
    cell_order = np.lexsort((-cell_score, cell_dom))

    top_ordered = top_idx[region_order]
    cell_ordered = cell_sel[cell_order]
    labels_ordered = labels[cell_ordered]
    rgba_ordered = label_rgba[cell_ordered]

    # -------- subset matrices for plotting -----------------------------------
    log.info("Building subset matrices (regions=%d, cells=%d)...", K, len(cell_ordered))
    orig_top_rows_in_orig = row_idx_o[top_ordered]
    orig_top_cols_in_orig = col_idx_o[cell_ordered]
    orig_sub_bin = (
        orig[orig_top_rows_in_orig][:, orig_top_cols_in_orig] > 0
    ).astype(np.int8).toarray()
    imp_sub = Prc[top_ordered][:, cell_ordered]

    # -------- plot ------------------------------------------------------------
    log.info("Rendering heatmap...")
    cap_n = min(2_000_000, imp_sub.size)
    cap_sample = imp_sub.ravel()[rng.choice(imp_sub.size, size=cap_n, replace=False)]
    vmax = float(np.quantile(cap_sample, args.imputed_quantile_cap))
    if vmax <= 0:
        vmax = float(imp_sub.max() or 1.0)

    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(
        2, 2,
        height_ratios=[0.04, 1.0],
        width_ratios=[1.0, 1.0],
        hspace=0.04, wspace=0.04,
    )
    ax_cb_l = fig.add_subplot(grid[0, 0])
    ax_cb_r = fig.add_subplot(grid[0, 1])
    ax_l = fig.add_subplot(grid[1, 0], sharex=ax_cb_l)
    ax_r = fig.add_subplot(grid[1, 1], sharex=ax_cb_r)

    cb_strip = rgba_ordered[np.newaxis, :, :3]
    ax_cb_l.imshow(cb_strip, aspect="auto"); ax_cb_l.set_xticks([]); ax_cb_l.set_yticks([])
    ax_cb_r.imshow(cb_strip, aspect="auto"); ax_cb_r.set_xticks([]); ax_cb_r.set_yticks([])
    ax_cb_l.set_title(
        f"Original observed (binarised)\n"
        f"{R_o:,} \u00d7 {C_o:,}  |  nnz={orig_nnz:,}  |  density={orig_density:.3g}"
    )
    ax_cb_r.set_title(
        f"cisTopic imputed P(r|c)\n"
        f"matched-threshold nnz={matched_nnz:,}  (t \u2248 {matched_thr:.3g})"
    )

    cmap_bin = ListedColormap(["white", "#c43"])
    ax_l.imshow(orig_sub_bin, aspect="auto", interpolation="nearest", cmap=cmap_bin)
    ax_l.set_xticks([]); ax_l.set_yticks([])
    ax_l.set_xlabel(f"Cells (n={orig_sub_bin.shape[1]:,}, ordered by topic)")
    ax_l.set_ylabel(f"Top {K:,} variable regions (ordered by topic)")

    cmap_cont = LinearSegmentedColormap.from_list(
        "white_red", ["white", "#fdd49e", "#e34a33", "#7f0000"]
    )
    ax_r.imshow(imp_sub, aspect="auto", interpolation="nearest",
                cmap=cmap_cont, vmin=0, vmax=vmax)
    ax_r.set_xticks([]); ax_r.set_yticks([])
    ax_r.set_xlabel(f"Cells (n={imp_sub.shape[1]:,}, ordered by topic)")
    ax_r.set_ylabel("")

    handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=c, label=l)
               for l, c in label_color_map.items()]
    if len(handles) <= 25:
        fig.legend(handles=handles, loc="lower center",
                   ncol=min(8, len(handles)),
                   bbox_to_anchor=(0.5, -0.02), fontsize=7,
                   title=f"cell label ({label_source})", title_fontsize=8)

    eval_dir = Path(cfg["paths"]["work_dir"]) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    png = eval_dir / f"{args.out_name}_heatmap.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", png)

    # -------- summary JSON ----------------------------------------------------
    summary = {
        "config": str(args.config),
        "mm_dir": str(mm_dir),
        "imputed_h5": str(h5_path),
        "label_source": label_source,
        "n_regions_displayed": int(K),
        "n_cells_displayed": int(len(cell_ordered)),
        "original": {
            "shape": [int(R_o), int(C_o)],
            "nnz": int(orig_nnz),
            "density": float(orig_density),
            "total_counts": float(orig_total_counts),
            "per_cell_total_counts": {
                "mean": float(orig_per_cell_count.mean()),
                "median": float(np.median(orig_per_cell_count)),
                "min": float(orig_per_cell_count.min()),
                "max": float(orig_per_cell_count.max()),
            },
            "per_cell_nnz": {
                "mean": float(orig_per_cell_nnz.mean()),
                "median": float(np.median(orig_per_cell_nnz)),
                "min": int(orig_per_cell_nnz.min()),
                "max": int(orig_per_cell_nnz.max()),
            },
        },
        "imputed_filtered": {
            "shape": [int(R_f), int(C_f)],
            "scale_factor": scale_factor,
            "per_cell_sum": {
                "mean": float(imp_per_cell_sum.mean()),
                "median": float(np.median(imp_per_cell_sum)),
                "min": float(imp_per_cell_sum.min()),
                "max": float(imp_per_cell_sum.max()),
            },
            "per_cell_nnz_thr0": {
                "mean": float(imp_per_cell_nnz0.mean()),
                "median": float(np.median(imp_per_cell_nnz0)),
                "min": int(imp_per_cell_nnz0.min()),
                "max": int(imp_per_cell_nnz0.max()),
            },
        },
        "imputed_lifted": {
            "shape": [int(R_o), int(C_o)],
            "thresholds": imp_density_at,
            "matched_threshold": {
                "value": float(matched_thr),
                "target_nnz": int(orig_nnz),
                "achieved_nnz": int(matched_nnz),
                "density_lifted": matched_nnz / n_total_lifted,
            },
        },
    }
    summary_path = eval_dir / f"{args.out_name}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    log.info("Wrote %s", summary_path)

    # -------- per-cell TSV ----------------------------------------------------
    per_cell_path = eval_dir / f"{args.out_name}_per_cell.tsv"
    imp_label_by_bar = dict(zip(imp_bars, labels))
    with open(per_cell_path, "w") as fh:
        fh.write("\t".join([
            "barcode",
            "orig_total_counts",
            "orig_nnz_regions",
            "imputed_sum",
            "imputed_nnz_thr0",
            "imputed_nnz_matched_thr",
            "label",
        ]) + "\n")
        for i, b in enumerate(orig_bars):
            o_count = float(orig_per_cell_count[i])
            o_nnz = int(orig_per_cell_nnz[i])
            i_sum = imp_pc_sum_by_bar.get(b, float("nan"))
            i_nnz0 = imp_pc_nnz0_by_bar.get(b, -1)
            i_nnzM = imp_pc_nnzM_by_bar.get(b, -1)
            lab = imp_label_by_bar.get(b, "FILTERED_OUT")
            fh.write(f"{b}\t{o_count:.0f}\t{o_nnz}\t{i_sum:.6g}\t{i_nnz0}\t{i_nnzM}\t{lab}\n")
    log.info("Wrote %s", per_cell_path)

    # -------- terse human-readable header summary ----------------------------
    log.info("------- summary -------")
    log.info("Original   : %d x %d | nnz=%d | density=%.3g",
             R_o, C_o, orig_nnz, orig_density)
    log.info("Imputed    : filtered %d x %d, lifted -> %d x %d (matches original)",
             R_f, C_f, R_o, C_o)
    log.info("Imputed @ matched-thr (%.3g) : nnz=%d | density(lifted)=%.3g",
             matched_thr, matched_nnz, matched_nnz / n_total_lifted)
    log.info("Per-cell counts (orig)       : mean=%.1f | median=%.1f",
             orig_per_cell_count.mean(), np.median(orig_per_cell_count))
    log.info("Per-cell sum  (imputed)      : mean=%.3g | median=%.3g",
             imp_per_cell_sum.mean(), np.median(imp_per_cell_sum))
    log.info("Per-cell nnz @ matched-thr   : mean=%.1f | median=%.1f",
             imp_per_cell_nnz_matched.mean(), np.median(imp_per_cell_nnz_matched))
    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
