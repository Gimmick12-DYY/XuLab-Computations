#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 05_impute.py
#
# Produce the cisTopic predictive distribution P(r | c) = phi^T * theta,
# which is what the paper describes as "imputation of drop-outs". Depending on
# impute.mode in the config we either:
#   * save only theta (cell-topic) and phi (region-topic) as parquet + npy, or
#   * materialise the imputed matrix into an HDF5 file (float32). The HDF5 row
#     axis is aligned to the *full pre-filter* region list (mm/regions.tsv.gz),
#     so its shape matches the original input matrix (R_full x C). Regions that
#     were dropped by the 02_build filters - or by hdf5_variable's top-K
#     selection - are stored as zeros and compress to ~nothing under gzip,
#     which lets downstream tools (e.g. bedGraph/bigWig export) map row index
#     directly to genome interval without an extra alignment step.
# -----------------------------------------------------------------------------
from __future__ import annotations

import gzip
import logging
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger("05_impute")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--mode", type=str, default=None,
                    choices=["theta_phi_only", "hdf5_all", "hdf5_variable"])
    ap.add_argument("--n-top-variable", type=int, default=None)
    ap.add_argument("--scale-factor",   type=float, default=None)
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    imp = cfg.setdefault("impute", {})
    if args.mode:              imp["mode"]             = args.mode
    if args.n_top_variable:    imp["n_top_variable"]   = args.n_top_variable
    if args.scale_factor:      imp["scale_factor"]     = args.scale_factor

    obj_path = Path(cfg["paths"]["obj"]) / "cistopic_obj.pkl"
    if not obj_path.exists():
        log.error("CistopicObject not found: %s", obj_path)
        return 1
    with open(obj_path, "rb") as fh:
        cistopic_obj = pickle.load(fh)

    if cistopic_obj.selected_model is None:
        log.error("No model attached to the CistopicObject; run 04_select_model.py first.")
        return 1

    # cell_topic: topics x cells; region_topic: regions x topics (pycisTopic conventions)
    theta = np.asarray(cistopic_obj.selected_model.cell_topic)     # K x C
    phi   = np.asarray(cistopic_obj.selected_model.topic_region)   # R x K
    cells   = list(cistopic_obj.cell_names)
    regions = list(cistopic_obj.region_names)
    K = theta.shape[0]
    log.info("theta: %s (K x C), phi: %s (R x K)", theta.shape, phi.shape)

    out_dir = Path(cfg["paths"]["impute"])

    # Always persist theta and phi as parquet + npy (small, cheap, always useful).
    pd.DataFrame(theta, columns=cells,
                 index=[f"Topic{i+1}" for i in range(K)]).to_parquet(out_dir / "cell_topic_theta.parquet")
    pd.DataFrame(phi, index=regions,
                 columns=[f"Topic{i+1}" for i in range(K)]).to_parquet(out_dir / "region_topic_phi.parquet")
    np.save(out_dir / "cell_topic_theta.npy", theta.astype(np.float32))
    np.save(out_dir / "region_topic_phi.npy", phi.astype(np.float32))
    log.info("Saved theta/phi to %s", out_dir)

    mode = imp.get("mode", "theta_phi_only")
    if mode == "theta_phi_only":
        log.info("mode=theta_phi_only; not materialising P(r|c). Done.")
        return 0

    # -- Materialise P(r|c) via pycisTopic.impute_accessibility ---------------
    from pycisTopic.diff_features import impute_accessibility

    selected_regions = None
    if mode == "hdf5_variable":
        try:
            from pycisTopic.diff_features import find_highly_variable_features
        except ImportError:  # old pycisTopic layout
            from pycisTopic.utils import find_highly_variable_features  # type: ignore

        n_top = int(imp.get("n_top_variable", 200000))
        log.info("Computing top-%d variable regions via pycisTopic...", n_top)
        # find_highly_variable_features expects an imputed accessibility object in
        # recent pycisTopic; use phi-weighted variance as a cheap proxy here.
        region_var = (phi.var(axis=1))
        idx = np.argsort(region_var)[::-1][:n_top]
        selected_regions = [regions[i] for i in sorted(idx)]

    log.info("Calling impute_accessibility (mode=%s)...", mode)
    imputed = impute_accessibility(
        cistopic_obj,
        selected_cells=None,
        selected_regions=selected_regions,
        scale_factor=float(imp.get("scale_factor", 1_000_000)),
    )
    # `imputed` is a CistopicImputedFeatures-like object: .mtx (R x C), .cell_names, .feature_names
    mtx = np.asarray(imputed.mtx, dtype=np.float32)
    feat = list(imputed.feature_names)
    cnames = list(imputed.cell_names)

    # pycisTopic's behaviour around scale_factor varies between versions.
    # We always want the saved HDF5 to contain P(r|c) * scale_factor so that
    # sum over regions per cell == scale_factor (e.g. 1e6 = CPM) and downstream
    # comparisons can recover P(r|c) by dividing by scale_factor.
    desired_scale = float(imp.get("scale_factor", 1_000_000))
    sum_per_cell = float(np.asarray(mtx.sum(axis=0)).mean())
    log.info(
        "impute_accessibility returned sum-per-cell mean=%.4g; desired scale_factor=%.0f",
        sum_per_cell, desired_scale,
    )
    if sum_per_cell <= 0:
        log.warning("Imputed matrix sum-per-cell is non-positive; skipping rescale.")
    elif abs(sum_per_cell - desired_scale) / desired_scale > 0.01:
        factor = desired_scale / sum_per_cell
        log.warning(
            "Per-cell sum (%.4g) differs from configured scale_factor (%.0f); "
            "rescaling matrix by %.4g so HDF5 stores P(r|c) * scale_factor.",
            sum_per_cell, desired_scale, factor,
        )
        mtx = mtx * factor
    log.info("Imputed (modeled) matrix shape: %s (float32 => %.1f GiB)",
             mtx.shape, mtx.nbytes / 1024**3)

    # Pad back to the original pre-filter region shape so the HDF5 row axis
    # matches the input matrix 1:1. mm/regions.tsv.gz is written by
    # 01_export_rds_to_mm.R before any filtering.
    mm_regions_path = Path(cfg["paths"]["mm"]) / "regions.tsv.gz"
    if not mm_regions_path.exists():
        log.error("Cannot pad to full shape: %s missing.", mm_regions_path)
        return 1
    with gzip.open(mm_regions_path, "rt") as fh:
        full_regions = [ln.rstrip("\n") for ln in fh]
    R_full = len(full_regions)

    modeled_to_pos = {r: i for i, r in enumerate(feat)}
    full_to_modeled = np.fromiter(
        (modeled_to_pos.get(r, -1) for r in full_regions),
        dtype=np.int64, count=R_full,
    )
    n_present = int((full_to_modeled >= 0).sum())
    if n_present != len(feat):
        log.warning("Modeled regions not all matched in full list: %d/%d found.",
                    n_present, len(feat))
    log.info(
        "Padding to full region space: R_modeled=%d -> R_full=%d (%.2f%% rows present).",
        len(feat), R_full, 100.0 * n_present / max(R_full, 1),
    )

    C = mtx.shape[1]
    h5_path = out_dir / f"imputed_Prc_{mode}.h5"
    log.info("Writing HDF5 -> %s (shape %d x %d)", h5_path, R_full, C)
    with h5py.File(h5_path, "w") as h5:
        dset = h5.create_dataset(
            "Prc",
            shape=(R_full, C),
            dtype=np.float32,
            compression="gzip", compression_opts=4,
            chunks=(min(4096, R_full), min(512, C)),
            fillvalue=np.float32(0.0),
        )
        # Stream rows in chunks so we don't allocate a second R_full x C
        # dense copy on top of the modeled mtx already in memory.
        chunk_rows = 50_000
        for start in range(0, R_full, chunk_rows):
            end = min(start + chunk_rows, R_full)
            idx = full_to_modeled[start:end]
            present = idx >= 0
            if not present.any():
                continue  # rely on fillvalue=0; chunks compress away
            block = np.zeros((end - start, C), dtype=np.float32)
            block[present] = mtx[idx[present]]
            dset[start:end] = block
        h5.create_dataset("regions",  data=np.asarray(full_regions, dtype="S"))
        h5.create_dataset("barcodes", data=np.asarray(cnames,       dtype="S"))
        h5.attrs["scale_factor"]      = float(imp.get("scale_factor", 1_000_000))
        h5.attrs["mode"]              = mode
        h5.attrs["n_modeled_regions"] = int(len(feat))
        h5.attrs["n_full_regions"]    = int(R_full)

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
