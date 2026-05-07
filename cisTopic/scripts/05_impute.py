#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 05_impute.py
#
# Persist cisTopic's predictive distribution P(r | c) ~ phi @ theta in a
# compact, HDF5-free format that the universal downstream/compare_pos_neg.py
# consumes:
#
#   <work>/impute/
#     factors.npz       phi (n_modeled, k) + theta (k, n_cells) + per_cell_factor
#     regions.tsv       one chr:start-end per row, in the order matching phi rows
#     barcodes.tsv      one cell barcode per col, matching theta cols
#     meta.json         k, scale_factor, source paths, n_modeled, n_cells, ...
#
# Downstream tools reconstruct any rows they need on-demand via
#   phi[idx] @ theta * per_cell_factor
# without ever materialising the full dense (R x C) matrix.
#
# We deliberately do NOT pad to the full pre-filter region universe -- the
# cisTopic LDA filter is documented to be necessary for stability, and
# downstream code now looks bins up by name in regions.tsv rather than by
# global mm row index.
#
# We also still keep theta/phi as parquet + npy on disk for any analyses that
# expect the legacy filenames.
# -----------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger("05_impute")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--scale-factor", type=float, default=None)
    ap.add_argument("--no-rescale", action="store_true")
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    imp = cfg.setdefault("impute", {})
    if args.scale_factor is not None:
        imp["scale_factor"] = args.scale_factor

    obj_path = Path(cfg["paths"]["obj"]) / "cistopic_obj.pkl"
    if not obj_path.exists():
        log.error("CistopicObject not found: %s", obj_path)
        return 1
    with open(obj_path, "rb") as fh:
        cistopic_obj = pickle.load(fh)

    if cistopic_obj.selected_model is None:
        log.error("No model attached to the CistopicObject; run 04_select_model.py first.")
        return 1

    theta = np.asarray(cistopic_obj.selected_model.cell_topic, dtype=np.float32)   # K x C
    phi   = np.asarray(cistopic_obj.selected_model.topic_region, dtype=np.float32) # R x K
    cells   = list(cistopic_obj.cell_names)
    regions = list(cistopic_obj.region_names)
    K = theta.shape[0]
    log.info("theta: %s (K x C), phi: %s (R x K)", theta.shape, phi.shape)

    out_dir = Path(cfg["paths"]["impute"])
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(theta, columns=cells,
                 index=[f"Topic{i+1}" for i in range(K)]).to_parquet(
                     out_dir / "cell_topic_theta.parquet")
    pd.DataFrame(phi, index=regions,
                 columns=[f"Topic{i+1}" for i in range(K)]).to_parquet(
                     out_dir / "region_topic_phi.parquet")
    np.save(out_dir / "cell_topic_theta.npy", theta)
    np.save(out_dir / "region_topic_phi.npy", phi)
    log.info("Saved theta/phi to %s", out_dir)

    desired_scale = float(imp.get("scale_factor", 1_000_000))
    if args.no_rescale:
        log.info("Storing factors verbatim (--no-rescale).")
        per_cell_factor = float(1.0)
        rescaled = False
    else:
        col_sums = (phi.sum(axis=0) @ theta).astype(np.float64)
        sum_per_cell_mean = float(col_sums.mean()) if col_sums.size else 0.0
        if sum_per_cell_mean <= 0:
            log.warning("phi @ theta per-cell sum mean <= 0; storing factors verbatim.")
            per_cell_factor = float(1.0)
            rescaled = False
        else:
            per_cell_factor = float(desired_scale / sum_per_cell_mean)
            log.info("Rescaling factor=%.4g so per-cell sum mean -> %.0f.",
                     per_cell_factor, desired_scale)
            rescaled = True

    factors_path = out_dir / "factors.npz"
    regions_out  = out_dir / "regions.tsv"
    barcodes_out = out_dir / "barcodes.tsv"
    meta_out     = out_dir / "meta.json"

    log.info("Writing %s (W=phi=%s, H=theta=%s, per_cell_factor=%.4g)",
             factors_path, phi.shape, theta.shape, per_cell_factor)
    np.savez_compressed(
        factors_path,
        W=phi,
        H=theta,
        per_cell_factor=np.float32(per_cell_factor),
    )
    regions_out.write_text("\n".join(regions) + "\n")
    barcodes_out.write_text("\n".join(cells) + "\n")

    summary = {
        "pipeline":        "cisTopic",
        "format":          "factored_npz",
        "factors_path":    str(factors_path),
        "regions_path":    str(regions_out),
        "barcodes_path":   str(barcodes_out),
        "shape":           [int(phi.shape[0]), int(theta.shape[1])],
        "n_modeled":       int(phi.shape[0]),
        "n_cells":         int(theta.shape[1]),
        "rank_k":          int(K),
        "scale_factor":    desired_scale,
        "per_cell_factor": float(per_cell_factor),
        "rescaled":        rescaled,
        "source_obj":      str(obj_path),
    }
    meta_out.write_text(json.dumps(summary, indent=2) + "\n")
    log.info("Done: %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
