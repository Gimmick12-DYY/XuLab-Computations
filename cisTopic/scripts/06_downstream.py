#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 06_downstream.py   (optional)
#
# Sanity-check downstream analyses you can run once imputation is done:
#   * UMAP of the topic-cell distribution (a cheap-but-useful QC plot)
#   * Per-topic region binarisation (Otsu / Yen / Li / AUCell)
# -----------------------------------------------------------------------------
from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _cfg import base_parser, load_config, resolve_paths  # noqa: E402

log = logging.getLogger("06_down")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--no-umap",     action="store_true")
    ap.add_argument("--no-binarize", action="store_true")
    ap.add_argument("--binarize-method", type=str, default=None,
                    choices=["otsu", "yen", "li", "aucell"])
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    dwn = cfg.setdefault("downstream", {})
    if args.binarize_method: dwn["binarize_method"] = args.binarize_method

    obj_path = Path(cfg["paths"]["obj"]) / "cistopic_obj.pkl"
    with open(obj_path, "rb") as fh:
        cistopic_obj = pickle.load(fh)

    out_dir = Path(cfg["paths"]["downstream"])

    # --- UMAP on topic-cell ---------------------------------------------------
    if not args.no_umap and dwn.get("run_umap", True):
        import umap

        theta = np.asarray(cistopic_obj.selected_model.cell_topic).T   # cells x K
        log.info("Computing UMAP on %d x %d cell-topic matrix", *theta.shape)
        emb = umap.UMAP(n_components=2, random_state=0, metric="euclidean").fit_transform(theta)
        pd.DataFrame(emb, index=cistopic_obj.cell_names, columns=["UMAP1", "UMAP2"]).to_csv(
            out_dir / "umap.tsv", sep="\t"
        )
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(emb[:, 0], emb[:, 1], s=2, alpha=0.5)
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        ax.set_title("cisTopic topic-cell UMAP")
        fig.tight_layout()
        fig.savefig(out_dir / "umap.png", dpi=150)
        plt.close(fig)
        log.info("UMAP saved under %s", out_dir)

    # --- Binarise topics -----------------------------------------------------
    if not args.no_binarize and dwn.get("run_binarize_topics", True):
        from pycisTopic.topic_binarization import binarize_topics

        method = dwn.get("binarize_method", "otsu")
        log.info("Binarising topics with method=%s", method)
        bin_regions = binarize_topics(cistopic_obj, method=method, plot=False)
        with open(out_dir / f"binarized_regions_{method}.pkl", "wb") as fh:
            pickle.dump(bin_regions, fh, protocol=pickle.HIGHEST_PROTOCOL)

        sizes = {t: len(v) for t, v in bin_regions.items()}
        pd.Series(sizes, name="n_regions").to_csv(
            out_dir / f"binarized_regions_{method}.counts.tsv", sep="\t"
        )
        log.info("Binarised regions per topic: %s", sizes)

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
