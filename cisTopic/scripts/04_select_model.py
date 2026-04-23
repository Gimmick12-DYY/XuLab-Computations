#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 04_select_model.py
#
# Evaluate the LDA grid with the four pycisTopic metrics, write a PNG with
# all curves, pick the winning K according to select.primary_metric, attach
# the chosen model to the CistopicObject, and re-pickle.
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

from _cfg import base_parser, load_config, resolve_paths  # noqa: E402

log = logging.getLogger("04_select")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--primary-metric", type=str, default=None,
                    choices=["arun_2010", "cao_juan_2009", "mimno_2011",
                             "loglikelihood", "manual"])
    ap.add_argument("--manual-k", type=int, default=None)
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    sel = cfg.setdefault("select", {})
    if args.primary_metric: sel["primary_metric"] = args.primary_metric
    if args.manual_k is not None: sel["manual_k"] = args.manual_k

    models_path = Path(cfg["paths"]["models"]) / "models.pkl"
    obj_path    = Path(cfg["paths"]["obj"])    / "cistopic_obj.pkl"

    if not models_path.exists() or not obj_path.exists():
        log.error("Inputs missing (models=%s, obj=%s). Run earlier steps first.",
                  models_path, obj_path)
        return 1

    log.info("Loading models + CistopicObject")
    with open(models_path, "rb") as fh:
        models = pickle.load(fh)
    with open(obj_path, "rb") as fh:
        cistopic_obj = pickle.load(fh)

    from pycisTopic.lda_models import evaluate_models

    metric = sel.get("primary_metric", "loglikelihood")

    if metric == "manual":
        k = int(sel["manual_k"])
        chosen = next((m for m in models if m.n_topic == k), None)
        if chosen is None:
            log.error("manual_k=%d not found among trained models (K=%s)",
                      k, [m.n_topic for m in models])
            return 1
    else:
        log.info("Evaluating models with pycisTopic...")
        chosen = evaluate_models(
            models,
            select_model=None,
            return_model=True,
            metrics=["Arun_2010", "Cao_Juan_2009", "Mimno_2011", "loglikelihood"],
            plot_metrics=False,
        )
        if metric != "loglikelihood":
            name_map = {
                "arun_2010":     "Arun_2010",
                "cao_juan_2009": "Cao_Juan_2009",
                "mimno_2011":    "Mimno_2011",
            }
            target = name_map[metric]
            # lower-is-better for Arun + Cao; higher-is-better for Mimno
            vals = np.array([getattr(m.metrics, target, np.nan) for m in models], dtype=float)
            ks   = np.array([m.n_topic for m in models])
            k = int(ks[np.nanargmin(vals)] if target != "Mimno_2011" else ks[np.nanargmax(vals)])
            chosen = next(m for m in models if m.n_topic == k)

    log.info("Selected model: K=%d", chosen.n_topic)

    # Plot all metric curves
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ks = np.array([m.n_topic for m in models])
    order = np.argsort(ks)
    ks = ks[order]
    models_sorted = [models[i] for i in order]

    for ax, name, lower_better in [
        (axes[0, 0], "Arun_2010",     True),
        (axes[0, 1], "Cao_Juan_2009", True),
        (axes[1, 0], "Mimno_2011",    False),
        (axes[1, 1], "loglikelihood", False),
    ]:
        try:
            vals = [getattr(m.metrics, name) for m in models_sorted]
        except AttributeError:
            continue
        ax.plot(ks, vals, marker="o")
        ax.axvline(chosen.n_topic, color="red", ls="--", lw=1)
        ax.set_xlabel("n topics")
        ax.set_title(f"{name} ({'lower better' if lower_better else 'higher better'})")

    fig.suptitle(f"cisTopic model selection ({metric}) — chose K = {chosen.n_topic}")
    out_png = Path(cfg["paths"]["select"]) / "model_selection.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", out_png)

    # Attach chosen model to the CistopicObject
    cistopic_obj.add_LDA_model(chosen)

    with open(obj_path, "wb") as fh:
        pickle.dump(cistopic_obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Re-pickled CistopicObject with the selected model embedded.")

    # Also persist the chosen model standalone
    with open(Path(cfg["paths"]["select"]) / "selected_model.pkl", "wb") as fh:
        pickle.dump(chosen, fh, protocol=pickle.HIGHEST_PROTOCOL)

    (Path(cfg["paths"]["select"]) / "selected_model.meta.yaml").write_text(
        f"metric: {metric}\nn_topics: {chosen.n_topic}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
