#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 03_run_lda_mallet.py
#
# Train a grid of MALLET-backed LDA models on the pickled CistopicObject.
# One model per K in lda.n_topics is trained with the same alpha/eta/seed.
# Every trained model is saved under <work_dir>/models/.
# -----------------------------------------------------------------------------
from __future__ import annotations

import logging
import os
import pickle
import sys
import time
from pathlib import Path

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger("03_lda")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--n-topics", type=int, nargs="+", default=None,
                    help="Override lda.n_topics (space-separated list).")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--threads",    type=int, default=None)
    ap.add_argument("--memory",     type=str, default=None,
                    help="Java heap size for MALLET, e.g. 200G.")
    ap.add_argument("--mallet-path", type=str, default=None)
    ap.add_argument("--tmp-path",   type=str, default=None,
                    help="Scratch directory for MALLET corpus/state files.")
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    lda = cfg.setdefault("lda", {})
    if args.n_topics:   lda["n_topics"]   = args.n_topics
    if args.iterations: lda["iterations"] = args.iterations
    if args.threads:    lda["threads"]    = args.threads
    if args.memory:     lda["memory"]     = args.memory
    if args.mallet_path:
        cfg["paths"]["mallet_path"] = args.mallet_path

    mallet_path = cfg["paths"].get("mallet_path")
    if not mallet_path or not Path(mallet_path).exists():
        log.error("mallet_path does not exist: %r. Download MALLET and update the config.", mallet_path)
        return 1

    os.environ["MALLET_MEMORY"] = str(lda.get("memory", "200G"))

    obj_path = Path(cfg["paths"]["obj"]) / "cistopic_obj.pkl"
    if not obj_path.exists():
        log.error("CistopicObject not found: %s (run 02_build_cistopic_obj.py first)", obj_path)
        return 1

    log.info("Loading CistopicObject from %s", obj_path)
    with open(obj_path, "rb") as fh:
        cistopic_obj = pickle.load(fh)

    try:
        from pycisTopic.lda_models import run_cgs_models_mallet
    except ImportError as e:
        log.error("pycisTopic not importable: %s", e)
        return 2

    tmp_path = Path(args.tmp_path or cfg["paths"].get("tmp_path") or
                    (Path(cfg["paths"]["work_dir"]) / "tmp_mallet"))
    tmp_path.mkdir(parents=True, exist_ok=True)
    save_path = Path(cfg["paths"]["models"])

    n_topics   = list(lda.get("n_topics", [10, 20, 30, 40, 50]))
    iterations = int(lda.get("iterations", 500))
    threads    = int(lda.get("threads", 16))
    alpha      = float(lda.get("alpha", 50))
    alpha_by_topic = bool(lda.get("alpha_by_topic", True))
    eta        = float(lda.get("eta", 0.1))
    eta_by_topic   = bool(lda.get("eta_by_topic", False))
    seed       = int(lda.get("random_seed", 555))

    log.info(
        "Running MALLET LDA: K=%s, iters=%d, threads=%d, alpha=%s (by_topic=%s), eta=%s",
        n_topics, iterations, threads, alpha, alpha_by_topic, eta,
    )

    models = []
    total_models = len(n_topics)
    for idx, topic_count in enumerate(n_topics, start=1):
        log.info(
            "[%d/%d] Starting MALLET model for K=%d",
            idx, total_models, topic_count,
        )
        model_start = time.perf_counter()
        trained = run_cgs_models_mallet(
            cistopic_obj,
            n_topics=[topic_count],
            n_cpu=threads,
            n_iter=iterations,
            random_state=seed,
            alpha=alpha,
            alpha_by_topic=alpha_by_topic,
            eta=eta,
            eta_by_topic=eta_by_topic,
            tmp_path=str(tmp_path),
            save_path=str(save_path),
            mallet_path=str(mallet_path),
        )
        elapsed_s = time.perf_counter() - model_start
        if isinstance(trained, list):
            models.extend(trained)
        else:
            models.append(trained)
        log.info(
            "[%d/%d] Finished K=%d in %.1f min",
            idx, total_models, topic_count, elapsed_s / 60.0,
        )

    out = save_path / "models.pkl"
    log.info("Pickling %d model(s) -> %s", len(models), out)
    with open(out, "wb") as fh:
        pickle.dump(models, fh, protocol=pickle.HIGHEST_PROTOCOL)

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
