#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 04_select_tracks.py
#
# Collapse the (n_bins, n_tracks) Borzoi output into a per-bin scalar prior
# for the TF this run is about.
#
# Cheap, rerunnable: this step does not touch the GPU. Edit
# tracks.{pattern,aggregate,apply_sigmoid} in default.yaml to re-aggregate
# without rerunning 03.
#
# Inputs:
#   <work>/predict/bin_track.h5
#
# Outputs (under <work>/predict/):
#   bin_score.npy        (n_kept_bins,) float32 in [0, 1] if apply_sigmoid
#   bin_score_meta.json  selected tracks, aggregate, n_bins
# -----------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import h5py
import numpy as np

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger("04_tracks")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--pattern", type=str, default=None,
                    help="Override tracks.pattern from config")
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    t_cfg = cfg.setdefault("tracks", {})
    pattern   = args.pattern or t_cfg.get("pattern", ".*")
    aggregate = str(t_cfg.get("aggregate", "mean")).lower()
    do_sigmoid = bool(t_cfg.get("apply_sigmoid", True))

    in_path = Path(cfg["paths"]["predict"]) / "bin_track.h5"
    if not in_path.is_file():
        log.error("Missing %s -- run 03_borzoi_predict.py first", in_path)
        return 1

    with h5py.File(in_path, "r") as h5:
        track_names = [
            s.decode() if isinstance(s, bytes) else s for s in h5["track_names"][:]
        ]
        X = np.asarray(h5["X"])           # (n_bins, n_tracks) f16
        mm_row_idx = np.asarray(h5["mm_row_idx"])
    log.info("bin_track.h5: %d bins x %d tracks", *X.shape)

    rx = re.compile(pattern)
    keep_track = np.asarray([bool(rx.search(n)) for n in track_names], dtype=bool)
    n_keep = int(keep_track.sum())
    if n_keep == 0:
        log.error("tracks.pattern %r matched 0 of %d tracks", pattern, len(track_names))
        return 2
    log.info("tracks.pattern %r: kept %d / %d tracks", pattern, n_keep, len(track_names))
    for name in [n for n, k in zip(track_names, keep_track) if k][:10]:
        log.info("  +  %s", name)
    if n_keep > 10:
        log.info("  +  ... and %d more", n_keep - 10)

    Xk = X[:, keep_track].astype(np.float32)
    if aggregate == "mean":
        score = Xk.mean(axis=1)
    elif aggregate == "max":
        score = Xk.max(axis=1)
    elif aggregate == "sum":
        score = Xk.sum(axis=1)
    else:
        log.error("Unknown tracks.aggregate %r", aggregate)
        return 3

    if do_sigmoid:
        # Borzoi outputs are softplus-ed counts (positive, unbounded). A plain
        # sigmoid is the wrong squash; use a robust min/max rescale to [0, 1].
        lo = float(np.quantile(score, 0.001))
        hi = float(np.quantile(score, 0.999))
        if hi > lo:
            score = np.clip((score - lo) / (hi - lo), 0.0, 1.0)
        else:
            score = np.zeros_like(score)
        log.info("Squashed score to [0,1] via robust min/max (q0.001=%.3f, q0.999=%.3f)", lo, hi)

    out_dir = Path(cfg["paths"]["predict"])
    np.save(out_dir / "bin_score.npy", score.astype(np.float32))
    np.save(out_dir / "bin_score_mm_row_idx.npy", mm_row_idx.astype(np.int64))

    meta = {
        "pattern":         pattern,
        "aggregate":       aggregate,
        "apply_sigmoid":   do_sigmoid,
        "n_tracks_kept":   n_keep,
        "n_bins":          int(len(score)),
        "score_min":       float(score.min()),
        "score_max":       float(score.max()),
        "score_mean":      float(score.mean()),
        "tracks_kept":     [n for n, k in zip(track_names, keep_track) if k][:200],
    }
    (out_dir / "bin_score_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("Done: %s", {k: meta[k] for k in
                          ("pattern", "n_tracks_kept", "n_bins",
                           "score_min", "score_max", "score_mean")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
