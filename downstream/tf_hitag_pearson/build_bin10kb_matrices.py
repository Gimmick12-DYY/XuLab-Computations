#!/usr/bin/env python3
"""Assemble the simplified 10 kb Pearson matrices from the saved signal arrays.

Per TF, columns in order, omitting a track that was not scored:
  Raw, Imputed, Pred, Bulk

Also writes raw-vs-imputed (all TFs) and imputed-vs-pred (TFs with Chromnitron).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from build_bin1k_correlations import pairwise, plot_matrix, write_tsv

ROOT = Path("/work/users/d/y/dyy12/XuLab")
SRC = ROOT / "downstream" / "archive" / "pearson_10kb" / "all_tfs_aggregate"
OUT = ROOT / "downstream" / "tf_hitag_pearson" / "bin10kb"


def _col(z, key: str):
    if key not in z.files:
        return None
    return np.asarray(z[key], dtype=np.float64)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(SRC.glob("*_signals_10kb.npz"))
    if not paths:
        raise SystemExit(f"no signal arrays in {SRC}")
    tfs = [p.name.replace("_signals_10kb.npz", "") for p in paths]
    raw, imp, pred, bulk = {}, {}, {}, {}
    for tf, path in zip(tfs, paths):
        z = np.load(path)
        raw[tf] = _col(z, "HiTAG_aggregate")
        imp[tf] = _col(z, "Imputed_aggregate")
        pred[tf] = _col(z, "Chromnitron")
        bulk[tf] = _col(z, "Bulk")
        z.close()
    if any(v is None for v in raw.values()) or any(v is None for v in imp.values()):
        raise SystemExit("every TF needs HiTAG_aggregate and Imputed_aggregate")

    no_pred = [tf for tf in tfs if pred[tf] is None]
    (OUT / "tfs_without_pred.txt").write_text("\n".join(no_pred) + "\n")
    print(f"tfs {len(tfs)} no Pred {len(no_pred)} with bulk {sum(bulk[t] is not None for t in tfs)}", flush=True)

    Rraw = np.column_stack([raw[tf] for tf in tfs])
    Rimp = np.column_stack([imp[tf] for tf in tfs])
    print("raw vs imputed", flush=True)
    R_ri = pairwise(Rraw, Rimp)
    write_tsv(OUT / "raw_vs_imputed_pearson.tsv", tfs, tfs, R_ri)
    plot_matrix(OUT / "raw_vs_imputed_pearson.png", tfs, tfs, R_ri, "Raw vs Imputed", numbers=True)

    pred_tfs = [tf for tf in tfs if pred[tf] is not None]
    P = np.column_stack([pred[tf] for tf in pred_tfs])
    I = np.column_stack([imp[tf] for tf in pred_tfs])
    print("imputed vs pred", flush=True)
    R_ip = pairwise(I, P)
    write_tsv(OUT / "imputed_vs_pred_pearson.tsv", pred_tfs, pred_tfs, R_ip)
    plot_matrix(OUT / "imputed_vs_pred_pearson.png", pred_tfs, pred_tfs, R_ip, "Imputed vs Pred", numbers=True)

    for tf in tfs:
        labels = ["Raw", "Imputed"]
        cols = [raw[tf], imp[tf]]
        if pred[tf] is not None:
            labels.append("Pred")
            cols.append(pred[tf])
        if bulk[tf] is not None:
            labels.append("Bulk")
            cols.append(bulk[tf])
        M = np.column_stack(cols)
        R = pairwise(M, M)
        write_tsv(OUT / f"{tf}_pearson.tsv", labels, labels, R)
        plot_matrix(OUT / f"{tf}_pearson.png", labels, labels, R, tf, numbers=True)
    print(f"[done] {OUT}", flush=True)
    print(f"TFs without Pred ({len(no_pred)}): {', '.join(no_pred)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
