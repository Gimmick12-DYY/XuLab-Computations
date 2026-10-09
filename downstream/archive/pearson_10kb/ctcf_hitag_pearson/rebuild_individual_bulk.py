#!/usr/bin/env python3
"""Rebuild CTCF 10 kb matrices: individual bulk + log1p imputed."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import rankdata

OUT = Path("/work/users/d/y/dyy12/XuLab/downstream/ctcf_hitag_pearson")

BULK_LABELS = ["ENCODE", "GSE103651", "293Tcon1"]


def write_tsv(path: Path, labels: list[str], R: np.ndarray) -> None:
    with path.open("w") as f:
        f.write("track\t" + "\t".join(labels) + "\n")
        for i, lab in enumerate(labels):
            f.write(lab + "\t" + "\t".join(f"{R[i, j]:.4f}" for j in range(len(labels))) + "\n")
    print(f"wrote {path}")


def main() -> None:
    z = np.load(OUT / "signals_partial.npz", allow_pickle=True)
    chrom = z["chrom"].astype(str)
    start, end = z["start"], z["end"]
    index = {(chrom[i], int(start[i]), int(end[i])): i for i in range(chrom.size)}
    bulk = np.full((chrom.size, 3), np.nan)
    with open(OUT / "consensus_bulk.tab") as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            i = index[(p[0].strip("'"), int(p[1]), int(p[2]))]
            for j in range(3):
                v = p[3 + j]
                bulk[i, j] = np.nan if v in ("nan", "NaN", "") else float(v)

    agg = np.load(OUT / "imputed_aggregate_10kb.npz")["score"].astype(float)
    imputed_log = np.log1p(np.clip(z["imputed"].astype(float), 0, None))
    agg_log = np.log1p(np.clip(agg, 0, None))
    # Chromnitron = SA replicate (former pred_2); eLife/pred_1 dropped.
    labels = [
        "Chromnitron",
        "ENCODE",
        "GSE103651",
        "293Tcon1",
        "HiTAG_weight",
        "HiTAG_macs2",
        "HiTAG_aggregate",
        "Imputed_log1p",
        "Imputed_aggregate_log1p",
    ]
    cols = [
        z["Chromnitron_pred2"].astype(float),
        bulk[:, 0],
        bulk[:, 1],
        bulk[:, 2],
        z["HiTAG_weight"].astype(float),
        z["HiTAG_macs2"].astype(float),
        z["HiTAG_aggregate"].astype(float),
        imputed_log,
        agg_log,
    ]
    X = np.column_stack(cols)
    keep = np.isfinite(X).all(axis=1)
    Xk = X[keep]
    print(f"complete bins {int(keep.sum()):,} / {len(keep):,}")
    for method, stem in (("pearson", "CTCF_pearson_10kb"), ("spearman", "CTCF_spearman_10kb")):
        if method == "pearson":
            R = np.corrcoef(Xk, rowvar=False)
        else:
            ranks = np.column_stack([rankdata(Xk[:, j], method="average") for j in range(Xk.shape[1])])
            R = np.corrcoef(ranks, rowvar=False)
        write_tsv(OUT / f"{stem}.tsv", labels, R)
        for src in ("Imputed_log1p", "Imputed_aggregate_log1p"):
            ia = labels.index(src)
            for lab in ("ENCODE", "GSE103651", "293Tcon1", "HiTAG_macs2", "HiTAG_aggregate"):
                j = labels.index(lab)
                print(f"  {method} {src} vs {lab}: {R[ia, j]:.3f}")


if __name__ == "__main__":
    main()
