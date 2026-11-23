#!/usr/bin/env python
"""Continuous imputed scores for the Fig 2 heatmap alternative v4_continuous: the model's pseudobulk mean predicted
probability per 1 kb bin (unified/work/<tf>/predict/pseudobulk_audit_bins.npz, mean_score over cells; written by
r9a_continuous_scores.py). These exist only for the audit bins (summit bin +-2 of the round-2 summits), so each V3 row
(heatmaps_<TF>.npz K_summit) gets the scores of its summit bin +-2 (5 bins, +-2 kb) when all five are audit bins.
Adds K_cont (rows x 5, NaN when not covered) and K_cont_ok (all five covered) to source_data/heatmaps_<TF>.npz and
writes source_data/heatmap_cont_coverage.tsv."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/downstream/fig2_qc")
from fig2_common import OUT, ROOT, TFS, bin_index  # noqa: E402

SD = ROOT / "paper_figures/fig2/source_data"
OC = OUT / "imputed_audit/continuous"
bi, off, N = bin_index()
bf = np.load(OC / "bins_frame.npy"); bm = np.load(OC / "bins_mm.npy")
rows = []
for t in TFS:
    p = np.load(ROOT / "unified/work" / t.lower() / "predict/pseudobulk_audit_bins.npz")
    assert np.array_equal(p["bins_mm"], bm), "audit bin order differs"
    v = np.full(N, np.nan, np.float32); v[bf] = p["mean_score"]
    z = dict(np.load(SD / f"heatmaps_{t}.npz"))
    c, s = z["K_chrom"].astype(str), z["K_summit"].astype(np.int64)
    gb = np.array([off[x][0] for x in c])[:, None] + (s // 1000)[:, None] + np.arange(-2, 3)[None, :]
    K = v[np.clip(gb, 0, N - 1)]
    ok = np.isfinite(K).all(1)
    z.update(K_cont=K, K_cont_ok=ok)
    np.savez_compressed(SD / f"heatmaps_{t}.npz", **z)
    rows.append(dict(TF=t, rows=len(ok), rows_covered=int(ok.sum()), share_covered=float(ok.mean()), n_cells_used=int(p["n_cells_used"])))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(SD / "heatmap_cont_coverage.tsv", sep="\t", index=False, float_format="%.4f")
