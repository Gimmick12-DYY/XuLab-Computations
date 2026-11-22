#!/usr/bin/env python
"""Write the Figure 2 source data into paper_figures/fig2/source_data/ (computation side; the panel scripts only render).
1 corr_matrix_6x6.tsv: summit-group Pearson 6x6 (round 3 recipe: preferred top-5,000 summits per reference, 30 groups,
  bulk log CPM +-250 bp, sc centre/flank vs the leave-one-out 79-TF pool, ZNF777 depth, 3 seeds), copied from
  round2/pearson/r2p_corr_preferred_top5000_znf777.tsv (rows sc, columns ChIP).
2 heatmaps_<TF>.npz (float16; not tracked, *.npz): per TF, +-1 kb at 50 bp (40 bins), from the r17 caches
  (round6_diffpeaks/heatmaps/kayaokur_style/cache_<TF>.pkl):
    V1  each dataset on its own peaks, sorted by its own centre signal (+-200 bp): ChIP peaks, lab MACS2 q<0.05 peaks
        (raw), top imputed 1 kb bins (as many as raw peaks).
    V2  one region set = ChIP summits + our raw summits > 500 bp from any ChIP peak; same rows, sorted by ChIP centre.
    V3  the reference's top 10,000 ChIP summits by signalValue (Kaya-Okur 2019 Fig 2g convention), same rows in every
        panel, sorted by ChIP centre signal.
  Arrays: <ver>_<track> (rows x 40, already in display order), plus counts.
3 heatmap_counts.tsv: number of sites per TF and version."""
import pickle
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/downstream/fig2_qc")
from fig2_common import OUT, REF, ROOT, TFS, bin_index  # noqa: E402

SD = ROOT / "paper_figures/fig2/source_data"; SD.mkdir(parents=True, exist_ok=True)
KO = OUT / "round6_diffpeaks/heatmaps/kayaokur_style"
shutil.copy(OUT / "round2/pearson/r2p_corr_preferred_top5000_znf777.tsv", SD / "corr_matrix_6x6_source.tsv")
B = pd.read_csv(OUT / "round2/pearson/r2p_corr_preferred_top5000_znf777.tsv", sep="\t", index_col=0)
M = B.iloc[6:, :6]
M.to_csv(SD / "corr_matrix_6x6.tsv", sep="\t", float_format="%.5f")
print(M.round(3).to_string())

W, NB = 1000, 40
CTR = slice(NB // 2 - 4, NB // 2 + 4)
CH = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
PEAKS = dict(REF); PEAKS["CTCF"] = ROOT / "downstream/tracks/ctcf_bulk_sources/1_ENCODE_HEK293_ENCFF314ZAL_peaks.bed"
bi, off, N = bin_index(); csize = dict(zip(bi.chrom, bi["size"]))
rows = []
for t in TFS:
    d, c = pickle.load(open(KO / f"cache_{t}.pkl", "rb"))
    out = {}
    for ty in ("ChIP", "raw", "imputed"):
        out[f"V1_{ty}"] = d["V1"][ty][d["V1o"][ty]].astype(np.float16)
        out[f"V2_{ty}"] = d["V2"][ty][d["o2b"]].astype(np.float16)
    # V3: top 10,000 ChIP summits; the first n_chip rows of V2 are the ChIP summits in file order (same filter as r17)
    P = pd.read_csv(PEAKS[t], sep="\t", header=None, usecols=range(10)); P = P[P[0].isin(CH)]
    sm = (P[1] + P[9].clip(lower=0)).values
    ok = np.array([ch in csize and x - W >= 0 and x + W <= csize[ch] for ch, x in zip(P[0].values, sm)])
    sig = P[6].values[ok]
    assert len(sig) == c["chip_peaks"], (t, len(sig), c["chip_peaks"])
    top = np.argsort(-sig, kind="stable")[:10000]
    o3 = top[np.argsort(-d["V2"]["ChIP"][top][:, CTR].mean(1), kind="stable")]
    for ty in ("ChIP", "raw", "imputed"):
        out[f"V3_{ty}"] = d["V2"][ty][o3].astype(np.float16)
    np.savez_compressed(SD / f"heatmaps_{t}.npz", **out)
    rows.append(dict(TF=t, V1_ChIP=len(out["V1_ChIP"]), V1_raw=len(out["V1_raw"]), V1_imputed=len(out["V1_imputed"]),
                     V2=len(out["V2_ChIP"]), V3=len(out["V3_ChIP"])))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(SD / "heatmap_counts.tsv", sep="\t", index=False)
print("done")
