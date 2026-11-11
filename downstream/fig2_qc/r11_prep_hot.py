#!/usr/bin/env python
"""Round 5 prep: HOT 1 kb bins = bins in the top decile of >= 40 of the 79 TFs (top decile among each TF's nonzero bins,
raw reads); plus the hg38 blacklist bins. Writes OUT/round5_reads/hot_or_blacklist.npy (frame bool) and hot_stats.tsv."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from fig2_common import BLACKLIST, COUNTS, INP, OUT, bin_index, interval_mask, read_bed

O5 = OUT / "round5_reads"
O5.mkdir(parents=True, exist_ok=True)
bi, off, N = bin_index()
raw = sp.load_npz(COUNTS / "raw.npz").tocsr()
cnt = np.zeros(N, np.int32)
for k in range(raw.shape[0]):
    r = raw[k]
    v = r.data
    thr = np.quantile(v, 0.9)
    cnt[r.indices[v >= thr]] += 1
hot = cnt >= 40
bl = interval_mask(read_bed(BLACKLIST), off, N)
np.save(O5 / "hot_or_blacklist.npy", hot | bl)
pd.DataFrame([dict(hot_bins=int(hot.sum()), blacklist_bins=int(bl.sum()), union=int((hot | bl).sum()))]).to_csv(O5 / "hot_stats.tsv", sep="\t", index=False)
print("HOT bins", int(hot.sum()), "blacklist bins", int(bl.sum()))
