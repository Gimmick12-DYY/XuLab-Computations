# Raw read 5' ends per union-ATAC element for one TF (fragment beds read only). usage: cm_count.py <TF>
import sys
import numpy as np, pandas as pd
from cm_common import *
t = sys.argv[1]; E = load_elements()
r = pd.read_csv(f"/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment/TF.{t}.bed.gz", sep="\t", header=None, usecols=[0, 1], names=["chrom", "start"], dtype={"chrom": str, "start": np.int64})
y = np.zeros(len(E), np.int64)
for c, d in E.groupby("chrom"):
    p = np.sort(r.start.values[r.chrom.values == c]); y[d.index.values] = np.searchsorted(p, d.end.values) - np.searchsorted(p, d.start.values)
np.savez_compressed(f"{U}/counts/{t}.npz", y=y, total=len(r)); print(t, len(r), int(y.sum()))
