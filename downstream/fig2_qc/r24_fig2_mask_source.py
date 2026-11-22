#!/usr/bin/env python
"""Fig 2: imputed tracks under the accessibility mask (MASK = "Mdrop5"), added to source_data/heatmaps_<TF>.npz.
MASK "Mdrop5" = 1 kb bins with mean ATAC CPM > 0.0781542 over the two pooled HEK293T ATAC bigwigs (the September
definition; regulatome/m46_stage1_imputed/work/masks_extra.npz). Imputed under the mask = raw-positive cells + model calls
inside the mask (inputs/imp79_mdrop5.npz from r20a_imp_mdrop5.py, built from the genome-wide pre-mask matrices).
Rows/window as r23 (V3 rows, +-10 kb, 21 x 1 kb bins). Arrays: K_impM (fraction of cells), K_maskM (bin inside MASK).
Writes heatmap_mask_share.tsv: share of rows whose summit bin is inside MASK vs inside the ATAC-peak union."""
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/downstream/fig2_qc")
from fig2_common import INP, OUT, REF, ROOT, TFS, bin_index  # noqa: E402

MASK = "Mdrop5"
SD = ROOT / "paper_figures/fig2/source_data"
U = ROOT / "unified/work"
bi, off, N = bin_index()
fr = np.load(INP / "frame.npz", allow_pickle=True)
tf79 = [str(t) for t in fr["tf79"]]; ncell = fr["ncells_imp79"]
mk = np.load(ROOT / "regulatome/m46_stage1_imputed/work/masks_extra.npz")[MASK]
reg = pd.read_csv(U / "nfya/impute/regions.tsv", header=None)[0].str.extract(r"^(chr[^:]+):(\d+)-")
keep = reg[0].isin(off).values
idx = np.array([off[c][0] if c in off else 0 for c in reg[0]]) + reg[1].astype(np.int64).values // 1000
fmask = np.zeros(N, bool); fmask[idx[keep]] = mk[keep]
imp = sp.load_npz(INP / "imp79_mdrop5.npz").tocsr()
print(MASK, "bins on the frame:", int(fmask.sum()), flush=True)
K = np.arange(-10, 11)
rows = []
for t in TFS:
    z = dict(np.load(SD / f"heatmaps_{t}.npz"))
    c3, s3 = z["K_chrom"].astype(str), z["K_summit"].astype(np.int64)   # V3 rows, written by r23
    gb = np.array([off[c][0] for c in c3])[:, None] + (s3 // 1000)[:, None] + K[None, :]
    lo = np.array([off[c][0] for c in c3])[:, None]; hi = lo + np.array([off[c][1] for c in c3])[:, None]
    inside = (gb >= lo) & (gb < hi); gbc = np.clip(gb, 0, N - 1)
    k = tf79.index(t)
    v = imp[k].toarray().ravel() / ncell[k]
    KM = np.where(inside, v[gbc], 0); MM = np.where(inside, fmask[gbc], False)
    z.update(K_impM=KM.astype(np.float32), K_maskM=MM)
    np.savez_compressed(SD / f"heatmaps_{t}.npz", **z)
    rows.append(dict(TF=t, rows=len(c3), summit_bin_inside_Mdrop5=float(MM[:, 10].mean()),
                     summit_bin_inside_ATAC_peak_union=float(z["K_maskA"][:, 10].mean())))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(SD / "heatmap_mask_share.tsv", sep="\t", index=False, float_format="%.4f")
print("done")
