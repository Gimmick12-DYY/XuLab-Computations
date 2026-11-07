#!/usr/bin/env python
"""Re-derive the panel-dependent inputs after a change of TFS (2026-11-06: NFYA dropped) without
recomputing GC/CpG/ATAC/imputed loading. Rewrites in OUT/inputs/: frame.npz (ref, refcount, unique,
shared3, npeaks, nunique_peaks), unique_<TF>.bed, shared3_bins.bed, raw_panel.npy, imp_panel.npy.
Same definitions as 00_build_inputs.py.
"""
import subprocess
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp

from fig2_common import BIN, COUNTS, INP, NT, REF, TFS, bin_index, interval_mask, read_bed

bi, off, N = bin_index()
fr = dict(np.load(INP / "frame.npz", allow_pickle=True))
for f in INP.glob("unique_*.bed"):
    f.unlink()
ref = np.zeros((NT, N), bool)
for k, tf in enumerate(TFS):
    ref[k] = interval_mask(read_bed(REF[tf]), off, N)
refcount = ref.sum(0)
tmp = tempfile.mkdtemp(dir=INP)
for tf in TFS:
    read_bed(REF[tf])[[0, 1, 2]].sort_values([0, 1]).to_csv(f"{tmp}/{tf}.bed", sep="\t", header=False, index=False)
unique = np.zeros((NT, N), bool)
npk, nun = [], []
for k, tf in enumerate(TFS):
    others = [f"{tmp}/{o}.bed" for o in TFS if o != tf]
    res = subprocess.run(["bedtools", "intersect", "-v", "-a", f"{tmp}/{tf}.bed", "-b", *others],
                         check=True, capture_output=True, text=True).stdout
    (INP / f"unique_{tf}.bed").write_text(res)
    ub = read_bed(INP / f"unique_{tf}.bed")
    npk.append(sum(1 for _ in open(f"{tmp}/{tf}.bed"))); nun.append(len(ub))
    unique[k] = interval_mask(ub, off, N) & (refcount == 1)
    print(tf, "peaks", npk[-1], "unique peaks", nun[-1], "unique bins", int((unique[k] & fr["universe"]).sum()), flush=True)
shared3 = refcount >= 3
sh = np.flatnonzero(shared3)
cc = np.searchsorted(bi.offset.values, sh, side="right") - 1
pd.DataFrame({0: bi.chrom.values[cc], 1: (sh - bi.offset.values[cc]) * BIN,
              2: (sh - bi.offset.values[cc] + 1) * BIN}).to_csv(INP / "shared3_bins.bed", sep="\t", header=False, index=False)

tf79 = list(fr["tf79"])
raw = sp.load_npz(COUNTS / "raw.npz").tocsr()
np.save(INP / "raw_panel.npy", np.vstack([raw[tf79.index(t)].toarray().ravel() for t in TFS]).astype(np.int32))
imp79 = sp.load_npz(INP / "imp79.npz").tocsr()
np.save(INP / "imp_panel.npy", np.vstack([imp79[tf79.index(t)].toarray().ravel() for t in TFS]).astype(np.float32))
for old in ("raw7.npy", "imp7.npy"):
    (INP / old).unlink(missing_ok=True)
fr.update(ref=ref, refcount=refcount.astype(np.int8), unique=unique, shared3=shared3, npeaks=np.array(npk),
          nunique_peaks=np.array(nun), panel=np.array(TFS))
np.savez_compressed(INP / "frame.npz", **fr)
print("shared>=3 bins", int((shared3 & fr["universe"]).sum()))
