#!/usr/bin/env python3
"""Fig 3 v7 accessibility control, step 1: the region universe U used to draw matched anchors.
U = union (overlaps merged) of every TF's Cicero peaks (cobinding/work/ourrun_k50/<TF>/<TF>.peak_annot.tsv; their sum is the 'all Cicero peaks' n of panel b) and the
HEK293T ATAC union (data/HEK293T_ATAC_union3.bed.gz). Per U region:
  atac_dec  0 = no ATAC union region overlaps; 1-10 = decile of ATAC signal (signal of a union region = max, over the three bulk
            narrowPeak sets gse152177 / gse283384 / OmniATAC IDR, of the within-set percentile of signalValue of overlapping peaks)
  prom      RefSeq Select TSS within the region +-1 kb (same rule as panel b, rebuild_hub_regions_chromhmm.py)
  n_tf      number of distinct TFs with an input peak overlapping the region
Writes inputs/universe.tsv.gz.   (cobinding env)"""
import glob, gzip, json
from pathlib import Path
import numpy as np, pandas as pd
X = Path("/work/users/d/y/dyy12/XuLab"); HERE = Path(__file__).resolve().parent; I = HERE / "inputs"
def ivs_from_names(names):
    out = []
    for n in names:
        c, r = n.split(":"); s, e = r.split("-"); out.append((c, int(s), int(e)))
    return out
rows = []
for f in sorted(glob.glob(str(X / "cobinding/work/ourrun_k50/*/*.peak_annot.tsv"))):
    tf = Path(f).parent.name
    for n in pd.read_csv(f, sep="\t", usecols=["peak"]).peak:
        c, r = n.split(":"); s, e = r.split("-"); rows.append((c, int(s), int(e), tf))
P = pd.DataFrame(rows, columns=["c", "s", "e", "tf"]); print("TF peaks", len(P), "TFs", P.tf.nunique())
A = pd.read_csv(X / "data/HEK293T_ATAC_union3.bed.gz", sep="\t", header=None, usecols=[0, 1, 2], names=["c", "s", "e"])
# ATAC signal of each union region
sig = np.zeros(len(A))
for f in ("HEK293T_ATAC_gse152177_wt_MACS2.narrowPeak.gz", "HEK293T_ATAC_gse283384_wt_MACS2.narrowPeak.gz", "HEK293T_OmniATAC_idr_optimal.narrowPeak.gz"):
    N = pd.read_csv(X / "data" / f, sep="\t", header=None, usecols=[0, 1, 2, 6], names=["c", "s", "e", "v"]); N["pct"] = N.v.rank(pct=True)
    for c, g in A.groupby("c"):
        n = N[N.c == c].sort_values("s"); 
        if n.empty: continue
        ns, ne, npct = n.s.values, n.e.values, n.pct.values; cm = np.maximum.accumulate(ne)
        for i, s, e in zip(g.index, g.s.values, g.e.values):
            j = np.searchsorted(ns, e)          # peaks starting before region end
            lo = np.searchsorted(cm, s, side="right")   # first peak whose running max end > s
            if j > lo:
                m = (ne[lo:j] > s); 
                if m.any(): sig[i] = max(sig[i], npct[lo:j][m].max())
A["sig"] = sig; A["dec"] = np.where(sig > 0, pd.qcut(sig.clip(1e-9), 10, labels=False, duplicates="drop") + 1, 0)
print("ATAC union regions", len(A), "with signal", (sig > 0).mean().round(3))
# union of everything
allv = pd.concat([P[["c", "s", "e"]], A[["c", "s", "e"]]]).sort_values(["c", "s"])
U = []
for c, g in allv.groupby("c", sort=False):
    s0, e0 = None, None
    for s, e in zip(g.s.values, g.e.values):
        if s0 is None: s0, e0 = s, e
        elif s <= e0: e0 = max(e0, e)
        else: U.append((c, s0, e0)); s0, e0 = s, e
    U.append((c, s0, e0))
U = pd.DataFrame(U, columns=["c", "s", "e"]); U["atac_dec"] = 0; U["n_tf"] = 0; U["prom"] = 0
TSS = {}
for f in glob.glob(str(I / "rs/*.json")):
    v = json.load(open(f))["ncbiRefSeqSelect"]
    if v: TSS[Path(f).stem] = np.sort([(g["txStart"] if g["strand"] == "+" else g["txEnd"]) for g in v])
for c, g in U.groupby("c"):
    us = g.s.values; idx = g.index.values
    a = A[A.c == c]; ia = np.searchsorted(us, a.s.values, side="right") - 1
    dmax = pd.Series(a.dec.values).groupby(ia).max(); U.loc[idx[dmax.index.values], "atac_dec"] = dmax.values
    p = P[P.c == c]; ip = np.searchsorted(us, p.s.values, side="right") - 1
    ntf = pd.DataFrame({"i": ip, "tf": p.tf.values}).drop_duplicates().groupby("i").size(); U.loc[idx[ntf.index.values], "n_tf"] = ntf.values
    if c in TSS:
        t = TSS[c]; j = np.searchsorted(t, g.s.values - 1000); ok = (j < len(t)) & (t[np.minimum(j, len(t) - 1)] <= g.e.values + 1000); U.loc[idx, "prom"] = ok.astype(int)
U.to_csv(I / "universe.tsv.gz", sep="\t", index=False)
print("U regions", len(U), "| in ATAC", (U.atac_dec > 0).mean().round(3), "| promoter", U.prom.mean().round(3), "| n_tf>0", (U.n_tf > 0).mean().round(3))
