# m7_step2: raw per-TF fragment 5' ends counted over the canonical ABC candidate elements (m2c_abc_canonical HEK293T_hicfixed EnhancerList). usage: count_canonical.py
# 5' end: + strand start, - strand end - 1 (single-end reads, not Tn5-shifted, not deduplicated). Writes features/canon_counts/raw_counts.npy (element x TF) and elements.tsv.
import os
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; R = f"{X}/regulatome"; O = f"{R}/m7_step2"; Out = f"{O}/features/canon_counts"; os.makedirs(Out, exist_ok=True)
tfs = [l.strip() for l in open(f"{R}/m1_binding/v2/tfs.txt") if l.strip()]
E = pd.read_csv(f"{R}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/EnhancerList.txt", sep="\t", usecols=["chr", "start", "end"]); NE = len(E); E = E.reset_index().rename(columns={"index": "i"})
idx = {}
for c, G in E.groupby("chr"):
    Gs = G.sort_values("start"); s, e, ix = Gs.start.values, Gs.end.values, Gs.i.values; assert (s[1:] >= e[:-1]).all(), c; idx[c] = (s, e, ix)
C = np.zeros((NE, len(tfs)), np.float32); tot = []
for j, t in enumerate(tfs):
    F = pd.read_csv(f"/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment/TF.{t}.bed.gz", sep="\t", header=None, usecols=[0, 1, 2, 5], names=["c", "s", "e", "st"], dtype={"c": str, "s": np.int64, "e": np.int64, "st": str})
    pos = np.where(F.st.values == "+", F.s.values, F.e.values - 1); n = 0
    for c, ix_c in F.groupby("c", sort=False).indices.items():
        if c not in idx: continue
        s, e, ix = idx[c]; p = pos[ix_c]; k = np.searchsorted(s, p, side="right") - 1; ok = k >= 0; ok[ok] = p[ok] < e[k[ok]]; np.add.at(C[:, j], ix[k[ok]], 1); n += int(ok.sum())
    tot.append(dict(tf=t, reads=len(F), in_elements=n)); print(t, len(F), n, flush=True)
np.save(f"{Out}/raw_counts.npy", C); pd.DataFrame(tot).to_csv(f"{Out}/read_totals.tsv", sep="\t", index=False); print("done")
