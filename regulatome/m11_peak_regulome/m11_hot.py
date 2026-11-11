# m11 Amendment 4: HOT bins. usage: m11_hot.py count <i> -> recall/hot/counts_<TF>.npy ; m11_hot.py combine -> recall/hot_bins.bed
import os, sys
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; H = f"{RG}/m11_peak_regulome/recall/hot"; os.makedirs(H, exist_ok=True)
tfs = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]
Q = pd.read_csv(f"{X}/complex_screen/pair_motif_composite/ref/bins_qc.tsv", sep="\t", usecols=["chrom", "s", "e", "bin", "exonic", "blacklist", "n_ok"])
Q = Q[(~Q.blacklist.astype(bool)) & (Q.exonic <= .5) & Q.n_ok.astype(bool)].reset_index(drop=True); Q["s1"] = Q.s - 400; Q["e1"] = Q.e + 400
if sys.argv[1] == "count":
    t = tfs[int(sys.argv[2])]; F = pd.read_csv(f"/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment/TF.{t}.bed.gz", sep="\t", header=None, usecols=[0, 1, 2, 5], names=["c", "s", "e", "st"], dtype={"c": str, "s": np.int64, "e": np.int64, "st": str})
    pos = np.where(F.st.values == "+", F.s.values, F.e.values - 1); n = np.zeros(len(Q), np.int32)
    for c, ix in F.groupby("c", sort=False).indices.items():
        m = np.flatnonzero(Q.chrom.values == c)
        if not len(m): continue
        p = np.sort(pos[ix]); n[m] = np.searchsorted(p, Q.e1.values[m], side="left") - np.searchsorted(p, Q.s1.values[m], side="left")
    np.save(f"{H}/counts_{t}.npy", n); print(t, len(Q), int(n.sum()))
else:
    C = np.vstack([np.load(f"{H}/counts_{t}.npy") for t in tfs]); top = np.zeros_like(C, bool)
    for i in range(len(tfs)): o = np.argsort(-C[i], kind="stable")[:int(round(C.shape[1] * .1))]; top[i, o] = True
    k = top.sum(0); out = Q[["chrom", "s1", "e1", "bin"]].assign(n_tfs_top_decile=k); out[k >= 40].to_csv(f"{RG}/m11_peak_regulome/recall/hot_bins.bed", sep="\t", header=False, index=False)
    out.to_csv(f"{H}/all_bins_top_decile_counts.tsv.gz", sep="\t", index=False); print("eligible", len(Q), "HOT (>=40 TFs)", int((k >= 40).sum()), "quantiles", np.percentile(k, [50, 90, 99, 100]))
