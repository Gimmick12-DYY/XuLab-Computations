# Peak re-calling, shared inputs (2026-11-11): per union-ATAC 1 kb bin -> ATAC CPM (HEK293T OmniATAC), ATAC-level quintile, HOT flag,
# matching strata; imputed "re-called" scores per TF: (i) loo = log2((S_t + e) / (mean of the other 78 TFs + e)), (ii) atac =
# log1p(S_t) minus its median within 50 ATAC-CPM quantile bins (PATTY-like ATAC regression). HOT = bin in the top 5% of raw CPM
# (depth-equalized 5,903-cell counts) for > 50% of TFs.
# Output recall/ref.npz (atac_cpm, atac_q, hot, strat, elig, part), recall/imputed_scores.npz (loo, atac, tfs)
import os, gzip, glob
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; R = f"{O}/recall"
TA = f"{X}/atac_omniatac/caper_work/atac/70bfc7e8-3760-496f-bd8e-5a4bc9e02b51/call-bam2ta"
Q = pd.read_csv(f"{O}/ref/bins_qc.tsv", sep="\t"); B = pd.read_csv(f"{O}/ref/bins1kb.bed", sep="\t", header=None, names=["c", "s", "e", "i"]); n = len(B)
num = Q.chrom.str.replace("chr", ""); part = np.where(num.isin([str(i) for i in range(1, 23, 2)]), "disc", np.where(num.isin([str(i) for i in range(2, 23, 2)] + ["X"]), "val", "none"))
cnt = np.zeros(n); tot = 0
for f in sorted(glob.glob(f"{TA}/shard-*/execution/*.tn5.tagAlign.gz")):
    for ch in pd.read_csv(f, sep="\t", header=None, usecols=[0, 1], names=["c", "p"], chunksize=5_000_000):
        tot += len(ch)
        for c, g in ch.groupby("c"):
            b = B[B.c == c]
            if not len(b): continue
            k = np.searchsorted(b.s.values, g.p.values, side="right") - 1; ok = (k >= 0) & (g.p.values < b.e.values[np.maximum(k, 0)])
            np.add.at(cnt, b.i.values[k[ok]], 1)
    print(f, tot, flush=True)
atac = cnt / tot * 1e6
cells = pd.read_csv(f"{X}/complex_screen/pair_motif_lookup/tf_cells.tsv", sep="\t").set_index("tf").n_cells; tfr = list(cells.index)
C = np.stack([np.load(f"{O}/reversed/depth/counts/{t}.npz")["N20"][0] for t in tfr]); C = C / C.sum(1, keepdims=True) * 1e6
hot = (C >= np.quantile(C, 0.95, axis=1, keepdims=True)).mean(0) > 0.5
elig = Q.n_ok.values & ~Q.blacklist.values & (Q.exonic.values <= 0.5) & (part != "none"); ei = np.flatnonzero(elig)
prom = np.zeros(n, bool); prom[[int(x) for x in open(f"{O}/ref/promoter_bins.txt").read().split()]] = True
cq = np.quantile(Q.cpg_oe.values[ei], [0.2, 0.4, 0.6, 0.8]); aq = np.quantile(atac[ei], [0.2, 0.4, 0.6, 0.8]); atac_q = np.searchsorted(aq, atac)
strat = ((np.floor(Q.gc.values / 0.025).astype(int) * 10 + np.searchsorted(cq, Q.cpg_oe.values)) * 2 + prom) * 5 + atac_q
np.savez_compressed(f"{R}/ref.npz", atac_cpm=atac, atac_q=atac_q, hot=hot, strat=strat, elig=elig, part=part, prom=prom)
z = np.load(f"{X}/complex_screen/motif_shared_20261103/fg/signal_mask.npz"); S = z["S"]; tfs = [t.upper() for t in z["tfs"]]
e = 0.1 * S[S > 0].mean(); loo = np.log2((S + e) / ((S.sum(0, keepdims=True) - S) / (len(tfs) - 1) + e))
ab = np.searchsorted(np.quantile(atac[ei], np.linspace(0, 1, 51)[1:-1]), atac); L = np.log1p(S); res = np.zeros_like(L)
for b in np.unique(ab): q = ab == b; res[:, q] = L[:, q] - np.median(L[:, q & elig], axis=1, keepdims=True) if (q & elig).any() else 0
np.savez_compressed(f"{R}/imputed_scores.npz", loo=loo.astype(np.float32), atac=res.astype(np.float32), tfs=np.array(tfs))
print("bins", n, "eligible", len(ei), "HOT", int(hot.sum()), "HOT eligible", int((hot & elig).sum()), "HOT promoter frac", round(prom[hot].mean(), 3))
print("corr(imputed mean, ATAC) =", round(np.corrcoef(S.mean(0)[ei], np.log1p(atac[ei]))[0, 1], 3))
