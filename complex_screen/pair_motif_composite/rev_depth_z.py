# NFY depth check (a)-(c). (a) Reversed-design z (as rev_enrich.py) recomputed on RAW reads with cells subsampled per TF to a common count
# (N20 = 20th-deepest TF's cells, N3k = 3,000; 3 seeds, z averaged): CPM minus 79-TF mean, rank-normal, residual vs GC x CpG x promoter
# strata, z on odd (disc) and even + X (val) chromosomes. Motifs reported: NFY, NRF1, CTCF (H12CORE). (b) For every TF, its NFY z against
# the 10 TFs nearest in cell count: depth-matched z = (z - mean_nb) / sd_nb, and rank among the 11, for imputed and raw z.
# (c) The ten NFY hits of the imputed reversed design: survive = raw strict at both depths (z_disc >= 4, z_val >= 2) AND imputed
# depth-matched z >= 2 on disc and val. Output reversed/nfy_depth_matched.tsv, reversed/depth/raw_z_all.tsv
import os, gzip, glob
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import norm, rankdata
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; R = f"{O}/reversed"; DP = f"{R}/depth"
MOT = {"NFY": "NFYA.H12CORE.0.P.B", "NRF1": "NRF1.H12CORE.0.PS.A", "CTCF": "CTCF.H12CORE.0.P.B"}
HITS = ["RBBP4", "SOX4", "ZNF507", "E2F3", "NME2", "MLX", "GLYR1", "HOXA10", "HOXB9", "FLYWCH1"]
Q = pd.read_csv(f"{O}/ref/bins_qc.tsv", sep="\t"); num = Q.chrom.str.replace("chr", "")
part = np.where(num.isin([str(i) for i in range(1, 23, 2)]), "disc", np.where(num.isin([str(i) for i in range(2, 23, 2)] + ["X"]), "val", "none"))
ok = Q.n_ok.values & ~Q.blacklist.values & (Q.exonic.values <= 0.5) & (part != "none"); ei = np.flatnonzero(ok)
prom = np.zeros(len(Q), bool); prom[[int(x) for x in open(f"{O}/ref/promoter_bins.txt").read().split()]] = True
cq = np.quantile(Q.cpg_oe.values[ei], [0.2, 0.4, 0.6, 0.8]); strat = (np.floor(Q.gc.values / 0.025).astype(int) * 10 + np.searchsorted(cq, Q.cpg_oe.values)) * 2 + prom
pos = {f"b{b}": i for i, b in enumerate(Q.bin.values[ei])}; rows, cols = [], []
for f in sorted(glob.glob(f"{R}/fimo/c*.tsv.gz")):
    for l in gzip.open(f, "rt"):
        m, s = l.rstrip("\n").split("\t")
        if s in pos: rows.append(pos[s]); cols.append(m)
mid = {m: k for k, m in enumerate(sorted(set(cols)))}; motifs = sorted(mid, key=mid.get)
H = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, [mid[c] for c in cols])), shape=(len(ei), len(motifs))); H.data[:] = 1
st = strat[ei]; pe = part[ei]
def zmat(D):  # D: TFs x all bins -> {part: DataFrame motifs x TFs}
    Y = np.apply_along_axis(lambda v: norm.ppf((rankdata(v) - 0.5) / len(v)), 1, D[:, ei]).T; out = {}
    for pt in ["disc", "val"]:
        sel = pe == pt; Ys = Y[sel].copy(); ss = st[sel]
        for s in np.unique(ss): q = ss == s; Ys[q] -= Ys[q].mean(0)
        Hs = H[sel]; n = np.asarray(Hs.sum(0)).ravel(); sd = Ys.std(0); z = (Hs.T @ Ys) / np.maximum(n, 1)[:, None] * np.sqrt(np.maximum(n, 1))[:, None] / np.maximum(sd, 1e-9)[None, :]
        z[n < 50] = np.nan; out[pt] = z
    return out
cells = pd.read_csv(f"{X}/complex_screen/pair_motif_lookup/tf_cells.tsv", sep="\t").set_index("tf").n_cells
tfs = list(cells.index); info = pd.concat([pd.read_csv(f"{DP}/counts/{t}.info.tsv", sep="\t") for t in tfs]).set_index("tf")
ZR = {}
for lv in ["N20", "N3k"]:
    acc = {pt: [] for pt in ["disc", "val"]}
    for sd in range(3):
        C = np.stack([np.load(f"{DP}/counts/{t}.npz")[lv][sd] for t in tfs]); C = C / C.sum(1, keepdims=True) * 1e6; z = zmat(C - C.mean(0))
        for pt in acc: acc[pt].append(z[pt])
        print(lv, sd, "done", flush=True)
    for pt in acc: ZR[(lv, pt)] = np.nanmean(np.stack(acc[pt]), 0); ZR[(lv, pt, "min")] = np.nanmin(np.stack(acc[pt]), 0)
mi = {m: k for k, m in enumerate(motifs)}; long = []
for nm, m in MOT.items():
    for j, t in enumerate(tfs):
        r = dict(motif=nm, tf=t)
        for lv in ["N20", "N3k"]:
            for pt in ["disc", "val"]: r[f"raw_{lv}_z_{pt}"] = round(float(ZR[(lv, pt)][mi[m], j]), 2); r[f"raw_{lv}_z_{pt}_minseed"] = round(float(ZR[(lv, pt, "min")][mi[m], j]), 2)
            r[f"raw_{lv}_rank_disc"] = int((ZR[(lv, "disc")][:, j] > ZR[(lv, "disc")][mi[m], j]).sum() + 1)
        long.append(r)
L = pd.DataFrame(long); L.to_csv(f"{DP}/raw_z_all.tsv", sep="\t", index=False)
Zd = pd.read_csv(f"{R}/z_disc.tsv", sep="\t", index_col=0); Zv = pd.read_csv(f"{R}/z_val.tsv", sep="\t", index_col=0)
out = []
for nm, m in MOT.items():
    Lm = L[L.motif == nm].set_index("tf")
    for t in tfs:
        nb = (cells.drop(t) - cells[t]).abs().sort_values().index[:10]
        r = dict(motif=nm, tf=t, n_cells=int(cells[t]), depth_rank=int((cells > cells[t]).sum() + 1), imp_z_disc=Zd.loc[m, t], imp_z_val=Zv.loc[m, t])
        for lab, get in [("imp", lambda u, pt: (Zd if pt == "disc" else Zv).loc[m, u]), ("rawN20", lambda u, pt: Lm.loc[u, f"raw_N20_z_{pt}"]), ("rawN3k", lambda u, pt: Lm.loc[u, f"raw_N3k_z_{pt}"])]:
            for pt in ["disc", "val"]:
                v = get(t, pt); w = np.array([get(u, pt) for u in nb], float)
                r[f"{lab}_dm_z_{pt}"] = round((v - w.mean()) / w.std(ddof=1), 2); r[f"{lab}_dm_rank_{pt}"] = int((w > v).sum() + 1)
        for c in Lm.columns:
            if c.startswith("raw_") and "minseed" not in c: r[c] = Lm.loc[t, c]
        r["N20_cells_used"] = int(info.loc[t, "N20_cells"]); r["N3k_cells_used"] = int(info.loc[t, "N3k_cells"]); r["neighbours"] = ",".join(nb)
        r["imp_strict"] = bool(r["imp_z_disc"] >= 4 and r["imp_z_val"] >= 2)
        r["raw_strict_both"] = bool(r["raw_N20_z_disc"] >= 4 and r["raw_N20_z_val"] >= 2 and r["raw_N3k_z_disc"] >= 4 and r["raw_N3k_z_val"] >= 2)
        r["raw_loose_both"] = bool(r["raw_N20_z_disc"] >= 3 and r["raw_N20_z_val"] >= 1.5 and r["raw_N3k_z_disc"] >= 3 and r["raw_N3k_z_val"] >= 1.5)
        r["imp_dm_pass"] = bool(r["imp_dm_z_disc"] >= 2 and r["imp_dm_z_val"] >= 2)
        r["raw_dm_pass"] = bool(min(r["rawN20_dm_z_disc"], r["rawN20_dm_z_val"], r["rawN3k_dm_z_disc"], r["rawN3k_dm_z_val"]) >= 2)
        r["nfy_hit"] = t in HITS; r["survives_both"] = r["raw_strict_both"] and r["imp_dm_pass"]; out.append(r)
T = pd.DataFrame(out); T.to_csv(f"{R}/nfy_depth_matched.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
k = ["tf", "n_cells", "depth_rank", "imp_z_disc", "imp_z_val", "imp_dm_z_disc", "imp_dm_z_val", "raw_N20_z_disc", "raw_N20_z_val", "raw_N3k_z_disc", "raw_N3k_z_val", "rawN20_dm_z_disc", "rawN20_dm_z_val", "raw_strict_both", "imp_dm_pass", "raw_dm_pass", "survives_both"]
for nm in MOT:
    s = T[T.motif == nm]; print("\n==", nm); print(s[s.nfy_hit | (s.tf == nm.replace("NFY", "NFYA")) | s.imp_strict | s.raw_strict_both][k].to_string(index=False))
    print(nm, "TFs imp_strict:", int(s.imp_strict.sum()), "raw_strict_both:", int(s.raw_strict_both.sum()), "imp_dm_pass:", int(s.imp_dm_pass.sum()), "raw_dm_pass:", int(s.raw_dm_pass.sum()))
from scipy.stats import spearmanr
s = T[T.motif == "NFY"]
for c in ["imp_z_disc", "raw_N20_z_disc", "raw_N3k_z_disc"]: print("spearman cells vs", c, spearmanr(s.n_cells, s[c]))
