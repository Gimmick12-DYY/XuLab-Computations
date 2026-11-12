# m20 Part 2: bulk ChIP overlap of our HITAG data, with every-other-TF, ATAC-only and promoter/distal controls. Diagnostic.
import os, subprocess
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m20_raw_binding_audit"; W = f"{O}/work"; OUT = f"{O}/out"; V = "/vast/som/xujie_lab/TAGATG-293T/bigwig"; LAB = "diagnostic"
BT = "/nas/longleaf/rhel9/apps/bedtools/2.31.1/bedtools2/bin/bedtools"; QMIN = -np.log10(0.05); rng = np.random.default_rng(20261112)
tfs79 = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]
CHIP = {"CTCF|HEK293T (GSE103651, ChIP-nexus)": ("CTCF", "CTCF_HEK293T_GSE103651_peaks.bed"), "CTCF|HEK293 (ENCODE ENCSR000DTW)": ("CTCF", "CTCF_HEK293_peaks.bed"), "MAZ|HEK293": ("MAZ", "MAZ_HEK293_peaks.bed"),
        "NFYA|K562": ("NFYA", "NFYA_K562_peaks.bed"), "SETDB1|HEK293": ("SETDB1", "SETDB1_HEK293_peaks.bed"), "ZBTB7A|HEK293": ("ZBTB7A", "ZBTB7A_HEK293_peaks.bed"), "ZIC2|HEK293": ("ZIC2", "ZIC2_HEK293_peaks.bed"),
        "ZNF282|HEK293": ("ZNF282", "ZNF282_HEK293_peaks.bed"), "ZNF777|HEK293": ("ZNF777", "ZNF777_HEK293_peaks.bed")}
sizes = pd.read_csv(f"{RG}/m2_abc/inputs/chrom_sizes.tsv", sep="\t", header=None, index_col=0)[1]; MAIN = [f"chr{i}" for i in list(range(1, 23)) + ["X"]]
U = pd.read_csv(f"{X}/data/HEK293T_ATAC_union3.bed.gz", sep="\t", header=None, usecols=[0, 1, 2], names=["c", "s", "e"]); U = U[U.c.isin(MAIN)].sort_values(["c", "s"]).reset_index(drop=True)
# random genome windows (earlier-style negatives), width = median union width, not overlapping any ChIP set (excluded per set below)
wdt = int((U.e - U.s).median()); nR = 60000; gc = rng.choice(MAIN, nR, p=(sizes[MAIN] / sizes[MAIN].sum()).values); gs = (rng.random(nR) * (sizes[gc].values - wdt)).astype(int)
RND = pd.DataFrame(dict(c=gc, s=gs, e=gs + wdt)).sort_values(["c", "s"]).reset_index(drop=True)
GL = pd.read_csv(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/GeneList.txt", sep="\t", usecols=["chr", "tss"]).dropna(); TSS = {c: np.sort(g.tss.values.astype(int)) for c, g in GL.groupby("chr")}
def near_tss(c, mid):
    out = np.zeros(len(mid), bool)
    for ch in np.unique(c):
        t = TSS.get(ch); m = c == ch
        if t is None: continue
        k = np.searchsorted(t, mid[m]); d = np.minimum(np.abs(mid[m] - t[np.clip(k - 1, 0, len(t) - 1)]), np.abs(mid[m] - t[np.clip(k, 0, len(t) - 1)])); out[m] = d <= 1000
    return out
def overlap(A, B):  # bool per row of A: overlaps any interval in B (both DataFrames c,s,e)
    out = np.zeros(len(A), bool)
    for ch, G in A.groupby("c"):
        H = B[B.c == ch]
        if not len(H): continue
        o = np.argsort(H.s.values); hs, he = H.s.values[o], np.maximum.accumulate(H.e.values[o]); k = np.searchsorted(hs, G.e.values, side="left") - 1; ok = k >= 0; ok[ok] = he[k[ok]] > G.s.values[ok]; out[G.index.values] = ok
    return out
def count_in(R, pos_c, pos):  # reads (5' ends) per region of R
    out = np.zeros(len(R))
    for ch, G in R.groupby("c"):
        p = np.sort(pos[pos_c == ch]); out[G.index.values] = np.searchsorted(p, G.e.values) - np.searchsorted(p, G.s.values)
    return out
CU = np.zeros((len(U), 79)); CR = np.zeros((len(RND), 79)); SUM = {}
for j, t in enumerate(tfs79):
    F = pd.read_csv(f"{V}/fragment/TF.{t}.bed.gz", sep="\t", header=None, usecols=[0, 1, 2, 5], names=["c", "s", "e", "st"]); p = np.where(F.st.values == "+", F.s.values, F.e.values - 1); c = F.c.values.astype(str)
    CU[:, j] = count_in(U, c, p); CR[:, j] = count_in(RND, c, p)
    P = pd.read_csv(f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 8, 9], names=["c", "s", "q", "off"]); P = P[P.q >= QMIN]; sm = (P.s + P.off).values
    SUM[t] = pd.DataFrame(dict(c=P.c.values.astype(str), s=sm, e=sm + 1)).reset_index(drop=True)
print("counted", flush=True)
# ATAC reads per region (OmniATAC tagAlign)
def atac_counts(R, name):  # bedtools -sorted needs the tagAlign chromosome order (chr1, chr2, ...): write in that order, map back
    o = R.assign(k=R.c.map({c: i for i, c in enumerate(MAIN)})).sort_values(["k", "s"]); o[["c", "s", "e"]].to_csv(f"{W}/{name}.bed", sep="\t", header=False, index=False)
    out = subprocess.run(f"{BT} intersect -a {W}/{name}.bed -b {RG}/m2_abc/inputs/omniatac_merged.tn5.tagAlign.gz -c -sorted -g {W}/genome.txt", shell=True, capture_output=True, text=True, check=True).stdout
    v = np.array([int(l.split("\t")[-1]) for l in out.strip().splitlines()]); res = np.zeros(len(R)); res[o.index.values] = v; return res
sizes[MAIN].to_csv(f"{W}/genome.txt", sep="\t", header=False)
try: AU = atac_counts(U, "union"); AR = atac_counts(RND, "random")
except Exception as e: print("ATAC sorted intersect failed, falling back", e, flush=True); AU = AR = None
prom = near_tss(U.c.values, ((U.s + U.e) // 2).values)
ROWS = []; RANK = []
for key, (tf, f) in CHIP.items():
    C = pd.read_csv(f"{X}/data/{f}", sep="\t", header=None, usecols=[0, 1, 2], names=["c", "s", "e"]); C = C[C.c.isin(MAIN)].reset_index(drop=True)
    lab = overlap(U, C); rlab = overlap(RND, C); per_tf = []
    for j, t in enumerate(tfs79):
        S = SUM[t]; S = S[S.c.isin(MAIN)].reset_index(drop=True); prec = float(overlap(S, C).mean()) if len(S) else np.nan
        rec = float(overlap(C, S).mean()) if len(S) else np.nan
        au = roc_auc_score(lab, CU[:, j]); aup = roc_auc_score(lab[prom], CU[prom, j]); aud = roc_auc_score(lab[~prom], CU[~prom, j])
        per_tf.append(dict(tf=t, own=t == tf, n_summits=len(S), precision_summits_in_chip=prec, recall_chip_with_summit=rec, auroc_union_atac=au, auroc_promoter=aup, auroc_distal=aud))
    D = pd.DataFrame(per_tf)
    # earlier-style AUROC: union-ATAC regions overlapping ChIP (positives) vs random genome windows not overlapping the ChIP set (negatives)
    neg = ~rlab; yy = np.r_[np.ones(int(lab.sum())), np.zeros(int(neg.sum()))]
    D["auroc_chip_regions_vs_random_genome"] = [roc_auc_score(yy, np.r_[CU[lab, j], CR[neg, j]]) for j in range(79)]
    D["rank_auroc_vs_random"] = D.auroc_chip_regions_vs_random_genome.rank(ascending=False)
    D["rank_auroc"] = D.auroc_union_atac.rank(ascending=False); D["rank_precision"] = D.precision_summits_in_chip.rank(ascending=False); D["rank_recall"] = D.recall_chip_with_summit.rank(ascending=False)
    D.insert(0, "chip", key); D.insert(0, "label", LAB); RANK.append(D)
    own = D[D.own].iloc[0]; oth = D[~D.own]
    r = dict(label=LAB, chip=key, n_chip_peaks=len(C), frac_union_atac_regions_in_chip=float(lab.mean()), frac_chip_peaks_overlapping_atac=float(overlap(C, U).mean()),
             own_tf=tf, own_n_summits=int(own.n_summits), own_precision=own.precision_summits_in_chip, others_precision_median=float(oth.precision_summits_in_chip.median()), own_rank_precision=int(own.rank_precision),
             own_recall=own.recall_chip_with_summit, others_recall_median=float(oth.recall_chip_with_summit.median()), own_rank_recall=int(own.rank_recall),
             own_auroc_vs_random_genome=own.auroc_chip_regions_vs_random_genome, others_auroc_vs_random_median=float(oth.auroc_chip_regions_vs_random_genome.median()), own_rank_auroc_vs_random=int(own.rank_auroc_vs_random),
             own_auroc=own.auroc_union_atac, others_auroc_median=float(oth.auroc_union_atac.median()), others_auroc_max=float(oth.auroc_union_atac.max()), own_rank_auroc=int(own.rank_auroc),
             own_auroc_promoter=own.auroc_promoter, own_auroc_distal=own.auroc_distal, others_auroc_promoter_median=float(oth.auroc_promoter.median()), others_auroc_distal_median=float(oth.auroc_distal.median()))
    if AU is not None:
        r.update(atac_auroc_vs_random_genome=roc_auc_score(yy, np.r_[AU[lab], AR[neg]]), atac_auroc_union=roc_auc_score(lab, AU), atac_auroc_promoter=roc_auc_score(lab[prom], AU[prom]), atac_auroc_distal=roc_auc_score(lab[~prom], AU[~prom]))
    ROWS.append(r); print(key, "done", flush=True)
R = pd.DataFrame(ROWS); R.to_csv(f"{OUT}/part2_bulk_overlap.tsv", sep="\t", index=False, float_format="%.4f"); pd.concat(RANK).to_csv(f"{OUT}/part2_all_tfs_vs_each_chip.tsv", sep="\t", index=False, float_format="%.4f")
print(R.drop(columns="label").T.to_string())
