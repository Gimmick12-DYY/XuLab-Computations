# m20 Part 1: compare the independent recount with the m18 features and m11 lists; build/coordinate checks; motif label check from m18 columns. Diagnostic.
import gzip, os, subprocess
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m20_raw_binding_audit"; W = f"{O}/work"; OUT = f"{O}/out"; V = "/vast/som/xujie_lab/TAGATG-293T/bigwig"
AUD = ["CTCF", "NRF1", "NFYA", "MAZ", "ZNF143", "E4F1"]; LAB = "diagnostic"
tfs79 = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]
IX = np.load(f"{RG}/m18_ml_v2/features/index.npz", allow_pickle=True); TT = list(IX["tfs"].astype(str)); ug = IX["genes"].astype(str); NG = len(ug)
pr = np.load(f"{RG}/m18_ml_v2/features/pair_real.npz", allow_pickle=True); pn = list(pr["names"]); PX = pr["X"].reshape(len(TT), NG, len(pn)); c = {n: k for k, n in enumerate(pn)}
gD = np.load(f"{RG}/m18_ml_v2/features/gene_D_real.npz", allow_pickle=True); Dn = list(gD["names"]); DX = gD["X"]
L11 = pd.read_csv(f"{RG}/m11_peak_regulome/out/long_q0.05.tsv.gz", sep="\t", usecols=["tf", "gene", "any_peak", "promoter_peak"]); L11 = L11[L11.tf.isin(AUD)]
R = []
for t in AUD:
    tot = int(open(f"{W}/recount_{t}_total.txt").read()); P = pd.read_csv(f"{W}/recount_{t}_prom.tsv", sep="\t", header=None, names=["gene", "n"]).groupby("gene").n.max()
    new_cpm = P.reindex(ug).values / tot * 1e6
    row = dict(label=LAB, tf=t, total_reads_recount=tot)
    # m10 counts (read by m18 block B and D)
    c10 = np.load(f"{RG}/m10_promoter_regulome/work/counts/{t}.npz", allow_pickle=True); row["total_reads_m10"] = int(c10["total"])
    if t in TT:
        i = TT.index(t); old = PX[i, :, c["B_prom_reads"]].astype(float); m = np.isfinite(new_cpm)
        row.update(B_prom_reads_pearson=float(np.corrcoef(new_cpm[m], old[m])[0, 1]), B_prom_reads_frac_exact=float(np.mean(np.isclose(new_cpm[m], old[m], rtol=1e-3, atol=1e-3))), n_genes_compared=int(m.sum()))
        anyp_m18 = (PX[i, :, c["B_prom_peak"]] > 0) | (PX[i, :, c["B_distal_abc_peak"]] > 0)
    else: anyp_m18 = None
    dcol = Dn.index(f"D_prom_reads_{t}"); oldD = DX[:, dcol].astype(float); m = np.isfinite(new_cpm)
    row.update(D_prom_reads_pearson=float(np.corrcoef(new_cpm[m], oldD[m])[0, 1]), D_frac_exact=float(np.mean(np.isclose(new_cpm[m], oldD[m], rtol=1e-3, atol=1e-3))))
    # best-matching D column (label shift check)
    cors = [np.corrcoef(new_cpm[m], DX[m, Dn.index(f"D_prom_reads_{u}")].astype(float))[0, 1] for u in tfs79]; row["D_best_matching_column"] = tfs79[int(np.nanargmax(cors))]
    newg = set(open(f"{W}/recount_{t}_peakgenes.txt").read().split()); newv = np.isin(ug, list(newg))
    l11 = L11[L11.tf == t].set_index("gene").any_peak.reindex(ug).fillna(False).values.astype(bool)
    row.update(peakgenes_recount=int(newv.sum()), peakgenes_m11=int(l11.sum()), agree_m11=float((newv == l11).mean()), recount_only_vs_m11=int((newv & ~l11).sum()), m11_only=int((~newv & l11).sum()))
    if anyp_m18 is not None: row.update(peakgenes_m18=int(anyp_m18.sum()), agree_m18=float((newv == anyp_m18).mean()), recount_only_vs_m18=int((newv & ~anyp_m18).sum()), m18_only=int((~newv & anyp_m18).sum()))
    R.append(row)
R = pd.DataFrame(R); R.to_csv(f"{OUT}/part1c_recount_vs_features.tsv", sep="\t", index=False, float_format="%.5f"); print(R.drop(columns="label").T.to_string())
# 1b build / coordinate checks
sizes = pd.read_csv(f"{RG}/m2_abc/inputs/chrom_sizes.tsv", sep="\t", header=None, index_col=0)[1]
B = []
def chk(name, path, cols, nmax=None):
    d = pd.read_csv(path, sep="\t", header=None, usecols=cols, nrows=nmax, comment="#"); d.columns = ["c", "s", "e"]
    bad_chr = (~d.c.astype(str).isin(sizes.index)).mean(); over = (d.e > d.c.map(sizes).fillna(np.inf)).sum(); zero = (d.s == 0).sum()
    chr1max = d[d.c == "chr1"].e.max(); B.append(dict(label=LAB, file=name, n=len(d), frac_chr_not_in_hg38_sizes=float(bad_chr), n_end_beyond_hg38_chr_end=int(over), chr1_max_end=int(chr1max) if np.isfinite(chr1max) else -1, hg38_chr1=int(sizes["chr1"])))
for t in AUD:
    chk(f"fragment TF.{t}", f"{V}/fragment/TF.{t}.bed.gz", [0, 1, 2]); chk(f"narrowPeak TF.{t}", f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", [0, 1, 2])
chk("EnhancerList (ABC)", f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/EnhancerList.bed", [0, 1, 2])
for f in ["CTCF_HEK293T_GSE103651_peaks.bed", "CTCF_HEK293_peaks.bed", "MAZ_HEK293_peaks.bed", "NFYA_K562_peaks.bed"]: chk(f"ChIP {f}", f"{X}/data/{f}", [0, 1, 2])
B = pd.DataFrame(B); B.to_csv(f"{OUT}/part1b_build_checks.tsv", sep="\t", index=False); print(B.drop(columns="label").to_string(index=False))
# 1d motif label check from m18 columns only
M = []
def orr(a, b): a11 = (a & b).sum(); a10 = (a & ~b).sum(); a01 = (~a & b).sum(); a00 = (~a & ~b).sum(); return (a11 + .5) * (a00 + .5) / ((a10 + .5) * (a01 + .5))
for t in AUD:
    if t not in TT: continue
    i = TT.index(t)
    if not PX[i, 0, c["C_has_motif"]]: continue
    mot = PX[i, :, c["C_prom_sites"]] > 0; ors = {}
    for u in tfs79:
        v = DX[:, Dn.index(f"D_prom_reads_{u}")].astype(float); top = np.zeros(NG, bool); top[np.argsort(-v, kind="stable")[:1000]] = True; ors[u] = orr(top, mot)
    s = pd.Series(ors).sort_values(ascending=False); orsB = {}
    for j, u in enumerate(TT): orsB[u] = orr(PX[j, :, c["B_prom_peak"]] > 0, mot)
    sB = pd.Series(orsB).sort_values(ascending=False)
    M.append(dict(label=LAB, motif_of=t, n_genes_with_motif=int(mot.sum()), OR_own_column_D=float(ors[t]), rank_own_among_79_D=int(list(s.index).index(t) + 1), top3_D=",".join(f"{k}:{v:.2f}" for k, v in s.head(3).items()),
                  OR_own_row_B_prom_peak=float(orsB[t]), rank_own_among_TFs_B=int(list(sB.index).index(t) + 1), n_tfs_B=len(sB), top3_B=",".join(f"{k}:{v:.2f}" for k, v in sB.head(3).items())))
M = pd.DataFrame(M); M.to_csv(f"{OUT}/part1d_motif_label_check.tsv", sep="\t", index=False, float_format="%.4f"); print(M.drop(columns="label").to_string(index=False))
