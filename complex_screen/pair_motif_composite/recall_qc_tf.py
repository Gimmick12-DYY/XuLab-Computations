# Peak re-calling QC, one TF, all variants (2026-11-11; criteria fixed before results, see recall_qc_summary.py).
# QC set = top 1,000 sites per TF and variant with NO HOT / ATAC filtering: raw = top 1,000 summits by -log10 p (+/- 100 bp, autosomes + X,
# blacklist removed); imputed = top 1,000 union-ATAC bins by score (bin centre +/- 100 bp). Per set: fraction in union ATAC, in HOT bins,
# within 1 kb of a TSS, mean ATAC CPM, bulk overlap (CTCF HEK293T ChIP-nexus GSE103651; ZNF282 HEK293T ChIP-exo GSM2466515; CTCF HEK293
# ENCODE, labelled HEK293), the 1 kb bin ids (for cross-TF Jaccard), n significant peaks (raw p <= 1e-5; imp_binom q < 0.05) and FRiP
# (raw: fraction of the TF's reads in its p <= 1e-5 peaks). Output recall/qc/<TF>.json
import os, sys, json, subprocess
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; R = f"{O}/recall"; t = sys.argv[1]; N = 1000; os.makedirs(f"{R}/qc/tmp", exist_ok=True)
FR = "/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment"; BL = f"{X}/downstream/cache/hg38-blacklist.v2.bed.gz"
BULK = {"CTCF_HEK293T": f"{X}/data/CTCF_HEK293T_GSE103651_peaks.bed", "ZNF282_HEK293T": f"{X}/data/geo_znf282/ZNF282_peaks_hg38.bed", "CTCF_HEK293": f"{X}/data/CTCF_HEK293_peaks.bed"}
CH = {f"chr{i}" for i in range(1, 23)} | {"chrX"}
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True, executable="/bin/bash"); assert r.returncode == 0, (c, r.stderr); return r.stdout
F = np.load(f"{R}/ref.npz"); hot, atac, prom = F["hot"], F["atac_cpm"], F["prom"]
B = pd.read_csv(f"{O}/ref/bins1kb.bed", sep="\t", header=None, names=["c", "s", "e", "i"]); Q = pd.read_csv(f"{O}/ref/bins_qc.tsv", sep="\t", usecols=["chrom", "s", "e"])
def ubin(c, pos):
    b = B[B.c == c]; k = np.searchsorted(b.s.values, pos, side="right") - 1; ok = (k >= 0) & (pos < b.e.values[np.maximum(k, 0)]); out = np.full(len(pos), -1); out[ok] = b.i.values[k[ok]]; return out
res = {}
for v in ["raw_lab", "raw_tf", "raw_atac", "imp_loo", "imp_atac", "imp_binom"]:
    r = {}
    if v.startswith("raw"):
        src = {"raw_lab": f"/vast/som/xujie_lab/TAGATG-293T/bigwig/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", "raw_tf": f"{R}/peaks_tf/TF.{t}_peaks.narrowPeak", "raw_atac": f"{R}/peaks_atac/TF.{t}_peaks.narrowPeak"}[v]
        if not os.path.exists(src): continue
        pk = pd.read_csv(src, sep="\t", header=None, usecols=[0, 1, 2, 7, 9], names=["c", "s", "e", "lp", "sm"]); pk = pk[pk.c.isin(CH)]
        sig = pk[pk.lp >= 5]; r["n_sig"] = int(len(sig)); f = f"{R}/qc/tmp/{t}_{v}_sig.bed"; sig[["c", "s", "e"]].sort_values(["c", "s"]).to_csv(f, sep="\t", header=False, index=False)
        nr = int(sh(f"zcat {FR}/TF.{t}.bed.gz | wc -l")); ni = int(sh(f"module load bedtools >/dev/null 2>&1; zcat {FR}/TF.{t}.bed.gz | cut -f1-3 | bedtools intersect -u -a - -b {f} | wc -l")) if len(sig) else 0; r["frip"] = round(ni / nr, 4)
        pk["m"] = pk.s + pk.sm; pk = pk.sort_values("lp", ascending=False).drop_duplicates(["c", "m"])
        w = pd.DataFrame(dict(c=pk.c, s=pk.m - 100, e=pk.m + 100, n=np.arange(len(pk)))).head(3 * N); wf = f"{R}/qc/tmp/{t}_{v}_w.bed"; w.to_csv(wf, sep="\t", header=False, index=False)
        bad = set(int(x) for x in sh(f"module load bedtools >/dev/null 2>&1; bedtools intersect -u -a {wf} -b {BL} | cut -f4").split()); w = w[~w.n.isin(bad)].head(N); pos = (w.s + 100).values; bi = np.full(len(w), -1)
        for c in w.c.unique(): q = (w.c == c).values; bi[q] = ubin(c, pos[q])
        gbin = [f"{c}:{p // 1000}" for c, p in zip(w.c, pos)]
    else:
        fz = f"{R}/imputed_scores_binom.npz" if v == "imp_binom" else f"{R}/imputed_scores.npz"
        if not os.path.exists(fz): continue
        z = np.load(fz); key = v.split("_")[1]; s = z[key][list(z["tfs"]).index(t)]; cand = np.flatnonzero(np.isin(Q.chrom.values, list(CH))); top = cand[np.argsort(-s[cand], kind="stable")[:N]]
        w = pd.DataFrame(dict(c=Q.chrom.values[top], s=Q.s.values[top], e=Q.e.values[top], n=np.arange(len(top)))); bi = top; gbin = [f"{c}:{(s_ + 100) // 1000}" for c, s_ in zip(w.c, w.s)]
        if v == "imp_binom":
            pt = pd.read_csv(f"{R}/imputed_binom_peaks.tsv", sep="\t").set_index("tf"); r["n_sig"] = int(pt.peaks_q05[t]); r["n_sig_in_atac"] = int(pt.in_atac[t])
        wf = f"{R}/qc/tmp/{t}_{v}_w.bed"; w.to_csv(wf, sep="\t", header=False, index=False)
    inb = bi >= 0; r["n_top"] = int(len(w)); r["frac_in_atac"] = round(inb.mean(), 3); r["frac_hot"] = round(hot[bi[inb]].sum() / len(w), 3); r["mean_atac_cpm"] = round(float(np.where(inb, atac[np.maximum(bi, 0)], 0).mean()), 3)
    r["frac_tss1kb"] = round(len(set(sh(f"module load bedtools >/dev/null 2>&1; bedtools window -u -w 1000 -a {wf} -b {O}/ref/tss.bed | cut -f4").split())) / len(w), 3)
    for k, f in BULK.items(): r[f"bulk_{k}"] = round(len(set(sh(f"module load bedtools >/dev/null 2>&1; cut -f1-3 {f} | bedtools intersect -u -a {wf} -b - | cut -f4").split())) / len(w), 3)
    r["bins"] = gbin; res[v] = r
json.dump(res, open(f"{R}/qc/{t}.json", "w")); print(t, {v: {k: x for k, x in r.items() if k != "bins"} for v, r in res.items()})
