# Threshold-free motif profile, step 1 (peak-first pools). Per TF: raw MACS2 summits (lab peaks, read-only), p <= 1e-5, ranked by
# -log10 p, top 2,000 kept after filters (as raw_pools.py: summit +/- 100 bp inside a union-ATAC 1 kb bin, no N, not blacklisted, <= 50% exonic, autosomes + chrX);
# the first 500 of them form the N = 500 set. Background per N: eligible union-ATAC bin centres not overlapping any of the TF's p <= 1e-5
# peaks, matched per stratum (GC 0.025 x CpG o/e quintile x promoter), up to 3 per foreground window. All chromosomes pooled (no cutoff,
# so no held-out split). Output profile/pools/<TF>.tsv (window, n500, n2000 flags), profile/pools/<TF>_bg{500,2000}.txt (bin ids),
# profile/windows.fa (all foreground windows, ids w<TF>_<k>), profile/chunks/ (known DB + de novo family representatives, 100 per chunk).
import os, subprocess
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; G = f"{O}/profile"; MAC = "/vast/som/xujie_lab/TAGATG-293T/bigwig/macs2"
NS = [500, 2000]; rng = np.random.default_rng(20261110)
for d in ["pools", "chunks", "fimo", "logs"]: os.makedirs(f"{G}/{d}", exist_ok=True)
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True, executable="/bin/bash"); assert r.returncode == 0, (c, r.stderr); return r.stdout
Q = pd.read_csv(f"{O}/ref/bins_qc.tsv", sep="\t"); CH = {f"chr{i}" for i in range(1, 23)} | {"chrX"}
elig = Q.n_ok.values & ~Q.blacklist.values & (Q.exonic.values <= 0.5) & Q.chrom.isin(CH).values; ei = np.flatnonzero(elig)
prom = np.zeros(len(Q), bool); prom[[int(x) for x in open(f"{O}/ref/promoter_bins.txt").read().split()]] = True
cq = np.quantile(Q.cpg_oe.values[ei], [0.2, 0.4, 0.6, 0.8]); strat_bg = (np.floor(Q.gc.values / 0.025).astype(int) * 10 + np.searchsorted(cq, Q.cpg_oe.values)) * 2 + prom
os.makedirs(f"{G}/tmp", exist_ok=True); tfs = sorted(d.split(".", 1)[1] for d in os.listdir(MAC) if d.startswith("TF.")); fa = open(f"{G}/windows.fa", "w"); rows = []
for t in tfs:
    f = f"{MAC}/TF.{t}/TF.{t}_peaks.narrowPeak"
    if not os.path.exists(f): continue
    pk = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2, 7, 9]); pk.columns = ["c", "s", "e", "lp", "summit"]; pk = pk[pk.lp >= 5]
    allp = f"{G}/tmp/{t}_allpeaks.bed"; pk[["c", "s", "e"]].to_csv(allp, sep="\t", header=False, index=False)
    pk = pk[pk.c.isin(CH)].sort_values("lp", ascending=False); pk["m"] = pk.s + pk.summit
    w = pd.DataFrame(dict(c=pk.c, s=pk.m - 100, e=pk.m + 100, n=[f"w{t}_{i}" for i in range(len(pk))], lp=pk.lp)); w = w[w.s > 0]; wf = f"{G}/tmp/{t}_win.bed"
    w[["c", "s", "e", "n"]].to_csv(wf, sep="\t", header=False, index=False)
    bad = set(sh(f"module load bedtools >/dev/null 2>&1; bedtools intersect -u -a {wf} -b {X}/downstream/cache/hg38-blacklist.v2.bed.gz | cut -f4").split())
    bad |= set(sh(f"module load bedtools >/dev/null 2>&1; sort -k1,1 -k2,2n {wf} | bedtools coverage -a - -b {O}/ref/exons.bed | awk '$8>0.5' | cut -f4").split())
    tss = set(sh(f"module load bedtools >/dev/null 2>&1; bedtools window -u -w 1000 -a {wf} -b {O}/ref/tss.bed | cut -f4").split())
    seq = dict(l.split("\t") for l in sh(f"module load bedtools >/dev/null 2>&1; bedtools getfasta -nameOnly -tab -fi {X}/downstream/cache/hg38.fa -bed {wf}").splitlines())
    inatac = set(sh(f"module load bedtools >/dev/null 2>&1; bedtools intersect -u -a {wf} -b {O}/ref/bins1kb.bed | cut -f4").split())
    w = w[~w.n.isin(bad) & w.n.isin(inatac)].copy(); w["q"] = [seq.get(n, "N").upper() for n in w.n]; w = w[~w.q.str.contains("N")].head(max(NS)).copy()
    gc = np.array([(q.count("G") + q.count("C")) / 200 for q in w.q]); cpg = np.array([q.count("CG") * 200 / max(q.count("C") * q.count("G"), 1) for q in w.q])
    w["st"] = (np.floor(gc / 0.025).astype(int) * 10 + np.searchsorted(cq, cpg)) * 2 + w.n.isin(tss).values; w["rank"] = np.arange(len(w))
    hit = set(int(x) for x in sh(f"module load bedtools >/dev/null 2>&1; bedtools intersect -u -a {O}/ref/bins1kb.bed -b {allp} | cut -f4").split())
    free = np.array([i for i in ei if i not in hit]); r = dict(tf=t, n_in_atac=len(inatac), n_win=len(w), gc_fg=round(gc.mean(), 3) if len(w) else np.nan)
    for N in NS:
        ww = w.head(N); bg = []
        for s in np.unique(ww.st):
            c = free[strat_bg[free] == s]; m = min(3 * int((ww.st == s).sum()), len(c))
            if m: bg.append(rng.choice(c, m, replace=False))
        bg = np.concatenate(bg) if bg else np.array([], int); open(f"{G}/pools/{t}_bg{N}.txt", "w").write("\n".join(f"b{Q.bin[i]}" for i in bg) + "\n")
        r[f"bg{N}"] = len(bg); r[f"gc_bg{N}"] = round(Q.gc.values[bg].mean(), 3) if len(bg) else np.nan; r[f"unmatched_fg{N}"] = int(sum((ww.st == s).sum() for s in np.unique(ww.st) if not (strat_bg[free] == s).any()))
    w[["n", "c", "s", "e", "lp", "st", "rank"]].to_csv(f"{G}/pools/{t}.tsv", sep="\t", index=False)
    for n, q in zip(w.n, w.q): fa.write(f">{n}\n{q}\n")
    rows.append(r); print(r, flush=True)
fa.close(); pd.DataFrame(rows).to_csv(f"{G}/pool_sizes.tsv", sep="\t", index=False)
txt = open(f"{O}/db/known.meme").read(); head, *bl = txt.split("\nMOTIF "); bl = ["MOTIF " + b.rstrip() for b in bl]
rp = open(f"{O}/per_tf_raw/denovo_cmp/reps.meme").read().split("\nMOTIF ")[1:]; bl += ["MOTIF " + b.rstrip() for b in rp]
for k in range(0, len(bl), 100): open(f"{G}/chunks/c{k // 100:02d}.meme", "w").write(head.rstrip() + "\n\n" + "\n\n".join(bl[k:k + 100]) + "\n")
print(len(bl), "motifs (known + de novo family reps);", (len(bl) + 99) // 100, "chunks")
