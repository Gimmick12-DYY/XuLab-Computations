# Fallback (ii) for Track B: per-TF foreground from the lab's raw MACS2 summits (read-only, /vast/som/xujie_lab/TAGATG-293T/bigwig/macs2),
# p <= 1e-5, summit +/- 100 bp, kept only when the window overlaps a union-ATAC 1 kb bin (2026-11-11, raw masked to open chromatin like the imputed data);
# drop windows with N, overlapping the ENCODE blacklist, or >50% exonic; then the top 5,000 by -log10 p.
# Background: union-ATAC bin centres (200 bp, eligible as in build_pools.py) not overlapping any of the TF's p <= 1e-5 peaks, matched per
# stratum (GC 0.025 x CpG o/e quintile x promoter = a GENCODE TSS within 1 kb), equal counts. Odd autosomes = disc, even + chrX = val.
# Output: per_tf_raw/pools/<TF>/{disc,val}/{fg,bg}.bed, per_tf_raw/tf_pools.tsv, per_tf_raw/tasks_tf.tsv (known + val_known first, then de novo)
import os, subprocess
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; G = f"{O}/per_tf_raw"; MAC = "/vast/som/xujie_lab/TAGATG-293T/bigwig/macs2"
TOP = 5000; rng = np.random.default_rng(20261110)
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True, executable="/bin/bash"); assert r.returncode == 0, (c, r.stderr); return r.stdout
Q = pd.read_csv(f"{O}/ref/bins_qc.tsv", sep="\t"); num = Q.chrom.str.replace("chr", "")
partf = lambda c: "disc" if c in [f"chr{i}" for i in range(1, 23, 2)] else ("val" if c in [f"chr{i}" for i in range(2, 23, 2)] + ["chrX"] else "none")
part = np.array([partf(c) for c in Q.chrom]); elig = Q.n_ok.values & ~Q.blacklist.values & (Q.exonic.values <= 0.5) & (part != "none"); ei = np.flatnonzero(elig)
prom = np.zeros(len(Q), bool); prom[[int(x) for x in open(f"{O}/ref/promoter_bins.txt").read().split()]] = True
cq = np.quantile(Q.cpg_oe.values[ei], [0.2, 0.4, 0.6, 0.8]); strat_bg = (np.floor(Q.gc.values / 0.025).astype(int) * 10 + np.searchsorted(cq, Q.cpg_oe.values)) * 2 + prom
os.makedirs(f"{G}/tmp", exist_ok=True); tfs = sorted(d.split(".", 1)[1] for d in os.listdir(MAC) if d.startswith("TF."))
rows, tasks, known_t, dn_t = [], [], [], []
for t in tfs:
    f = f"{MAC}/TF.{t}/TF.{t}_peaks.narrowPeak"
    if not os.path.exists(f): continue
    pk = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2, 7, 9]); pk.columns = ["c", "s", "e", "lp", "summit"]; pk = pk[pk.lp >= 5]
    allp = f"{G}/tmp/{t}_allpeaks.bed"; pk[["c", "s", "e"]].to_csv(allp, sep="\t", header=False, index=False)
    pk = pk[pk.c.map(partf) != "none"].sort_values("lp", ascending=False); pk["m"] = pk.s + pk.summit
    w = pd.DataFrame(dict(c=pk.c, s=pk.m - 100, e=pk.m + 100, n=[f"p{i}" for i in range(len(pk))])); w = w[w.s > 0]; wf = f"{G}/tmp/{t}_win.bed"; w.to_csv(wf, sep="\t", header=False, index=False)
    bad = set(sh(f"module load bedtools >/dev/null 2>&1; bedtools intersect -u -a {wf} -b {X}/downstream/cache/hg38-blacklist.v2.bed.gz | cut -f4").split())
    ex = sh(f"module load bedtools >/dev/null 2>&1; sort -k1,1 -k2,2n {wf} | bedtools coverage -a - -b {O}/ref/exons.bed | awk '$8>0.5' | cut -f4"); bad |= set(ex.split())
    tss = set(sh(f"module load bedtools >/dev/null 2>&1; bedtools window -u -w 1000 -a {wf} -b {O}/ref/tss.bed | cut -f4").split())
    seq = dict(l.split("\t") for l in sh(f"module load bedtools >/dev/null 2>&1; bedtools getfasta -nameOnly -tab -fi {X}/downstream/cache/hg38.fa -bed {wf}").splitlines())
    inatac = set(sh(f"module load bedtools >/dev/null 2>&1; bedtools intersect -u -a {wf} -b {O}/ref/bins1kb.bed | cut -f4").split()); n_all = len(w)
    w = w[~w.n.isin(bad) & w.n.isin(inatac)].copy(); w["q"] = [seq[n.split("::")[0]].upper() if n in seq else "N" for n in w.n]; w = w[~w.q.str.contains("N")].head(TOP)
    gc = np.array([(q.count("G") + q.count("C")) / 200 for q in w.q]); cpg = np.array([q.count("CG") * 200 / max(q.count("C") * q.count("G"), 1) for q in w.q])
    w["st"] = (np.floor(gc / 0.025).astype(int) * 10 + np.searchsorted(cq, cpg)) * 2 + w.n.isin(tss).values; w["pt"] = w.c.map(partf)
    hit = set(int(x) for x in sh(f"module load bedtools >/dev/null 2>&1; bedtools intersect -u -a {O}/ref/bins1kb.bed -b {allp} | cut -f4").split())
    free = np.array([i for i in ei if i not in hit]); r = dict(tf=t, n_windows_all=n_all, n_windows_in_atac=len(inatac), n_peaks_p5=len(pd.read_csv(allp, sep="\t", header=None)) if os.path.getsize(allp) else 0)
    for pt in ["disc", "val"]:
        ww = w[w.pt == pt]; rest = free[part[free] == pt]; fg, bg = [], []
        for s in np.unique(ww.st):
            a = ww[ww.st == s]; c = rest[strat_bg[rest] == s]; m = min(len(a), len(c))
            if m: fg.append(a.sample(m, random_state=int(rng.integers(1e9)))); bg.append(rng.choice(c, m, replace=False))
        fg = pd.concat(fg) if fg else ww.head(0); bg = np.concatenate(bg) if bg else np.array([], int); d = f"{G}/pools/{t}/{pt}"; os.makedirs(d, exist_ok=True)
        fg.sort_values(["c", "s"]).assign(dot=".", strand="+")[["c", "s", "e", "n", "dot", "strand"]].to_csv(f"{d}/fg.bed", sep="\t", header=False, index=False)
        with open(f"{d}/bg.bed", "w") as fo:
            for i in sorted(bg, key=lambda i: (Q.chrom[i], Q.s[i])): fo.write(f"{Q.chrom[i]}\t{Q.s[i]}\t{Q.e[i]}\tb{Q.bin[i]}\t.\t+\n")
        r.update({f"{pt}_win": len(ww), f"{pt}_fg": len(fg), f"{pt}_gc_fg": round(np.mean([(q.count('G') + q.count('C')) / 200 for q in fg.q]), 3) if len(fg) else np.nan,
                  f"{pt}_gc_bg": round(Q.gc.values[bg].mean(), 3) if len(bg) else np.nan})
    rows.append(r)
    if r["disc_fg"] >= 300: known_t += [f"{t}\tknown", f"{t}\tval_known"]; dn_t += [f"{t}\t8", f"{t}\t10", f"{t}\t12"]
pd.DataFrame(rows).to_csv(f"{G}/tf_pools.tsv", sep="\t", index=False); open(f"{G}/tasks_tf.tsv", "w").write("\n".join(known_t + dn_t) + "\n")
print(pd.DataFrame(rows).to_string()); print(len(known_t) // 2, "TFs with >= 300 discovery windows")
