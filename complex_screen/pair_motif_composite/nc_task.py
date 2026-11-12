# Negative-control pseudo-libraries for motif enrichment (2026-11-12, user: is a negative control needed?). Same pipeline as the own-motif gate.
#  mode "pooled": 3,331 cells (panel median) drawn at random from ALL 79 libraries pooled -> a Tn5 library with no single target TF.
#  mode "random": 1,000 random eligible non-HOT open-chromatin bins as "sites" (bin centre +/- 100 bp) -> pure composition control.
# Then: top 1,000 summits in eligible non-HOT union-ATAC bins, matched background (GC x CpG x promoter x ATAC quintile, 2 per site), HOMER known
# on odd and even chromosomes. Usage: python nc_task.py <pooled|random> <seed> -> motif_tables/nc/<mode>_<seed>_{disc,val}.tsv (HOMER knownResults)
import os, sys, glob, shutil, subprocess
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; T = f"{O}/motif_tables/nc"; FR = "/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment"; os.makedirs(T, exist_ok=True)
mode, seed = sys.argv[1], int(sys.argv[2]); rng = np.random.default_rng(1000 + seed); W = f"{os.environ.get('TMPDIR', '/tmp')}/nc_{mode}_{seed}"; os.makedirs(W, exist_ok=True)
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True, executable="/bin/bash"); assert r.returncode == 0, (c, r.stderr[-1500:]); return r.stdout
Q = pd.read_csv(f"{O}/ref/bins_qc.tsv", sep="\t", usecols=["chrom", "s", "e", "bin"]); F = np.load(f"{O}/recall/ref.npz"); strat, elig, part, hot = F["strat"], F["elig"], F["part"], F["hot"]; ok = elig & ~hot
if mode == "pooled":
    cells = pd.read_csv(f"{X}/complex_screen/pair_motif_lookup/tf_cells.tsv", sep="\t").set_index("tf").n_cells; tot = cells.sum(); keep_n = {t: int(rng.binomial(cells[t], 3331 / tot)) for t in cells.index}
    parts = []
    for t, k in keep_n.items():
        if k == 0: continue
        R = pd.read_csv(f"{FR}/TF.{t}.bed.gz", sep="\t", header=None); cs = R[3].unique(); kk = set(rng.choice(cs, min(k, len(cs)), replace=False)); parts.append(R[R[3].isin(kk)])
    pd.concat(parts).to_csv(f"{W}/r.bed", sep="\t", header=False, index=False)
    sh(f"module load macs/2.2.9.1 >/dev/null 2>&1; macs2 callpeak -t {W}/r.bed -f BED -g hs -p 0.01 --nomodel --shift -75 --extsize 150 --keep-dup all --call-summits -n x --outdir {W}")
    pk = pd.read_csv(f"{W}/x_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 7, 9], names=["c", "s", "lp", "sm"]).sort_values("lp", ascending=False); pk["m"] = pk.s + pk.sm
    B = pd.read_csv(f"{O}/ref/bins1kb.bed", sep="\t", header=None, names=["c", "s", "e", "i"]); idx = np.full(len(pk), -1)
    for c, g in pk.groupby("c"):
        b = B[B.c == c]
        if not len(b): continue
        k = np.searchsorted(b.s.values, g.m.values, side="right") - 1; inb = (k >= 0) & (g.m.values < b.e.values[np.maximum(k, 0)]); idx[np.flatnonzero((pk.c == c).values)[inb]] = b.i.values[k[inb]]
    pk["bin"] = idx; pk = pk[pk.bin >= 0]; pk = pk[ok[pk.bin.values]].drop_duplicates("bin").head(1000)
    fg = pd.DataFrame(dict(c=pk.c.values, s=pk.m.values - 100, e=pk.m.values + 100, bin=pk.bin.values))
else:
    cand = np.flatnonzero(ok); pick = rng.choice(cand, 1000, replace=False); fg = pd.DataFrame(dict(c=Q.chrom.values[pick], s=Q.s.values[pick], e=Q.e.values[pick], bin=pick))
used = np.zeros(len(ok), bool); used[fg.bin.values] = True; free = np.flatnonzero(ok & ~used)
for pt in ["disc", "val"]:
    f_ = fg[part[fg.bin.values] == pt]; rest = free[part[free] == pt]; bg = []
    for s_, cnt in pd.Series(strat[f_.bin.values]).value_counts().items():
        c = rest[strat[rest] == s_]
        if len(c): bg += list(rng.choice(c, min(2 * cnt, len(c)), replace=False))
    f_.assign(n=[f"p{k}" for k in range(len(f_))], d=".", st="+")[["c", "s", "e", "n", "d", "st"]].to_csv(f"{W}/{pt}_fg.bed", sep="\t", header=False, index=False)
    pd.DataFrame(dict(c=Q.chrom.values[bg], s=Q.s.values[bg], e=Q.e.values[bg], n=[f"b{x}" for x in Q.bin.values[bg]], d=".", st="+")).to_csv(f"{W}/{pt}_bg.bed", sep="\t", header=False, index=False)
    sh(f"module load homer/5.1 >/dev/null 2>&1; findMotifsGenome.pl {W}/{pt}_fg.bed hg38 {W}/h_{pt} -bg {W}/{pt}_bg.bed -size given -nomotif -p 1 -preparsedDir {O}/preparsed > /dev/null 2>&1")
    shutil.copy(f"{W}/h_{pt}/knownResults.txt", f"{T}/{mode}_{seed}_{pt}.tsv")
print("done", mode, seed, len(fg)); shutil.rmtree(W, ignore_errors=True)
