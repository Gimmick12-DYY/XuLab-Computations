# Depth downsampling of own-motif recovery (2026-11-12, user go). One task = (TF, n_cells, seed): subsample that many cells from the TF's single-cell
# reads, call peaks exactly like the lab (MACS2 -p 0.01 --nomodel --shift -75 --extsize 150 --keep-dup all --call-summits, no control), take the
# top 1,000 summits inside eligible non-HOT union-ATAC bins, background matched on GC x CpG x promoter x ATAC quintile (2 per site), and run
# HOMER known (own library) on odd and even chromosomes. Records the own-motif rank (same matching as recall_gate.py) and fold.
# Usage: python ds_task.py <TF> <n_cells> <seed>  -> downsample/res/<TF>_<n>_<seed>.json
import os, sys, json, subprocess, shutil, re
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; D = f"{O}/downsample"; FR = "/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment"
t, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); tag = f"{t}_{n}_{seed}"; W = f"{os.environ.get('TMPDIR', '/tmp')}/ds_{tag}"; os.makedirs(W, exist_ok=True); os.makedirs(f"{D}/res", exist_ok=True)
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True, executable="/bin/bash"); assert r.returncode == 0, (c, r.stderr[-1500:]); return r.stdout
R = pd.read_csv(f"{FR}/TF.{t}.bed.gz", sep="\t", header=None); cells = R[3].unique(); rng = np.random.default_rng(seed)
keep = set(rng.choice(cells, min(n, len(cells)), replace=False)); Rs = R[R[3].isin(keep)]; Rs.to_csv(f"{W}/r.bed", sep="\t", header=False, index=False)
sh(f"module load macs/2.2.9.1 >/dev/null 2>&1; macs2 callpeak -t {W}/r.bed -f BED -g hs -p 0.01 --nomodel --shift -75 --extsize 150 --keep-dup all --call-summits -n x --outdir {W}")
pk = pd.read_csv(f"{W}/x_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 7, 9], names=["c", "s", "lp", "sm"]); pk = pk.sort_values("lp", ascending=False); pk["m"] = pk.s + pk.sm
info = dict(tf=t, n_cells=int(len(keep)), seed=seed, n_reads=int(len(Rs)), n_peaks_p1e5=int((pk.lp >= 5).sum()))
Q = pd.read_csv(f"{O}/ref/bins_qc.tsv", sep="\t", usecols=["chrom", "s", "e", "bin"]); F = np.load(f"{O}/recall/ref.npz"); strat, elig, part, hot = F["strat"], F["elig"], F["part"], F["hot"]; ok = elig & ~hot
B = pd.read_csv(f"{O}/ref/bins1kb.bed", sep="\t", header=None, names=["c", "s", "e", "i"]); idx = np.full(len(pk), -1)
for c, g in pk.groupby("c"):
    b = B[B.c == c]
    if not len(b): continue
    k = np.searchsorted(b.s.values, g.m.values, side="right") - 1; inb = (k >= 0) & (g.m.values < b.e.values[np.maximum(k, 0)]); idx[np.flatnonzero((pk.c == c).values)[inb]] = b.i.values[k[inb]]
pk["bin"] = idx; pk = pk[pk.bin >= 0]; pk = pk[ok[pk.bin.values]].drop_duplicates("bin").head(1000); info["n_fg"] = len(pk); info["min_lp_fg"] = round(float(pk.lp.min()), 2) if len(pk) else None
used = np.zeros(len(ok), bool); used[pk.bin.values] = True; free = np.flatnonzero(ok & ~used)
FAM = {"MLX": r"^(Usf[12]|Max|c-Myc|n-Myc|CLOCK|NPAS2|MNT|bHLHE40)\(", "E2F3": r"^E2F[0-9]*\(", "SOX4": r"^Sox[0-9]+\(", "NFYA": r"^NFY\(", "NRF1": r"^NRF1?\(", "ZNF143": r"ZNF143", "MAZ": r"^Maz\(", "CTCF": r"^CTCF\(", "RELA": r"^(p65|NFkB-p65)", "ZIC2": r"^Zic[0-9]?\(", "FOXK1": r"^FOXK[12]\("}
for pt in ["disc", "val"]:
    f_ = pk[part[pk.bin.values] == pt]; rest = free[part[free] == pt]; bg = []
    for s_, cnt in pd.Series(strat[f_.bin.values]).value_counts().items():
        c = rest[strat[rest] == s_]
        if len(c): bg += list(rng.choice(c, min(2 * cnt, len(c)), replace=False))
    f_[["c"]].assign(s=f_.m - 100, e=f_.m + 100, n=[f"p{k}" for k in range(len(f_))], d=".", st="+").to_csv(f"{W}/{pt}_fg.bed", sep="\t", header=False, index=False)
    pd.DataFrame(dict(c=Q.chrom.values[bg], s=Q.s.values[bg], e=Q.e.values[bg], n=[f"b{x}" for x in Q.bin.values[bg]], d=".", st="+")).to_csv(f"{W}/{pt}_bg.bed", sep="\t", header=False, index=False)
    info[f"{pt}_fg"] = len(f_)
    if len(f_) < 100: continue
    sh(f"module load homer/5.1 >/dev/null 2>&1; findMotifsGenome.pl {W}/{pt}_fg.bed hg38 {W}/h_{pt} -bg {W}/{pt}_bg.bed -size given -nomotif -p 1 -preparsedDir {O}/preparsed > /dev/null 2>&1")
    k = pd.read_csv(f"{W}/h_{pt}/knownResults.txt", sep="\t"); k.columns = ["name", "cons", "p", "logp", "q", "nt", "pt", "nb", "pb"]
    m = k.name.str.split("(").str[0].str.upper().str.split(r"[|:/\-]").apply(lambda L: t in L)
    if t in FAM: m |= k.name.str.contains(FAM[t], regex=True)
    if m.any():
        j = int(np.flatnonzero(m.values)[0]); b = k.iloc[j]; info[f"{pt}_own_rank"] = j + 1; info[f"{pt}_own_fold"] = round(float(str(b.pt).rstrip("%")) / max(float(str(b.pb).rstrip("%")), .01), 2); info[f"{pt}_own_log10p"] = round(b.logp / np.log(10), 1)
    info[f"{pt}_top"] = k.name.iloc[0].split("/")[0]
json.dump(info, open(f"{D}/res/{tag}.json", "w")); print(info); shutil.rmtree(W, ignore_errors=True)
