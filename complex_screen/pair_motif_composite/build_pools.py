# Pools for the composite-motif pair test.
# Per TF: 1 kb union-ATAC bins ranked by imputed CPM minus the 79-TF mean (signal matrix from motif_shared_20261103/fg/signal_mask.npz),
# after dropping bins with N in the centre 200 bp, bins overlapping the ENCODE blacklist and bins >50% exonic (GENCODE v46 basic); top 10,000 kept.
# Pair A-B: pair = in both top sets; Aonly / Bonly; input = eligible bins in neither, GC x CpG o/e matched to the pair pool (2 per pair bin where available).
# Sequences = bin centre +/- 100 bp (hg38). Discovery = odd autosomes, validation = even autosomes + chrX.
# Usage: python build_pools.py            -> pairs.tsv, pool_sizes.tsv, pools/<pair>/{disc,val}/*
#        python build_pools.py --perms    -> pools/<pair>/perm/<k>/{pair,Aonly,Bonly}.bed for the gate/negative pairs
import os, sys, subprocess, itertools
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; M = f"{X}/complex_screen/motif_shared_20261103/fg"
TOP = 10000; MINPOOL = 300; NPERM = 50; SEED = 20261106
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True, executable="/bin/bash"); assert r.returncode == 0, (c, r.stderr); return r.stdout
z = np.load(f"{M}/signal_mask.npz"); S = z["S"]; tfs = [t.upper() for t in z["tfs"]]; ix = {t: k for k, t in enumerate(tfs)}; D = S - S.mean(0)
bed = pd.read_csv(f"{M}/mask_center.bed", sep="\t", header=None, names=["chrom", "s", "e", "bin"]); assert len(bed) == S.shape[1]
os.makedirs(f"{O}/ref", exist_ok=True)
if not os.path.exists(f"{O}/ref/bins_qc.tsv"):
    b1 = bed.assign(s=bed.s - 400, e=bed.e + 400, i=np.arange(len(bed)))[["chrom", "s", "e", "i"]]; b1.to_csv(f"{O}/ref/bins1kb.bed", sep="\t", header=False, index=False)
    sh(f"zcat {X}/data/gencode_v46/gencode.v46.basic.annotation.gtf.gz | awk -F'\\t' '$3==\"exon\"' | cut -f1,4,5 | sort -k1,1 -k2,2n > {O}/ref/exons.bed")
    ex = sh(f"module load bedtools >/dev/null 2>&1; sort -k1,1 -k2,2n {O}/ref/bins1kb.bed | bedtools coverage -a - -b {O}/ref/exons.bed | cut -f4,8")
    exf = np.zeros(len(bed)); 
    for l in ex.splitlines(): i, f = l.split("\t"); exf[int(i)] = float(f)
    bl = sh(f"module load bedtools >/dev/null 2>&1; bedtools intersect -u -a {O}/ref/bins1kb.bed -b {X}/downstream/cache/hg38-blacklist.v2.bed.gz | cut -f4")
    blk = np.zeros(len(bed), bool); blk[[int(x) for x in bl.split()]] = True
    fa = sh(f"module load bedtools >/dev/null 2>&1; bedtools getfasta -nameOnly -tab -fi {X}/downstream/cache/hg38.fa -bed {M}/mask_center.bed")
    sq = {}
    for l in fa.splitlines(): n, q = l.split("\t"); sq[int(n.split("::")[0])] = q.upper()
    seq = [sq[i] for i in bed.bin]
    gc = np.array([(q.count("G") + q.count("C")) / max(len(q), 1) for q in seq]); cpg = np.array([q.count("CG") * len(q) / max(q.count("C") * q.count("G"), 1) for q in seq])
    okn = np.array([("N" not in q) and len(q) == 200 for q in seq])
    pd.DataFrame(dict(chrom=bed.chrom, s=bed.s, e=bed.e, bin=bed.bin, seq=seq, gc=gc, cpg_oe=cpg, exonic=exf, blacklist=blk, n_ok=okn)).to_csv(f"{O}/ref/bins_qc.tsv", sep="\t", index=False)
Q = pd.read_csv(f"{O}/ref/bins_qc.tsv", sep="\t")
num = Q.chrom.str.replace("chr", "")
part = np.where(num.isin([str(i) for i in range(1, 23, 2)]), "disc", np.where(num.isin([str(i) for i in range(2, 23, 2)] + ["X"]), "val", "none"))
elig = Q.n_ok.values & ~Q.blacklist.values & (Q.exonic.values <= 0.5) & (part != "none")
ei = np.flatnonzero(elig)
top = {t: set(ei[np.argsort(-D[ix[t]][ei], kind="stable")[:TOP]]) for t in tfs}
gcb = np.round(Q.gc.values * 50).astype(int); cq = np.quantile(Q.cpg_oe.values[ei], [0.2, 0.4, 0.6, 0.8]); cb = np.searchsorted(cq, Q.cpg_oe.values); strat = gcb * 10 + cb
def name(a, b): return f"{a}-{b}"
gate = [("CTCF", "STAG2", "gate"), ("MAZ", "ZNF143", "gate"), ("NFYA", "ZNF644", "gate"), ("E2F3", "HES1", "gate"), ("RBBP7", "SRCAP", "negative")]
alias = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}
cand = []; dropped = []
for f in ["af3_scores.tsv", "af3_scores_weak.tsv"]:
    t = pd.read_csv(f"{X}/complex_screen/af3_novel_20261103/{f}", sep="\t")
    for p in t[t.tier.isin(["strong", "weak"])].pair:
        a, b = [alias.get(x, x) for x in p.split("__")]
        (cand if a in ix and b in ix else dropped).append((a, b, f"af3_{'strong5' if f == 'af3_scores.tsv' else 'weak1'}"))
cand += [(a, b, "named") for a, b in [("FOXK1", "MLX"), ("MLX", "MSX2"), ("HOXD10", "SREBF2"), ("EZH2", "MTF2"), ("WIZ", "ZNF644"), ("PCGF6", "RBBP4"), ("RBBP4", "RBBP7"), ("DR1", "THAP6")]]
seen = set(); pairs = []
for a, b, c in gate + cand:
    k = tuple(sorted((a, b)))
    if k not in seen: seen.add(k); pairs.append((a, b, c))
ov = {k: len(top[k[0]] & top[k[1]]) for k in itertools.combinations(sorted(tfs), 2)}
rng = np.random.default_rng(SEED); taken = set(seen)
for a, b, c in gate:
    o = ov[tuple(sorted((a, b)))]; pool = sorted((k for k in ov if k not in taken), key=lambda k: abs(np.log((ov[k] + 1) / (o + 1))))[:30]
    for j in rng.choice(len(pool), 4, replace=False): k = pool[j]; taken.add(k); pairs.append((k[0], k[1], f"random_matched_{a}-{b}"))
def wbed(fn, idx):
    with open(fn, "w") as f:
        for i in sorted(idx, key=lambda i: (Q.chrom[i], Q.s[i])): f.write(f"{Q.chrom[i]}\t{Q.s[i]}\t{Q.e[i]}\tb{Q.bin[i]}\t.\t+\n")
def wfa(fn, idx):
    with open(fn, "w") as f:
        for i in idx: f.write(f">b{Q.bin[i]}\n{Q.seq[i]}\n")
if "--perms" not in sys.argv:
    rows = []
    for a, b, c in pairs:
        A, B = top[a], top[b]; P = A & B; Ao = A - B; Bo = B - A; rest = np.setdiff1d(ei, np.fromiter(A | B, int))
        r = dict(pair=name(a, b), a=a, b=b, category=c, overlap_all=len(P))
        for pt in ["disc", "val"]:
            pp = np.array([i for i in P if part[i] == pt], int); ao = np.array([i for i in Ao if part[i] == pt], int); bo = np.array([i for i in Bo if part[i] == pt], int)
            rp = rest[part[rest] == pt]; inp = []; short = 0
            for s, n in zip(*np.unique(strat[pp], return_counts=True)):
                cands = rp[strat[rp] == s]; short += max(2 * n - len(cands), 0); inp.append(rng.choice(cands, min(2 * n, len(cands)), replace=False))
            inp = np.concatenate(inp) if inp else np.array([], int)
            d = f"{O}/pools/{name(a, b)}/{pt}"; os.makedirs(d, exist_ok=True)
            for nm, idx in [("pair", pp), ("Aonly", ao), ("Bonly", bo), ("input", inp), ("singles", np.concatenate([ao, bo]))]: wbed(f"{d}/{nm}.bed", idx); wfa(f"{d}/{nm}.fa", idx)
            r.update({f"{pt}_pair": len(pp), f"{pt}_Aonly": len(ao), f"{pt}_Bonly": len(bo), f"{pt}_input": len(inp), f"{pt}_input_short": short,
                      f"{pt}_gc_pair": round(Q.gc.values[pp].mean(), 3) if len(pp) else np.nan, f"{pt}_gc_input": round(Q.gc.values[inp].mean(), 3) if len(inp) else np.nan})
        r["testable"] = r["disc_pair"] >= MINPOOL; rows.append(r)
    T = pd.DataFrame(rows); T.to_csv(f"{O}/pool_sizes.tsv", sep="\t", index=False)
    T[["pair", "a", "b", "category"]].to_csv(f"{O}/pairs.tsv", sep="\t", index=False)
    print(f"eligible bins {len(ei)} (disc {np.sum(part[ei] == 'disc')}, val {np.sum(part[ei] == 'val')}); dropped blacklist {Q.blacklist.sum()}, exonic>0.5 {(Q.exonic > 0.5).sum()}, N {(~Q.n_ok).sum()}")
    print("candidate pairs not in panel:", dropped); print(T.to_string())
else:
    # label permutations on discovery chromosomes: shuffle pair / A-only / B-only labels across A u B, within expected-read (79-TF mean CPM, quintile) x GC strata
    er = S.mean(0); eq = np.searchsorted(np.quantile(er[ei], [0.2, 0.4, 0.6, 0.8]), er); ps = eq * 100 + gcb
    for a, b, c in gate:
        A, B = top[a], top[b]; U = np.array(sorted(i for i in A | B if part[i] == "disc")); lab = np.array([2 if (i in A and i in B) else (0 if i in A else 1) for i in U])
        for k in range(NPERM):
            L = lab.copy()
            for s in np.unique(ps[U]): m = np.flatnonzero(ps[U] == s); L[m] = rng.permutation(L[m])
            d = f"{O}/pools/{name(a, b)}/perm/{k:02d}"; os.makedirs(d, exist_ok=True)
            for nm, v in [("pair", 2), ("Aonly", 0), ("Bonly", 1)]: wbed(f"{d}/{nm}.bed", U[L == v])
    print("perms written")
