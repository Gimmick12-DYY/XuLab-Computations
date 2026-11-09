# GC-matched variant of the gate run (gate/negative + 20 random pairs).
# Starts from the pools in ../pools/<pair>/{disc,val} and subsamples them so pair, A-only, B-only and input have the same
# composition: strata = GC (0.025 steps) x CpG o/e (quintiles of eligible bins) x promoter/distal (TSS of a GENCODE v46 basic
# transcript within the 1 kb bin +/- 1 kb). Per stratum: m = min(n_pair, n_Aonly, n_Bonly) bins kept from each of the three;
# input = up to m eligible bins outside both top sets from the same stratum (shortfall reported).
# Writes gc_matched/pools/<pair>/{disc,val}, gc_matched/pools/<pair>/perm/<k> (gate/negative pairs: labels shuffled within strata),
# gc_matched/pool_sizes.tsv and gc_matched/matching.tsv (GC, CpG o/e, promoter fraction per pool before and after).
import os, subprocess
import numpy as np, pandas as pd
O = "/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite"; G = f"{O}/gc_matched"; X = "/work/users/d/y/dyy12/XuLab"
SEED = 20261109; NPERM = 50; MINPOOL = 300; rng = np.random.default_rng(SEED)
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True, executable="/bin/bash"); assert r.returncode == 0, (c, r.stderr); return r.stdout
Q = pd.read_csv(f"{O}/ref/bins_qc.tsv", sep="\t"); pos = {b: i for i, b in enumerate(Q.bin)}
if not os.path.exists(f"{O}/ref/promoter_bins.txt"):
    sh(f"""zcat {X}/data/gencode_v46/gencode.v46.basic.annotation.gtf.gz | awk -F'\\t' 'BEGIN{{OFS="\\t"}} $3=="transcript"{{t=($7=="+")?$4-1:$5-1; print $1,t,t+1}}' | sort -u > {O}/ref/tss.bed""")
    out = sh(f"module load bedtools >/dev/null 2>&1; bedtools window -u -w 1000 -a {O}/ref/bins1kb.bed -b {O}/ref/tss.bed | cut -f4")
    open(f"{O}/ref/promoter_bins.txt", "w").write(out)
prom = np.zeros(len(Q), bool); prom[[int(x) for x in open(f"{O}/ref/promoter_bins.txt").read().split()]] = True
num = Q.chrom.str.replace("chr", "")
part = np.where(num.isin([str(i) for i in range(1, 23, 2)]), "disc", np.where(num.isin([str(i) for i in range(2, 23, 2)] + ["X"]), "val", "none"))
elig = Q.n_ok.values & ~Q.blacklist.values & (Q.exonic.values <= 0.5) & (part != "none"); ei = np.flatnonzero(elig)
cq = np.quantile(Q.cpg_oe.values[ei], [0.2, 0.4, 0.6, 0.8]); strat = (np.floor(Q.gc.values / 0.025).astype(int) * 10 + np.searchsorted(cq, Q.cpg_oe.values)) * 2 + prom
def rd(f): return np.array([pos[int(l.split("\t")[3][1:])] for l in open(f)], int) if os.path.getsize(f) else np.array([], int)
def wbed(fn, idx):
    with open(fn, "w") as f:
        for i in sorted(idx, key=lambda i: (Q.chrom[i], Q.s[i])): f.write(f"{Q.chrom[i]}\t{Q.s[i]}\t{Q.e[i]}\tb{Q.bin[i]}\t.\t+\n")
def wfa(fn, idx):
    with open(fn, "w") as f:
        for i in idx: f.write(f">b{Q.bin[i]}\n{Q.seq[i]}\n")
def desc(idx): return (len(idx), round(Q.gc.values[idx].mean(), 3) if len(idx) else np.nan, round(Q.cpg_oe.values[idx].mean(), 3) if len(idx) else np.nan, round(prom[idx].mean(), 3) if len(idx) else np.nan)
PS = pd.read_csv(f"{O}/pool_sizes.tsv", sep="\t"); PS = PS[PS.category.isin(["gate", "negative"]) | PS.category.str.startswith("random_matched")]
sizes, match = [], []
for _, row in PS.iterrows():
    p = row.pair; r = dict(pair=p, a=row.a, b=row.b, category=row.category)
    for pt in ["disc", "val"]:
        src = {k: rd(f"{O}/pools/{p}/{pt}/{k}.bed") for k in ["pair", "Aonly", "Bonly", "input"]}
        AB = np.concatenate([src["pair"], src["Aonly"], src["Bonly"]]); rest = np.setdiff1d(ei[part[ei] == pt], AB)
        keep = {k: [] for k in ["pair", "Aonly", "Bonly", "input"]}; short = 0
        for s in np.unique(strat[src["pair"]]):
            pools = {k: src[k][strat[src[k]] == s] for k in ["pair", "Aonly", "Bonly"]}; m = min(len(v) for v in pools.values())
            if m == 0: continue
            for k, v in pools.items(): keep[k].append(rng.choice(v, m, replace=False))
            rs = rest[strat[rest] == s]; keep["input"].append(rng.choice(rs, min(m, len(rs)), replace=False)); short += max(m - len(rs), 0)
        keep = {k: np.concatenate(v) if v else np.array([], int) for k, v in keep.items()}; keep["singles"] = np.concatenate([keep["Aonly"], keep["Bonly"]])
        d = f"{G}/pools/{p}/{pt}"; os.makedirs(d, exist_ok=True)
        for k, idx in keep.items(): wbed(f"{d}/{k}.bed", idx); wfa(f"{d}/{k}.fa", idx)
        for k in ["pair", "Aonly", "Bonly", "input"]:
            for when, idx in [("before", src[k]), ("after", keep[k])]:
                n, gc, cpg, pr = desc(idx); match.append(dict(pair=p, part=pt, pool=k, stage=when, n=n, gc=gc, cpg_oe=cpg, promoter_frac=pr))
        r.update({f"{pt}_pair": len(keep["pair"]), f"{pt}_pair_before": len(src["pair"]), f"{pt}_Aonly": len(keep["Aonly"]), f"{pt}_Bonly": len(keep["Bonly"]),
                  f"{pt}_input": len(keep["input"]), f"{pt}_input_short": short, f"{pt}_gc_pair": desc(keep["pair"])[1], f"{pt}_gc_input": desc(keep["input"])[1]})
        if pt == "disc" and row.category in ("gate", "negative"):
            U = np.concatenate([keep["pair"], keep["Aonly"], keep["Bonly"]]); lab = np.repeat([2, 0, 1], [len(keep["pair"]), len(keep["Aonly"]), len(keep["Bonly"])])
            for k in range(NPERM):
                L = lab.copy()
                for s in np.unique(strat[U]): ix = np.flatnonzero(strat[U] == s); L[ix] = rng.permutation(L[ix])
                dd = f"{G}/pools/{p}/perm/{k:02d}"; os.makedirs(dd, exist_ok=True)
                for nm, v in [("pair", 2), ("Aonly", 0), ("Bonly", 1)]: wbed(f"{dd}/{nm}.bed", U[L == v])
    r["testable"] = r["disc_pair"] >= MINPOOL; sizes.append(r)
pd.DataFrame(sizes).to_csv(f"{G}/pool_sizes.tsv", sep="\t", index=False); M = pd.DataFrame(match); M.to_csv(f"{G}/matching.tsv", sep="\t", index=False)
print(pd.DataFrame(sizes)[["pair", "category", "disc_pair_before", "disc_pair", "disc_Aonly", "disc_Bonly", "disc_input", "disc_input_short", "val_pair_before", "val_pair", "testable"]].to_string())
print(M[M.part == "disc"].pivot_table(index=["pair", "stage"], columns="pool", values=["gc", "cpg_oe", "promoter_frac"]).round(3).to_string())
