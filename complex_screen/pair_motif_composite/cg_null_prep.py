# Within-pair null for the composite / gapped study (2026-11-13): the random non-candidate pairs have too few co-bound sites (median 6), so for each
# tested pair build 2 pseudo co-bound sets = random single-bound sites (A-only + B-only pool) of the same size and the same GC x CpG x promoter x
# ATAC strata, with a matched background drawn from the rest of the pool. Same pipeline (cg_task.sh) -> how often arbitrary single-bound subsets
# yield "composite" or "gapped" hits. Output pools/<pair>__nullK/..., rows appended to pairs.tsv (set = null_within), tasks_null.tsv
import os, numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; G = f"{O}/composite_gapped"; rng = np.random.default_rng(777)
Q = pd.read_csv(f"{O}/ref/bins_qc.tsv", sep="\t", usecols=["chrom", "s", "e", "bin"]); F = np.load(f"{O}/recall/ref.npz"); strat = F["strat"]
P = pd.read_csv(f"{G}/pairs.tsv", sep="\t"); tasks = [l.strip() for l in open(f"{G}/tasks.tsv") if l.strip()]; B1 = pd.read_csv(f"{O}/ref/bins1kb.bed", sep="\t", header=None, names=["c", "s", "e", "i"])
def rd(f):
    d = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2]); d.columns = ["c", "s", "e"]; d["m"] = (d.s + d.e) // 2; idx = np.full(len(d), -1)
    for c, g in d.groupby("c"):
        b = B1[B1.c == c]; k = np.searchsorted(b.s.values, g.m.values, side="right") - 1; inb = (k >= 0) & (g.m.values < b.e.values[np.maximum(k, 0)]); idx[np.flatnonzero((d.c == c).values)[inb]] = b.i.values[k[inb]]
    d["bin"] = idx; return d[d.bin >= 0]
def wbed(fn, d, w): d.assign(s=d.m - w, e=d.m + w, n=[f"s{k}" for k in range(len(d))], d0=".", st="+")[["c", "s", "e", "n", "d0", "st"]].to_csv(fn, sep="\t", header=False, index=False)
new, nt = [], []
for pair in tasks:
    a, b = pair.split("-"); r = P[P.pair == pair].iloc[0].to_dict()
    for k in [1, 2]:
        name = f"{pair}__null{k}"; info = dict(pair=name, tf_a=a, tf_b=b, set="null_within", af3=r["af3"], best_pct=r["best_pct"])
        for pt in ["disc", "val"]:
            cob = rd(f"{G}/pools/{pair}/{pt}/cob.bed"); pool = pd.concat([rd(f"{G}/pools/{pair}/{pt}/bg.bed"), rd(f"{G}/pools/{pair}/{pt}/aonly.bed")]).drop_duplicates("bin"); pool = pool[~pool.bin.isin(set(cob.bin))]
            fg, bg = [], []
            for s_, n in pd.Series(strat[cob.bin.values]).value_counts().items():
                c = pool[strat[pool.bin.values] == s_].sample(frac=1, random_state=int(rng.integers(1e9)))
                fg.append(c.head(n)); bg.append(c.iloc[n:n + 2 * n])
            fg, bg = pd.concat(fg), pd.concat(bg); d = f"{G}/pools/{name}/{pt}"; os.makedirs(d, exist_ok=True)
            wbed(f"{d}/cob.bed", fg, 100); wbed(f"{d}/bg.bed", bg, 100); wbed(f"{d}/cob_300.bed", fg, 150); wbed(f"{d}/aonly.bed", bg, 150)
            info[f"{pt}_cob"] = len(fg); info[f"{pt}_bg"] = len(bg)
        new.append(info); nt.append(name)
pd.concat([P[~P.pair.str.contains("__null")], pd.DataFrame(new)]).to_csv(f"{G}/pairs.tsv", sep="\t", index=False); open(f"{G}/tasks_null.tsv", "w").write("\n".join(nt) + "\n")
print(len(nt), "null tasks"); print(pd.DataFrame(new)[["pair", "disc_cob", "disc_bg", "val_cob"]].head(6).to_string(index=False))
