# GC-matched pools for the candidate pairs: triple-support T1 + T2 (triple_support_20261109/triple_support.tsv) plus the AF3/named
# candidates of the first run (pool_sizes.tsv), deduplicated. Pools are built as in build_pools.py (top 10,000 bins per TF,
# same eligibility, odd/even chromosome split), then matched as in gc_match.py (GC x CpG o/e x promoter strata, equal counts of
# pair / A-only / B-only per stratum, input up to the same count). Pairs with < 300 matched discovery pair bins are untestable.
# Output: triple_support/pools/<pair>/{disc,val,perm/<k>}, triple_support/pool_sizes.tsv, triple_support/matching.tsv
import os
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; G = f"{O}/triple_support"; M = f"{X}/complex_screen/motif_shared_20261103/fg"
TOP = 10000; MINPOOL = 300; NPERM = 50; rng = np.random.default_rng(20261110)
z = np.load(f"{M}/signal_mask.npz"); S = z["S"]; tfs = [t.upper() for t in z["tfs"]]; ix = {t: k for k, t in enumerate(tfs)}; D = S - S.mean(0)
Q = pd.read_csv(f"{O}/ref/bins_qc.tsv", sep="\t"); num = Q.chrom.str.replace("chr", "")
part = np.where(num.isin([str(i) for i in range(1, 23, 2)]), "disc", np.where(num.isin([str(i) for i in range(2, 23, 2)] + ["X"]), "val", "none"))
elig = Q.n_ok.values & ~Q.blacklist.values & (Q.exonic.values <= 0.5) & (part != "none"); ei = np.flatnonzero(elig)
top = {t: set(ei[np.argsort(-D[ix[t]][ei], kind="stable")[:TOP]]) for t in tfs}
prom = np.zeros(len(Q), bool); prom[[int(x) for x in open(f"{O}/ref/promoter_bins.txt").read().split()]] = True
cq = np.quantile(Q.cpg_oe.values[ei], [0.2, 0.4, 0.6, 0.8]); strat = (np.floor(Q.gc.values / 0.025).astype(int) * 10 + np.searchsorted(cq, Q.cpg_oe.values)) * 2 + prom
# NULLOF="FOCAL:REF,..." (with OUT=<folder>): instead, for each focal TF, 20 random partners whose discovery top-set overlap with the
# focal TF is closest to that of FOCAL-REF (excluding REF), written to OUT. Used for the NFYA / DDIT3 partner nulls.
if os.environ.get("NULLOF"):
    G = os.environ["OUT"]; src = []
    for fr in os.environ["NULLOF"].split(","):
        f_, r_ = fr.split(":"); ov = lambda c: len([i for i in top[f_] & top[c] if part[i] == "disc"]); target = ov(r_)
        cands = sorted((c for c in tfs if c not in (f_, r_)), key=lambda c: abs(ov(c) - target))[:20]
        src += [(f"{f_}-{c}", f"null_{f_}_vs_{r_}(overlap {target})") for c in cands]
else:
    T = pd.read_csv(f"{X}/complex_screen/triple_support_20261109/triple_support.tsv", sep="\t"); T = T[T.tier.isin(["T1", "T2"])]
    PS = pd.read_csv(f"{O}/pool_sizes.tsv", sep="\t"); PS = PS[PS.category.str.startswith("af3") | (PS.category == "named")]
    src = list(zip(T.pair, T.tier)) + list(zip(PS.pair, PS.category))
pairs, seen = [], {}
for p, c in src:
    k = tuple(sorted(p.split("-")))
    if k in seen: seen[k][2] += f";{c}"; continue
    seen[k] = [*p.split("-"), c]; pairs.append(seen[k])
def wbed(fn, idx):
    with open(fn, "w") as f:
        for i in sorted(idx, key=lambda i: (Q.chrom[i], Q.s[i])): f.write(f"{Q.chrom[i]}\t{Q.s[i]}\t{Q.e[i]}\tb{Q.bin[i]}\t.\t+\n")
def wfa(fn, idx):
    with open(fn, "w") as f:
        for i in idx: f.write(f">b{Q.bin[i]}\n{Q.seq[i]}\n")
def desc(idx): return [len(idx)] + ([round(v[idx].mean(), 3) for v in (Q.gc.values, Q.cpg_oe.values, prom)] if len(idx) else [np.nan] * 3)
sizes, match = [], []
for a, b, cat in pairs:
    assert a in ix and b in ix, (a, b)
    A, B = top[a], top[b]; r = dict(pair=f"{a}-{b}", a=a, b=b, category=cat)
    for pt in ["disc", "val"]:
        sel = lambda s_: np.array(sorted(i for i in s_ if part[i] == pt), int)
        src = dict(pair=sel(A & B), Aonly=sel(A - B), Bonly=sel(B - A)); rest = np.setdiff1d(ei[part[ei] == pt], np.fromiter(A | B, int))
        keep = {k: [] for k in ["pair", "Aonly", "Bonly", "input"]}; short = 0
        for s in np.unique(strat[src["pair"]]):
            pools = {k: v[strat[v] == s] for k, v in src.items()}; m = min(len(v) for v in pools.values())
            if m == 0: continue
            for k, v in pools.items(): keep[k].append(rng.choice(v, m, replace=False))
            rs = rest[strat[rest] == s]; keep["input"].append(rng.choice(rs, min(m, len(rs)), replace=False)); short += max(m - len(rs), 0)
        keep = {k: np.concatenate(v) if v else np.array([], int) for k, v in keep.items()}; keep["singles"] = np.concatenate([keep["Aonly"], keep["Bonly"]])
        d = f"{G}/pools/{a}-{b}/{pt}"; os.makedirs(d, exist_ok=True)
        for k, v in keep.items(): wbed(f"{d}/{k}.bed", v); wfa(f"{d}/{k}.fa", v)
        for k in ["pair", "Aonly", "Bonly"]:
            for when, v in [("before", src[k]), ("after", keep[k])]: match.append(dict(zip(["pair", "part", "pool", "stage", "n", "gc", "cpg_oe", "promoter_frac"], [f"{a}-{b}", pt, k, when] + desc(v))))
        match.append(dict(zip(["pair", "part", "pool", "stage", "n", "gc", "cpg_oe", "promoter_frac"], [f"{a}-{b}", pt, "input", "after"] + desc(keep["input"]))))
        r.update({f"{pt}_pair_before": len(src["pair"]), f"{pt}_pair": len(keep["pair"]), f"{pt}_Aonly": len(keep["Aonly"]), f"{pt}_Bonly": len(keep["Bonly"]), f"{pt}_input": len(keep["input"]),
                  f"{pt}_input_short": short, f"{pt}_gc_pair": desc(keep["pair"])[1], f"{pt}_gc_input": desc(keep["input"])[1]})
        if pt == "disc":
            U = np.concatenate([keep["pair"], keep["Aonly"], keep["Bonly"]]); lab = np.repeat([2, 0, 1], [len(keep["pair"])] * 3)
            for k in range(NPERM if not os.environ.get("NULLOF") else 0):
                L = lab.copy()
                for s in np.unique(strat[U]): q = np.flatnonzero(strat[U] == s); L[q] = rng.permutation(L[q])
                dd = f"{G}/pools/{a}-{b}/perm/{k:02d}"; os.makedirs(dd, exist_ok=True)
                for nm, v in [("pair", 2), ("Aonly", 0), ("Bonly", 1)]: wbed(f"{dd}/{nm}.bed", U[L == v])
    r["testable"] = r["disc_pair"] >= MINPOOL; sizes.append(r)
Z = pd.DataFrame(sizes); Z.to_csv(f"{G}/pool_sizes.tsv", sep="\t", index=False); pd.DataFrame(match).to_csv(f"{G}/matching.tsv", sep="\t", index=False)
print(len(Z), "pairs,", int(Z.testable.sum()), "testable"); print(Z[["pair", "category", "disc_pair_before", "disc_pair", "disc_input_short", "val_pair", "testable"]].to_string())
