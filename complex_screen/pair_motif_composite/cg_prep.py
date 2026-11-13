# Composite + gapped (spaced) motif study, prep (2026-11-13, user: run both in parallel). Same site rules as the p <= 0.05 pair table:
# per TF the top 5,000 lab MACS2 summits (p <= 1e-5 peaks) whose summit lies in an eligible, non-HOT union-ATAC bin.
# Pair A-B: co-bound = A summits with a B summit within 150 bp; A-only = A summits with no B peak (p <= 1e-5) within 500 bp; same for B-only.
# Composite background = A-only + B-only, matched to co-bound on GC x CpG x promoter x ATAC quintile (2 per site). Odd chr = discovery, even + X = held-out.
# Pairs: AF3 strong/weak candidates, known pairs (excl. NEUROD4 / STAG2-class), and 60 random non-candidate, non-known pairs as the null.
# Output composite_gapped/pairs.tsv, composite_gapped/pools/<pair>/{disc,val}/{cob,bg,aonly}.bed (+ cob_300.fa for SpaMo), tasks.tsv
import os, subprocess
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; G = f"{O}/composite_gapped"; MAC = "/vast/som/xujie_lab/TAGATG-293T/bigwig/macs2"; rng = np.random.default_rng(20261113)
os.makedirs(f"{G}/pools", exist_ok=True)
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True, executable="/bin/bash"); assert r.returncode == 0, (c, r.stderr[-1500:]); return r.stdout
Q = pd.read_csv(f"{O}/ref/bins_qc.tsv", sep="\t", usecols=["chrom", "s", "e", "bin", "gc", "cpg_oe"]); F = np.load(f"{O}/recall/ref.npz"); strat, elig, part, hot = F["strat"], F["elig"], F["part"], F["hot"]; ok = elig & ~hot
B = pd.read_csv(f"{O}/ref/bins1kb.bed", sep="\t", header=None, names=["c", "s", "e", "i"]); tfs = [l.strip() for l in open(f"{O}/recall/tfs.txt")]
S, ALL = {}, {}
for t in tfs:
    pk = pd.read_csv(f"{MAC}/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 2, 7, 9], names=["c", "s", "e", "lp", "sm"]); pk = pk[pk.lp >= 5]; pk["m"] = pk.s + pk.sm
    ALL[t] = pk[["c", "s", "e"]]; pk = pk.sort_values("lp", ascending=False); idx = np.full(len(pk), -1)
    for c, g in pk.groupby("c"):
        b = B[B.c == c]
        if not len(b): continue
        k = np.searchsorted(b.s.values, g.m.values, side="right") - 1; inb = (k >= 0) & (g.m.values < b.e.values[np.maximum(k, 0)]); idx[np.flatnonzero((pk.c == c).values)[inb]] = b.i.values[k[inb]]
    pk["bin"] = idx; pk = pk[pk.bin >= 0]; pk = pk[ok[pk.bin.values]].drop_duplicates(["c", "m"]).head(5000); S[t] = pk[["c", "m", "bin", "lp"]].reset_index(drop=True)
A = pd.read_csv(f"{X}/complex_screen/candidates_frozen_v1/candidates_all3081.tsv", sep="\t"); A["af3"] = [b if b != "not_run" else a for a, b in zip(A.af3_1seed_tier, A.af3_5seed_tier)]
af3 = A[(A.candidate.astype(str) == "True") & A.af3.isin(["strong", "weak"])].assign(set="af3_" + A.af3)
kn = A[(A.known.astype(str) == "True") & ~A.known_reason.isin(["NEUROD4", "STAG2_class"])].assign(set="known")
rest = A[(A.candidate.astype(str) != "True") & (A.known.astype(str) != "True")]; nul = rest.sample(60, random_state=20261113).assign(set="null_random")
PR = pd.concat([af3, kn, nul])[["pair", "tf_a", "tf_b", "best_pct", "af3", "set"]].drop_duplicates("pair")
def near(x, y, d):  # for each row of x, is there a y position (same chrom) within d
    out = np.zeros(len(x), bool)
    for c, g in x.groupby("c"):
        yy = np.sort(y[y.c == c].m.values) if "m" in y else None
        if yy is None or not len(yy): continue
        k = np.searchsorted(yy, g.m.values); lo = np.abs(g.m.values - yy[np.clip(k - 1, 0, len(yy) - 1)]); hi = np.abs(g.m.values - yy[np.clip(k, 0, len(yy) - 1)]); out[np.flatnonzero((x.c == c).values)] = np.minimum(lo, hi) <= d
    return out
def overl(x, iv, pad):
    out = np.zeros(len(x), bool)
    for c, g in x.groupby("c"):
        v = iv[iv.c == c]
        if not len(v): continue
        s = np.sort(v.s.values - pad); e = v.e.values[np.argsort(v.s.values)] + pad; em = np.maximum.accumulate(e); k = np.searchsorted(s, g.m.values, side="right") - 1
        out[np.flatnonzero((x.c == c).values)] = (k >= 0) & (g.m.values <= em[np.maximum(k, 0)])
    return out
rows, tasks = [], []
def wbed(fn, d, w): d.assign(s=d.m - w, e=d.m + w, n=[f"s{k}" for k in range(len(d))], d0=".", st="+")[["c", "s", "e", "n", "d0", "st"]].to_csv(fn, sep="\t", header=False, index=False)
for r in PR.itertuples():
    a, b = r.tf_a, r.tf_b
    if a not in S or b not in S: continue
    Sa, Sb = S[a], S[b]; cob = Sa[near(Sa, Sb, 150)]; aon = Sa[~overl(Sa, ALL[b], 500)]; bon = Sb[~overl(Sb, ALL[a], 500)]; info = dict(pair=r.pair, tf_a=a, tf_b=b, set=r.set, af3=r.af3, best_pct=r.best_pct)
    used = set(cob.bin); pool = pd.concat([aon, bon]); pool = pool[~pool.bin.isin(used)].drop_duplicates("bin")
    for pt in ["disc", "val"]:
        cb = cob[part[cob.bin.values] == pt]; pl = pool[part[pool.bin.values] == pt]; bg = []
        for s_, n in pd.Series(strat[cb.bin.values]).value_counts().items():
            c = pl[strat[pl.bin.values] == s_]
            if len(c): bg.append(c.sample(min(2 * n, len(c)), random_state=int(rng.integers(1e9))))
        bg = pd.concat(bg) if bg else pl.head(0); d = f"{G}/pools/{r.pair}/{pt}"; os.makedirs(d, exist_ok=True)
        wbed(f"{d}/cob.bed", cb, 100); wbed(f"{d}/bg.bed", bg, 100); wbed(f"{d}/aonly.bed", aon[part[aon.bin.values] == pt], 150); wbed(f"{d}/cob_300.bed", cb, 150)
        info[f"{pt}_cob"] = len(cb); info[f"{pt}_bg"] = len(bg); info[f"{pt}_aonly"] = int((part[aon.bin.values] == pt).sum())
    rows.append(info)
    if info["disc_cob"] >= 100 and info["val_cob"] >= 50: tasks.append(r.pair)
P = pd.DataFrame(rows); P.to_csv(f"{G}/pairs.tsv", sep="\t", index=False); open(f"{G}/tasks.tsv", "w").write("\n".join(tasks) + "\n")
print(len(P), "pairs;", len(tasks), "with >= 100 discovery and >= 50 held-out co-bound sites"); print(P.groupby("set")[["disc_cob", "val_cob"]].median().to_string()); print(P.set.value_counts().to_dict(), pd.Series([P.set[P.pair == t].iloc[0] for t in tasks]).value_counts().to_dict())
