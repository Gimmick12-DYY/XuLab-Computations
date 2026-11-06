"""Parallel check only: ROSE-style SE candidate calling from HEK293T H3K27ac (GSE246314, 3 untreated replicates, MACS3 narrowPeak, hg38), same rules as
se_hub/scripts/01_se_candidates.py. Consensus = regions covered by peaks in >= 2 of 3 replicates; signal of a consensus peak = mean signalValue of overlapping peaks.
Each replicate is also called alone. Compared to the 342 ENCODE HEK293 calls and the SE_0027 hub elements."""
import glob, gzip, re, sys
from pathlib import Path
import numpy as np, pandas as pd
O = Path(sys.argv[1]); X = sys.argv[2]; SEH = Path(sys.argv[3])
tss = []
with gzip.open(X, "rt") as g:
    for ln in g:
        if ln.startswith("#"): continue
        x = ln.split("\t", 9)
        if x[2] != "gene" or not re.search(r'gene_type "(protein_coding|lncRNA)"', x[8]): continue
        tss.append((x[0], int(x[3]) if x[6] == "+" else int(x[4])))
TSS = pd.DataFrame(tss, columns=["chrom", "pos"]); by = {c: np.sort(d.pos.values) for c, d in TSS.groupby("chrom")}
def inside(c, s, e):
    a = by.get(c)
    return False if a is None else bool(((a - 2500 <= s) & (a + 2500 >= e)).any())
def call(P):
    P = P[P.chrom.str.match(r"^chr[0-9XY]+$")].copy(); P["prom"] = [inside(c, s, e) for c, s, e in zip(P.chrom, P.s, P.e)]; rows = []
    for c, d in P[~P.prom].sort_values(["chrom", "s"]).groupby("chrom"):
        cs, ce, sc = None, None, 0.0
        for s, e, g in zip(d.s, d.e, d.sig):
            if cs is not None and s - ce <= 12500: ce = max(ce, e); sc += g * (e - s)
            else:
                if cs is not None: rows.append((c, cs, ce, sc))
                cs, ce, sc = s, e, g * (e - s)
        rows.append((c, cs, ce, sc))
    S = pd.DataFrame(rows, columns=["chrom", "start", "end", "score"]).sort_values("score").reset_index(drop=True)
    y = S.score.values / S.score.max() * len(S); gr = np.gradient(y)
    cut = int(np.argmin(np.abs(gr - 1) + (gr < 1) * 1e9)) if (gr >= 1).any() else len(S) - 1
    SE = S.iloc[cut:].sort_values("score", ascending=False).reset_index(drop=True); SE["rank"] = np.arange(1, len(SE) + 1); return P, S, SE
files = sorted(glob.glob(str(O / "raw/*H3K27ac_CTRL_Rep*.narrowPeak.gz"))); assert len(files) == 3, files
R = [pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2, 6], names=["chrom", "s", "e", "sig"]) for f in files]
for f, r in zip(files, R): print(Path(f).name, len(r), "median width", int((r.e - r.s).median()), "median signal", round(r.sig.median(), 2))
cons = []
for c in sorted(set(R[0].chrom) | set(R[1].chrom) | set(R[2].chrom)):
    ev = []
    for k, r in enumerate(R):
        d = r[r.chrom == c].sort_values("s"); ms, me = None, None
        for s, e in zip(d.s, d.e):   # merge within replicate so each counts once
            if ms is not None and s <= me: me = max(me, e)
            else:
                if ms is not None: ev += [(ms, 1), (me, -1)]
                ms, me = s, e
        if ms is not None: ev += [(ms, 1), (me, -1)]
    ev.sort(key=lambda t: (t[0], t[1])); n = 0; st = None
    for p, dl in ev:
        n2 = n + dl
        if n < 2 <= n2: st = p
        if n >= 2 > n2 and st is not None and p > st: cons.append((c, st, p))
        n = n2
C = pd.DataFrame(cons, columns=["chrom", "s", "e"]); sig = np.zeros(len(C)); cnt = np.zeros(len(C))
for r in R:
    for c, gi in C.groupby("chrom").groups.items():
        d = r[r.chrom == c].sort_values("s"); a = C.loc[gi]
        for j, s, e in zip(a.index, a.s, a.e):
            i0 = np.searchsorted(d.e.values, s, side="right"); i1 = np.searchsorted(d.s.values, e, side="left")
            if i1 > i0: sig[j] += d.sig.values[i0:i1].max(); cnt[j] += 1
C["sig"] = sig / np.maximum(cnt, 1)
EN = pd.read_csv(SEH / "step1/HEK293_SE_rose_hg38.tsv", sep="\t"); EL = pd.read_csv(SEH / "model/inputs/hub_elements.tsv", sep="\t")
def ov(A, B):
    h = np.zeros(len(A), bool)
    for c, d in A.groupby("chrom"):
        b = B[B.chrom == c]
        for j, s, e in zip(d.index, d.start, d.end): h[A.index.get_loc(j)] = ((b.start < e) & (b.end > s)).any()
    return h
pse = lambda SE, c, s, e: SE[(SE.chrom == c) & (SE.start < e) & (SE.end > s)]
Enh = pd.DataFrame({"chrom": ["chr19"], "s": [1203658], "e": [1267059]})
res = {}
for name, P0 in [("consensus_2of3", C)] + [(f"rep{k + 1}", r) for k, r in enumerate(R)]:
    P, S, SE = call(P0); res[name] = SE
    hit = pse(SE, "chr19", 1203658, 1267059); h = ov(EN, SE); h2 = ov(SE, EN.rename(columns={}))
    print(f"\n{name}: peaks {len(P)} (promoter-contained {int(P.prom.sum())}); stitched {len(S)}; SE candidates {len(SE)}; median {int((SE.end - SE.start).median())} bp")
    print(f"  ENCODE 342 overlapping a 293T call: {int(h.sum())}; 293T calls overlapping an ENCODE call: {int(h2.sum())}/{len(SE)}; ENCODE top-50 overlapped: {int(h[np.argsort(EN['rank'].values)][:50].sum())}")
    print("  SE_0027 window hits:", hit[["start", "end", "score", "rank"]].round(0).to_dict("records"))
    st = S[(S.chrom == "chr19") & (S.start < 1267059) & (S.end > 1203658)].copy()
    if len(st): st["rank_all"] = len(S) - np.searchsorted(np.sort(S.score.values), st.score.values, side="left"); print("  stitched regions in window (rank among all stitched):", st[["start", "end", "score", "rank_all"]].round(0).to_dict("records"), "cutoff score", round(SE.score.min()))
SE = res["consensus_2of3"]; SE.to_csv(O / "HEK293T_SE_candidates_consensus_hg38.tsv", sep="\t", index=False)
mid = lambda r: tuple(map(int, r.split(":")[1].split("-")))
print("\nhub elements (consensus):")
for r in EL.itertuples():
    s, e = mid(r.element); c = C[(C.chrom == "chr19") & (C.s < e) & (C.e > s)]; h = pse(SE, "chr19", s, e)
    print(f"  {r.label:26s} H3K27ac peak: {'yes' if len(c) else 'no ':3s} max signal {c.sig.max() if len(c) else 0:6.1f}  in 293T SE cand.: {'rank ' + str(int(h['rank'].iloc[0])) if len(h) else 'no'}")
