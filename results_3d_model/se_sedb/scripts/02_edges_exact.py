"""Edges from the exact SEdb SE SE_02_195100173 (chr19:1248039-1254347; anchors must overlap it, no pad). All Cicero clique partners > 25 kb, outside the SE, pooled by 10 kb bin: on_loop = on a WT Peakachu 0.7 loop (+-30 kb). Classes as in 01_screen_se.py."""
import glob, gzip, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
IN = sys.argv[1]; R0 = Path(__file__).resolve().parents[1]; I = R0 / "inputs"; PAD = 30000; C, S0, E0, SID = "chr19", 1248039, 1254347, "SE_02_195100173"
SE = pd.read_csv(R0 / "HEK293T_WT_SEdb_SE_02_1951_hg38.bed", sep="\t", header=None, names=["chrom", "start", "end", "se_id"]); h = SE[SE.chrom == C]
L = []
for r_ in (5000, 10000):
    for ln in open(f"{IN}/loops/peakachu_{r_}.loops.0.7.bedpe"):
        x = ln.split()
        if x[0] == C and x[3] == C:
            a, b = sorted(((int(x[1]), int(x[2])), (int(x[4]), int(x[5])))); L.append((*a, *b))
L = np.array(L)
def on_loop(a, b): return bool(((L[:, 0] - PAD < a[1]) & (L[:, 1] + PAD > a[0]) & (L[:, 2] - PAD < b[1]) & (L[:, 3] + PAD > b[0])).any())
def pr(x): c, a = x.split(":"); s, e = map(int, a.split("-")); return c, s, e
R = pd.read_csv(gzip.open(I / "ncbiRefSeqSelect.txt.gz", "rt"), sep="\t", header=None, usecols=[2, 3, 4, 5, 12], names=["c", "st", "s", "e", "g"]); R = R[R.c == C]; R["tss"] = np.where(R.st == "+", R.s, R.e); Cm = pd.read_csv(gzip.open(I / "HEK293T_chromHMM18.bed.gz", "rt"), sep="\t", header=None, usecols=[0, 1, 2, 3], names=["c", "s", "e", "st"]); Cm = Cm[(Cm.c == C) & Cm.st.str.contains("Enh")]
def cls(x):
    _, s, e = pr(x); g = R[(R.tss >= s - 2500) & (R.tss <= e + 2500)]
    if len(g): return "promoter", "/".join(sorted(set(g.g)))
    o = h[(h.end > s) & (h.start < e) & (h.se_id != SID)]; en = bool(((Cm.e > s) & (Cm.s < e)).any()); n = R.loc[(R.tss - (s + e) // 2).abs().idxmin()]
    return ("enhancer" if en or len(o) else "other"), f"{n.g} {'+' if n.tss < s else '-'}{abs(n.tss - (s + e) // 2) // 1000} kb"
ed = defaultdict(lambda: dict(tfs=set(), p=None, a=None))
for f_ in sorted(glob.glob(f"{IN}/cl/*/cliques.tsv")):
    tf = Path(f_).parent.name; d = pd.read_csv(f_, sep="\t", usecols=["chromosome", "regions"]); d = d[d.chromosome == C]
    for x in d.itertuples():
        rg = x.regions.split(";")
        for u in rg:
            _, s, e = pr(u)
            if not (s < E0 and e > S0): continue
            for v in rg:
                if v == u: continue
                _, s2, e2 = pr(v); mu, mv = (s + e) // 2, (s2 + e2) // 2
                if abs(mu - mv) <= 25000 or (s2 < E0 and e2 > S0): continue
                t = ed[mv // 10000]; t["tfs"].add(tf); t["p"] = t["p"] or v; t["a"] = t["a"] or u
rows = []
for b, t in sorted(ed.items()):
    _, s, e = pr(t["a"]); _, s2, e2 = pr(t["p"]); lo = on_loop(*(((s, e), (s2, e2)) if s < s2 else ((s2, e2), (s, e)))); c_, l_ = cls(t["p"]); rows.append(dict(anchor=t["a"], partner=t["p"], kb=abs((s + e) - (s2 + e2)) // 2000, n_tfs=len(t["tfs"]), tfs=",".join(sorted(t["tfs"])), on_loop=int(lo), cls=c_, label=l_))
T = pd.DataFrame(rows); T.to_csv(R0 / "edges_exact_se.tsv", sep="\t", index=False); pd.set_option("display.width", 200); print(T.drop(columns="anchor").to_string(index=False)); print(pd.crosstab(T.on_loop, T.cls)); print(pd.crosstab(T.on_loop, T.n_tfs >= 2))
