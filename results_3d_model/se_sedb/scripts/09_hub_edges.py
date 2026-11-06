"""All Cicero edges within the SE hub: the exact SE plus its 31 partners (edges_exact_se.tsv) as 10 kb-bin elements. An edge = two hub elements in the same TF's clique, > 25 kb; on_loop = on a WT Peakachu 0.7 loop (+-30 kb). Loop type by what it connects: promoter (both ends promoters), enhancer (either end the SE or an enhancer), other (the rest)."""
import glob, itertools, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
IN = sys.argv[1]; R0 = Path(__file__).resolve().parents[1]; PAD = 30000; C = "chr19"; S0, E0 = 1248039, 1254347
T = pd.read_csv(R0 / "edges_exact_se.tsv", sep="\t"); mid = lambda r: sum(map(int, r.split(":")[1].split("-"))) // 2; T["pos"] = [mid(x) for x in T.partner]; el = {p // 10000: (l, c, p) for p, l, c in zip(T.pos, T.label, T.cls)}; se_bins = set(range(S0 // 10000, (E0 - 1) // 10000 + 1)); L = []
for r_ in (5000, 10000):
    for ln in open(f"{IN}/loops/peakachu_{r_}.loops.0.7.bedpe"):
        x = ln.split()
        if x[0] == C and x[3] == C:
            a, b = sorted(((int(x[1]), int(x[2])), (int(x[4]), int(x[5])))); L.append((*a, *b))
L = np.array(L)
def on_loop(a, b): return bool(((L[:, 0] - PAD < a[1]) & (L[:, 1] + PAD > a[0]) & (L[:, 2] - PAD < b[1]) & (L[:, 3] + PAD > b[0])).any())
def pr(x): c, a = x.split(":"); s, e = map(int, a.split("-")); return s, e
def key(x):
    s, e = pr(x); m = (s + e) // 2
    if s < E0 and e > S0: return "SE"
    return m // 10000 if m // 10000 in el else None
ed = defaultdict(lambda: dict(tfs=set(), loop=False, a=None, b=None))
for f_ in sorted(glob.glob(f"{IN}/cl/*/cliques.tsv")):
    tf = Path(f_).parent.name; d = pd.read_csv(f_, sep="\t", usecols=["chromosome", "regions"]); d = d[d.chromosome == C]
    for x in d.itertuples():
        rg = [(key(u), u) for u in x.regions.split(";")]; rg = [q for q in rg if q[0] is not None]
        for (ka, a), (kb, b) in itertools.combinations(rg, 2):
            if ka == kb: continue
            sa, ea = pr(a); sb, eb = pr(b)
            if abs((sa + ea) - (sb + eb)) // 2 <= 25000: continue
            k = tuple(sorted((ka, kb), key=str)); t = ed[k]; t["tfs"].add(tf); ra, rb = (a, b) if k[0] == ka else (b, a); t["a"], t["b"] = t["a"] or ra, t["b"] or rb
            t["loop"] = t["loop"] or on_loop(*(((sa, ea), (sb, eb)) if sa < sb else ((sb, eb), (sa, ea))))
rows = []
for (ka, kb), t in ed.items():
    ca = "se" if ka == "SE" else el[ka][1]; cb = "se" if kb == "SE" else el[kb][1]; typ = "promoter" if ca == cb == "promoter" else "enhancer" if {"se", "enhancer"} & {ca, cb} else "other"
    rows.append(dict(a=ka, b=kb, label_a="super-enhancer" if ka == "SE" else el[ka][0], label_b="super-enhancer" if kb == "SE" else el[kb][0], cls_a=ca, cls_b=cb, type=typ, on_loop=int(t["loop"]), n_tfs=len(t["tfs"]), tfs=",".join(sorted(t["tfs"])), reg_a=t["a"], reg_b=t["b"], from_se=int("SE" in (ka, kb))))
H = pd.DataFrame(rows); H.to_csv(R0 / "hub_edges_all.tsv", sep="\t", index=False); pd.set_option("display.width", 200); print(len(H)); print(pd.crosstab([H.from_se, H.on_loop], H.type, margins=True)); print(H[H.from_se == 0].sort_values(["on_loop", "n_tfs"], ascending=False)[["label_a", "label_b", "type", "on_loop", "n_tfs"]].head(25).to_string(index=False))
