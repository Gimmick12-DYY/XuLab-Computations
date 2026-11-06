"""Region classes: promoter = RefSeq Select TSS ±1 kb; otherwise the HEK293T 18-state ChromHMM state at the region midpoint, grouped."""
import gzip, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "."); from desc_tables import P, parse, KEEP, S  # RefSeq TSS arrays, region parser
GROUP = {"EnhA1": "active enhancer", "EnhA2": "active enhancer", "EnhG1": "active enhancer", "EnhG2": "active enhancer",
         "EnhWk": "weak/poised enhancer", "EnhBiv": "weak/poised enhancer",
         "TssA": "other TSS/flank", "TssFlnk": "other TSS/flank", "TssFlnkU": "other TSS/flank", "TssFlnkD": "other TSS/flank", "TssBiv": "other TSS/flank",
         "Tx": "transcribed", "TxWk": "transcribed",
         "ReprPC": "repressed/het", "ReprPCWk": "repressed/het", "Het": "repressed/het", "ZNF/Rpts": "repressed/het", "Quies": "quiescent"}
CLASSES = ["promoter", "active enhancer", "weak/poised enhancer", "other TSS/flank", "transcribed", "repressed/het", "quiescent"]
H = defaultdict(list)
with gzip.open("HEK293T_chromHMM18.bed.gz", "rt") as fh:
    for ln in fh:
        f = ln.split("\t"); H[f[0]].append((int(f[1]), int(f[2]), f[3]))
HS = {c: (np.array([a for a, _, _ in sorted(v)]), np.array([b for _, b, _ in sorted(v)]), [s for _, _, s in sorted(v)]) for c, v in H.items()}
def cls2(c, s, e):
    if c in P:
        p = P[c]; i = np.searchsorted(p, s - 1000)
        if i < len(p) and p[i] <= e + 1000: return "promoter"
    if c not in HS: return "quiescent"
    st, en, lab = HS[c]; m = (s + e) // 2; i = np.searchsorted(st, m, side="right") - 1
    return GROUP.get(lab[i], "quiescent") if i >= 0 and en[i] > m else "quiescent"
def main():
    rows = []
    for ln in (S / "basic/inputs.tsv").read_text().splitlines():
        tf, cp, rp = ln.split("\t")
        with gzip.open(S / "strat/x" / rp.strip().lstrip("/"), "rt") as fh:
            pk = [p for p in (parse(x) for x in fh if x.strip()) if p[0] in KEEP]
        d = pd.read_csv(S / "strat/x" / cp.strip().lstrip("/"), sep="\t", usecols=["regions"])
        d = d[d.regions.str.split(":").str[0].isin(KEEP)]
        creg = sorted({parse(x) for rr in d.regions for x in rr.split(";")})
        pc = pd.Series([cls2(*p) for p in pk]).value_counts(); cc = pd.Series([cls2(*p) for p in creg]).value_counts() if creg else pd.Series(dtype=int)
        for k in CLASSES:
            rows.append({"tf": tf, "class": k, "peaks": int(pc.get(k, 0)), "clique_regions": int(cc.get(k, 0))})
    pd.DataFrame(rows).to_csv("annotation_counts_chromhmm.tsv", sep="\t", index=False)
    Hh = pd.read_csv(S / "basic/hub_regions.tsv", sep="\t", usecols=["hub_region", "n_tfs"])
    Hh["cls"] = [cls2(r.split(":")[0], *map(int, r.split(":")[1].split("-"))) for r in Hh.hub_region]
    Hh.to_csv("hub_regions_chromhmm.tsv.gz", sep="\t", index=False)
    A = pd.DataFrame(rows).groupby("class")[["peaks", "clique_regions"]].sum()
    print((A / A.sum()).reindex(CLASSES).round(3)); print(pd.crosstab(pd.cut(Hh.n_tfs, [0, 1, 2, 4, 9, 100]), Hh.cls, normalize="index").round(3)[CLASSES])
if __name__ == "__main__":
    main()
