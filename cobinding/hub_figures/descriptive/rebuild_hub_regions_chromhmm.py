"""Rebuild hub_regions_chromhmm.tsv.gz (lost with a scratch folder) with the same rule as annot_chromhmm.py:
promoter = RefSeq Select TSS ±1 kb, otherwise the HEK293T 18-state ChromHMM state at the region midpoint, grouped.
Inputs in chromhmm_rebuild_inputs/: hub_regions.tsv (ourrun_k50_clique_hubs_blacklist_removed), HEK293T_chromHMM18.bed.gz, rs/<chrom>.json (UCSC ncbiRefSeqSelect, hg38)."""
import gzip, json, glob, re
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; I = HERE / "chromhmm_rebuild_inputs"
GROUP = {"EnhA1": "active enhancer", "EnhA2": "active enhancer", "EnhG1": "active enhancer", "EnhG2": "active enhancer",
         "EnhWk": "weak/poised enhancer", "EnhBiv": "weak/poised enhancer",
         "TssA": "other TSS/flank", "TssFlnk": "other TSS/flank", "TssFlnkU": "other TSS/flank", "TssFlnkD": "other TSS/flank", "TssBiv": "other TSS/flank",
         "Tx": "transcribed", "TxWk": "transcribed",
         "ReprPC": "repressed/het", "ReprPCWk": "repressed/het", "Het": "repressed/het", "ZNF/Rpts": "repressed/het", "Quies": "quiescent"}
P = {}
for f in glob.glob(str(I / "rs/*.json")):
    v = json.load(open(f))["ncbiRefSeqSelect"]
    if v: P[Path(f).stem] = np.sort([(g["txStart"] if g["strand"] == "+" else g["txEnd"]) for g in v])
H = defaultdict(list)
with gzip.open(I / "HEK293T_chromHMM18.bed.gz", "rt") as fh:
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
Hh = pd.read_csv(I / "hub_regions.tsv", sep="\t", usecols=["hub_region", "n_tfs"])
Hh["cls"] = [cls2(r.split(":")[0], *map(int, r.split(":")[1].split("-"))) for r in Hh.hub_region]
Hh.to_csv(HERE / "hub_regions_chromhmm.tsv.gz", sep="\t", index=False)
# check against the old figure: the percentage labels (segments >= 7%) printed in figC1_class_bars.svg, in drawing order
CLASSES = ["promoter", "enhancer", "other TSS/flank", "transcribed", "repressed/het", "quiescent"]
MERGE = {"active enhancer": "enhancer", "weak/poised enhancer": "enhancer"}
A = pd.read_csv(HERE / "annotation_counts_chromhmm.tsv", sep="\t"); A["class"] = A["class"].replace(MERGE); A = A.groupby(["tf", "class"], as_index=False)[["peaks", "clique_regions"]].sum()
Hc = Hh.copy(); Hc["cls"] = Hc.cls.replace(MERGE); Hc["bin"] = pd.cut(Hc.n_tfs, [0, 1, 2, 4, 9, 100], labels=["1", "2", "3–4", "5–9", "≥10"])
bars = {"peaks": A.groupby("class").peaks.sum(), "cliques": A.groupby("class").clique_regions.sum()}; ns = {}
for b, d in Hc.groupby("bin", observed=True): bars[b] = d.cls.value_counts(); ns[b] = len(d)
B = pd.DataFrame(bars).reindex(CLASSES).fillna(0); B = B / B.sum()
new = [f"{v:.0%}" for k in CLASSES for v in B.loc[k] if v >= 0.07]
old = re.findall(r">\s*(\d+%)\s*<", open(HERE / "figC1_class_bars.svg").read())
oldn = re.findall(r"n = ([\d,]+)", open(HERE / "figC1_class_bars.svg").read())
newn = [f"{int(A.peaks.sum()):,}", f"{int(A.clique_regions.sum()):,}"] + [f"{ns[b]:,}" for b in ns]
print("old labels", old); print("new labels", new); print("old n", oldn); print("new n", newn)
print("MATCH" if old == new and oldn == newn else "MISMATCH")
print(B.round(4).to_string())
