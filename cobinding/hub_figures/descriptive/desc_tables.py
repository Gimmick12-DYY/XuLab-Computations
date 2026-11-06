"""Per-TF descriptive tables: peaks, cliques, clique sizes/spans, genomic class of peaks and clique regions."""
import gzip, json, glob
from pathlib import Path
import numpy as np, pandas as pd
S = Path("..")
KEEP = {f"chr{i}" for i in range(1, 23)} | {"chrX"}
genes = {}
for f in glob.glob(str(S / "goodex/rs/*.json")):
    c = Path(f).stem
    genes[c] = [(g["strand"], g["txStart"], g["txEnd"], list(zip(map(int, g["exonStarts"].strip(",").split(",")), map(int, g["exonEnds"].strip(",").split(",")))))
                for g in json.load(open(f))["ncbiRefSeqSelect"]]
P, X, T = {}, {}, {}
for c, v in genes.items():
    P[c] = np.sort([(t1 if s == "+" else t2) for s, t1, t2, _ in v])
    ex = sorted((a, b) for *_, e in v for a, b in e); X[c] = (np.array([a for a, _ in ex]), np.maximum.accumulate(np.array([b for _, b in ex])))
    tx = sorted((t1, t2) for _, t1, t2, _ in v); T[c] = (np.array([a for a, _ in tx]), np.maximum.accumulate(np.array([b for _, b in tx])))
def cls(c, s, e):
    if c not in P: return "intergenic"
    p = P[c]; i = np.searchsorted(p, s - 1000)
    if i < len(p) and p[i] <= e + 1000: return "promoter"
    a, b = X[c]; j = np.searchsorted(a, e)
    if j > 0 and b[j - 1] > s: return "exon"
    a, b = T[c]; j = np.searchsorted(a, e)
    if j > 0 and b[j - 1] > s: return "intron"
    return "intergenic"
def parse(x):
    x = x.strip()
    if ":" in x:
        c, se = x.split(":"); s, e = se.split("-")
    else:
        c, s, e = x.split("_")
    return c, int(s), int(e)
def main():
    rows, cl_rows, ann = [], [], []
    for ln in (S / "basic/inputs.tsv").read_text().splitlines():
        tf, cp, rp = ln.split("\t")
        with gzip.open(S / "strat/x" / rp.strip().lstrip("/"), "rt") as fh:
            pk = [p for p in (parse(x) for x in fh if x.strip()) if p[0] in KEEP]
        d = pd.read_csv(S / "strat/x" / cp.strip().lstrip("/"), sep="\t", usecols=["clique_id", "n_regions", "span_bp", "regions"])
        d = d[d.regions.str.split(":").str[0].isin(KEEP)]
        creg = sorted({parse(x) for rr in d.regions for x in rr.split(";")})
        pc = pd.Series([cls(*p) for p in pk]).value_counts()
        cc = pd.Series([cls(*p) for p in creg]).value_counts() if creg else pd.Series(dtype=int)
        for k in ("promoter", "exon", "intron", "intergenic"):
            ann.append({"tf": tf, "class": k, "peaks": int(pc.get(k, 0)), "clique_regions": int(cc.get(k, 0))})
        rows.append({"tf": tf, "cicero_peaks": len(pk), "cliques": len(d), "clique_regions": len(creg)})
        for r in d.itertuples():
            cl_rows.append({"tf": tf, "n_regions": r.n_regions, "span_kb": r.span_bp / 1e3})
        print(tf, len(pk), len(d), flush=True)
    pd.DataFrame(rows).to_csv("per_tf_counts.tsv", sep="\t", index=False)
    pd.DataFrame(cl_rows).to_csv("clique_sizes.tsv.gz", sep="\t", index=False)
    pd.DataFrame(ann).to_csv("annotation_counts.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
