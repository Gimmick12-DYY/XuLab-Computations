# Summarise ABC runs (power-law primary, WT Hi-C 5 kb side arm): links at ABC >= 0.02, links per gene, distance, distal coverage,
# known loci (MYC, 16p11.2), and power-law vs Hi-C agreement
import sys
import numpy as np, pandas as pd
A = sys.argv[1]
cls = pd.read_csv(f"{A}/inputs/candidate_class.tsv", sep="\t").set_index("name")["class"]
ngenes = sum(1 for _ in open(f"{A}/inputs/genes_pc_v46.bed"))
P16 = ["KCTD13", "MAPK3", "TAOK2", "ALDOA", "PPP4C", "SEZ6L2", "ASPHD1", "YPEL3", "TBX6", "HIRIP3", "INO80E", "DOC2A", "CDIPT",
       "MVP", "KIF22", "MAZ", "PRRT2", "PAGR1", "CORO1A", "QPRT", "GDPD3", "TMEM219", "FAM57B", "ZG16"]
runs = {}
for run in ["powerlaw", "hic5kb"]:
    f = f"{A}/results/Predictions_{run}/EnhancerPredictionsFull_thr0.02.tsv"
    try: d = pd.read_csv(f, sep="\t")
    except FileNotFoundError: print(run, "missing"); continue
    d["elem_class"] = d["name"].str.split("|").str[-1].map(cls) if d["name"].str.contains(r"\|").any() else d["name"].map(cls)
    if d.elem_class.isna().all():  # ABC names enhancers as chr:start-end; recover class by coordinates
        c = pd.read_csv(f"{A}/inputs/candidates_active.bed", sep="\t", header=None, names=["chr", "start", "end", "eid"])
        c["key"] = c.chr + ":" + c.start.astype(str) + "-" + c.end.astype(str); m = dict(zip(c.key, c.eid))
        d["eid"] = (d.chr + ":" + d.start.astype(str) + "-" + d.end.astype(str)).map(m); d["elem_class"] = d.eid.map(cls)
    self_p = d.isSelfPromoter.astype(bool) if "isSelfPromoter" in d else np.zeros(len(d), bool)
    dist = d[~self_p & (d["class"] != "promoter")]
    lpg = d[~self_p].groupby("TargetGene").size()
    q = dist.distance.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(-2).astype(int).to_dict()
    print(f"\n== {run}: {len(d)} links (incl. {self_p.sum()} self-promoter); non-self {(~self_p).sum()}; "
          f"distal (enhancer not a promoter element) {len(dist)}")
    print(f"   genes with >=1 non-self link {lpg.size}/{ngenes}; links per linked gene median {lpg.median():.0f}, mean {lpg.mean():.1f}, max {lpg.max()}")
    print(f"   genes with >=1 distal link {dist.TargetGene.nunique()} ({dist.TargetGene.nunique() / ngenes:.1%} of protein-coding)")
    print(f"   distal distance quantiles (bp) {q}; >100 kb {(dist.distance > 1e5).mean():.1%}, >500 kb {(dist.distance > 5e5).mean():.1%}")
    print(f"   element classes among distal links {dist.elem_class.value_counts(normalize=True).round(3).to_dict()}")
    print(f"   ABC class column {d['class'].value_counts().to_dict()}")
    for g in ["MYC"] + P16:
        x = dist[dist.TargetGene == g].sort_values("ABC.Score", ascending=False)
        if g == "MYC" or len(x):
            print(f"   {g}: {len(x)} distal links; top " + "; ".join(f"{r.chr}:{r.start}-{r.end} {int(r.distance / 1e3)} kb ABC {r['ABC.Score']:.3f}"
                                                          for _, r in x.head(3).iterrows()))
    runs[run] = d.assign(key=d.chr + ":" + d.start.astype(str) + "-" + d.end.astype(str) + ">" + d.TargetGene)
    s = d[~self_p].groupby("TargetGene").size()
    print("   16p11.2 genes with links:", {g: int(s.get(g, 0)) for g in P16 if g in set(d.TargetGene) or True})
if len(runs) == 2:
    a, b = set(runs["powerlaw"].key), set(runs["hic5kb"].key)
    print(f"\npower-law vs Hi-C links: {len(a & b)} shared, {len(a - b)} power-law only, {len(b - a)} Hi-C only; Jaccard {len(a & b) / len(a | b):.2f}")
