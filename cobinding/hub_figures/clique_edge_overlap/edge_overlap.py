"""Descriptive overlap of clique EDGES across TFs (all cliques, blacklist removed).

Shared region set: union of all TFs' Cicero input peaks, merged where they overlap (no gap merging).
Each clique edge (pair of regions inside one clique) is mapped to a pair of shared-region ids; two TFs
"share an edge" when both have a clique edge joining the same two shared regions.
Outputs: per-TF summary, edges by number of TFs, TF x TF shared-edge counts, top shared edges.
"""
import gzip, sys
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd

X = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
inputs = [l.split("\t") for l in Path(sys.argv[3]).read_text().splitlines()]
TSS = Path(sys.argv[4]); BL = Path(sys.argv[5]); LOOPS = sys.argv[6:8]
KEEP = {f"chr{i}" for i in range(1, 23)} | {"chrX"}

def parse(r):
    r = r.strip()
    if ":" in r:
        c, se = r.split(":"); s, e = se.split("-")
    else:
        c, s, e = r.split("_")
    return c, int(s), int(e)

peaks, cl = {}, {}
for tf, cpath, rpath in inputs:
    with gzip.open(X / rpath.lstrip("/"), "rt") as fh:
        peaks[tf] = [p for p in (parse(x) for x in fh if x.strip()) if p[0] in KEEP]
    d = pd.read_csv(X / cpath.strip().lstrip("/"), sep="\t", usecols=["clique_id", "regions"])
    cl[tf] = [(str(r.clique_id), [parse(x) for x in str(r.regions).split(";")]) for r in d.itertuples()]
tfs = sorted(peaks)
allp = sorted({p for v in peaks.values() for p in v} | {r for v in cl.values() for _, rr in v for r in rr if r[0] in KEEP})
uni = []
for c, s, e in allp:
    if uni and uni[-1][0] == c and s <= uni[-1][2]:
        uni[-1][2] = max(uni[-1][2], e)
    else:
        uni.append([c, s, e])
U = pd.DataFrame(uni, columns=["chrom", "start", "end"])
starts = {c: g.start.values for c, g in U.groupby("chrom")}
offs = {c: g.index.values for c, g in U.groupby("chrom")}
def uid(r):
    c, s, e = r
    i = np.searchsorted(starts[c], s, side="right") - 1
    return int(offs[c][i])
blk = set()
bl = defaultdict(list)
for ln in gzip.open(BL, "rt"):
    f = ln.split(); bl[f[0]].append((int(f[1]), int(f[2])))
for i, (c, s, e) in enumerate(uni):
    if any(a < e and b > s for a, b in bl.get(c, [])):
        blk.add(i)
name = lambda i: f"{uni[i][0]}:{uni[i][1]}-{uni[i][2]}"

edge_tfs = defaultdict(set); edge_cliques = defaultdict(list); rows = []
for tf in tfs:
    n_cl = n_e = 0; regs = set(); es = set()
    for cid, rr in cl[tf]:
        if rr[0][0] not in KEEP:
            continue
        ids = sorted({uid(r) for r in rr})
        if any(i in blk for i in ids) or len(ids) < 2:
            continue
        n_cl += 1; regs.update(ids)
        for a, b in combinations(ids, 2):
            es.add((a, b)); edge_cliques[(a, b)].append(f"{tf}|{cid}")
    for e in es:
        edge_tfs[e].add(tf)
    rows.append({"tf": tf, "cliques": n_cl, "clique_regions": len(regs), "clique_edges": len(es), "_edges": es, "_regs": regs})
region_tfs = defaultdict(set)
for r in rows:
    for i in r["_regs"]:
        region_tfs[i].add(r["tf"])
for r in rows:
    es = r.pop("_edges"); rg = r.pop("_regs")
    r["edges_shared_with_another_TF"] = sum(len(edge_tfs[e]) > 1 for e in es)
    r["pct_edges_shared"] = 100 * r["edges_shared_with_another_TF"] / r["clique_edges"] if r["clique_edges"] else np.nan
    r["regions_shared_with_another_TF"] = sum(len(region_tfs[i]) > 1 for i in rg)
    r["pct_regions_shared"] = 100 * r["regions_shared_with_another_TF"] / r["clique_regions"] if r["clique_regions"] else np.nan
    r["cliques_with_a_shared_edge"] = 0
summ = pd.DataFrame(rows)
# TF x TF shared clique regions (for the Jaccard heatmap)
RM = pd.DataFrame(0, index=tfs, columns=tfs)
for i, ts in region_tfs.items():
    ts = sorted(ts)
    for p_, q_ in combinations(ts, 2):
        RM.loc[p_, q_] += 1; RM.loc[q_, p_] += 1
for r in rows:
    RM.loc[r["tf"], r["tf"]] = r["clique_regions"]
RM.to_csv(OUT / "tf_x_tf_shared_clique_regions.tsv", sep="\t")
# cliques that contain at least one edge seen in another TF
for tf in tfs:
    k = 0
    for cid, rr in cl[tf]:
        if rr[0][0] not in KEEP:
            continue
        ids = sorted({uid(r) for r in rr})
        if any(i in blk for i in ids) or len(ids) < 2:
            continue
        if any(len(edge_tfs[(a, b)]) > 1 for a, b in combinations(ids, 2)):
            k += 1
    summ.loc[summ.tf == tf, "cliques_with_a_shared_edge"] = k
summ["pct_cliques_with_shared_edge"] = 100 * summ.cliques_with_a_shared_edge / summ.cliques.replace(0, np.nan)
summ = summ.sort_values("clique_edges", ascending=False)
summ.to_csv(OUT / "per_tf_edge_overlap_summary.tsv", sep="\t", index=False)

# edges by number of TFs, regions by number of TFs
ek = Counter(len(v) for v in edge_tfs.values()); rk = Counter(len(v) for v in region_tfs.values())
pd.DataFrame({"n_tfs": sorted(set(ek) | set(rk))}).assign(
    edges=lambda d: d.n_tfs.map(ek).fillna(0).astype(int), regions=lambda d: d.n_tfs.map(rk).fillna(0).astype(int)
).to_csv(OUT / "edges_and_regions_by_n_tfs.tsv", sep="\t", index=False)

# TF x TF shared edges
M = pd.DataFrame(0, index=tfs, columns=tfs)
for e, ts in edge_tfs.items():
    ts = sorted(ts)
    for a, b in combinations(ts, 2):
        M.loc[a, b] += 1; M.loc[b, a] += 1
for t in tfs:
    M.loc[t, t] = int(summ.loc[summ.tf == t, "clique_edges"].iloc[0])
M.to_csv(OUT / "tf_x_tf_shared_clique_edges.tsv", sep="\t")

# loops (0.7) for annotation of shared edges
lp = defaultdict(list)
for f in LOOPS:
    for ln in open(f):
        x = ln.split()
        if len(x) >= 6 and x[0] == x[3]:
            a, b = (int(x[1]), int(x[2])), (int(x[4]), int(x[5]))
            if a > b: a, b = b, a
            lp[x[0]].append((a[0], a[1], b[0], b[1]))
def on_loop(i, j, slop=30000):
    c = uni[i][0]
    (s1, e1), (s2, e2) = sorted([(uni[i][1], uni[i][2]), (uni[j][1], uni[j][2])])
    return any(as_ <= e1 + slop and ae >= s1 - slop and bs <= e2 + slop and be >= s2 - slop for as_, ae, bs, be in lp.get(c, []))
tss = defaultdict(list)
for ln in gzip.open(TSS, "rt"):
    f = ln.split(); tss[f[0]].append((int(f[1]), f[3].split(".")[0]))
def prom(i):
    c, s, e = uni[i]
    g = [gid for p, gid in tss.get(c, []) if s - 1000 <= p <= e + 1000]
    return ",".join(sorted(set(g))) if g else ""
top = sorted(((len(v), e) for e, v in edge_tfs.items() if len(v) > 1), reverse=True)
trows = []
for k, (a, b) in top:
    trows.append({"region1": name(a), "region2": name(b), "distance_kb": round(abs((uni[b][1] + uni[b][2]) - (uni[a][1] + uni[a][2])) / 2000, 1),
                  "n_tfs": k, "tfs": ",".join(sorted(edge_tfs[(a, b)])), "on_loop_0.7": int(on_loop(a, b)),
                  "region1_promoter_gene_ids": prom(a), "region2_promoter_gene_ids": prom(b),
                  "cliques": ";".join(edge_cliques[(a, b)])})
pd.DataFrame(trows).to_csv(OUT / "top_shared_clique_edges.tsv", sep="\t", index=False)
shared = [e for e, v in edge_tfs.items() if len(v) > 1]
print(f"TFs {len(tfs)}; distinct clique edges {len(edge_tfs):,}; shared by >=2 TFs {len(shared):,} ({100*len(shared)/len(edge_tfs):.1f}%)")
print("edges by n TFs:", dict(sorted(ek.items())))
print(summ[["tf","cliques","clique_edges","pct_edges_shared","pct_cliques_with_shared_edge","pct_regions_shared"]].round(1).head(25).to_string(index=False))
print(pd.DataFrame(trows).head(12)[["region1","region2","distance_kb","n_tfs","on_loop_0.7","tfs"]].to_string(index=False))
