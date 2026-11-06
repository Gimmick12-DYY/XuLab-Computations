"""Shared-edge modules: connected components of clique edges shared by >=2 TFs."""
import sys, pandas as pd, networkx as nx
d = pd.read_csv(sys.argv[1], sep="\t")
d["tfl"] = d.tfs.str.split(",").apply(lambda l: list(l))
d = d[d.tfl.str.len() >= 2]
G = nx.Graph()
prom = {}
for r in d.itertuples():
    G.add_edge(r.region1, r.region2, tfs=set(r.tfl), dist=r.distance_kb, loop=r._6)
    for reg, g in ((r.region1, r.region1_promoter_gene_ids), (r.region2, r.region2_promoter_gene_ids)):
        prom[reg] = "" if pd.isna(g) else g
rows = []
for comp in nx.connected_components(G):
    H = G.subgraph(comp)
    tfs = set().union(*[e["tfs"] for _, _, e in H.edges(data=True)])
    regs = sorted(comp, key=lambda x: (x.split(":")[0], int(x.split(":")[1].split("-")[0])))
    pos = [int(x.split(":")[1].split("-")[0]) for x in regs]
    tfn = {}
    for _, _, e in H.edges(data=True):
        for t in e["tfs"]:
            tfn[t] = tfn.get(t, 0) + 1
    rows.append({"chrom": regs[0].split(":")[0], "start": min(pos), "end": max(int(x.split("-")[1]) for x in regs),
                 "n_regions": len(regs), "n_shared_edges": H.number_of_edges(), "n_tfs": len(tfs),
                 "n_edges_on_loop": sum(e["loop"] for _, _, e in H.edges(data=True)),
                 "max_edge_kb": max(e["dist"] for _, _, e in H.edges(data=True)),
                 "promoter_gene_ids": ",".join(sorted({g for x in regs for g in prom[x].split(",") if g})),
                 "regions": ";".join(regs), "tf_edges": ",".join(f"{t}({n})" for t, n in sorted(tfn.items(), key=lambda z: -z[1]))})
M = pd.DataFrame(rows).sort_values(["n_tfs", "n_shared_edges"], ascending=False)
M.to_csv(sys.argv[2], sep="\t", index=False)
print(len(M), "modules;", (M.n_tfs >= 5).sum(), ">=5 TFs;", (M.n_tfs >= 10).sum(), ">=10 TFs")
print(M.head(20).drop(columns=["regions", "tf_edges"]).to_string(index=False))
