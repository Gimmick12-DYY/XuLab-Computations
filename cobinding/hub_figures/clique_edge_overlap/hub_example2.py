"""Hub example, one arc per shared clique edge (>25 kb, >=MINT TFs) in one connected hub; arc width =
number of TFs sharing the edge, red = on a 0.7 Hi-C loop (±30 kb). Below: which TFs' shared edges touch each element."""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx, numpy as np, pandas as pd
plt.rcParams.update({"font.size": 12, "svg.fonttype": "none"})
seed_region, MINT, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sym = pd.read_csv("chord/ensg_symbol.tsv", sep="\t", header=None, index_col=0)[1].astype(str).to_dict()
d = pd.read_csv("edges/top_shared_clique_edges.tsv", sep="\t")
d["tfl"] = d.tfs.str.split(",").apply(lambda l: list(l)); d["n"] = d.tfl.str.len()
L = d[(d.distance_kb > 25) & (d.n >= MINT)]
G = nx.Graph()
gene = {}
for r in L.itertuples():
    G.add_edge(r.region1, r.region2, n=r.n, loop=r._6, tfs=r.tfl, kb=r.distance_kb)
    for reg, g in ((r.region1, r.region1_promoter_gene_ids), (r.region2, r.region2_promoter_gene_ids)):
        if isinstance(g, str):
            gene[reg] = "/".join(sym.get(x, x) for x in g.split(",") if sym.get(x, x) != "nan")
comp = nx.node_connected_component(G, seed_region)
H = G.subgraph(comp)
pos = lambda r: (int(r.split(":")[1].split("-")[0]) + int(r.split("-")[1])) / 2
nodes = sorted(comp, key=pos)
chrom = nodes[0].split(":")[0]
tfs = sorted({t for *_, e in H.edges(data=True) for t in e["tfs"]})
M = pd.DataFrame(0, index=tfs, columns=nodes)
for a, b, e in H.edges(data=True):
    for t in e["tfs"]:
        M.loc[t, a] += 1; M.loc[t, b] += 1
M = M.loc[M.sum(1).sort_values(ascending=False).index]
lo, hi = pos(nodes[0]), pos(nodes[-1]); pad = 0.04 * (hi - lo); lo -= pad; hi += pad
fig = plt.figure(figsize=(14, 4.2 + 0.22 * len(tfs)))
hA = 3.6 / (4.2 + 0.22 * len(tfs))
ax = fig.add_axes([0.13, 1 - hA, 0.84, hA - 0.02])
axm = fig.add_axes([0.13, 0.06, 0.84, 1 - hA - 0.09], sharex=ax)
for a, b, e in sorted(H.edges(data=True), key=lambda z: z[2]["n"]):
    x1, x2 = sorted((pos(a), pos(b)))
    t = np.linspace(0, np.pi, 80)
    xs = (x1 + x2) / 2 - (x2 - x1) / 2 * np.cos(t); ys = (x2 - x1) / (hi - lo) * np.sin(t)
    ax.plot(xs, ys, color="#C0392B" if e["loop"] else "#4A6FA5", lw=0.8 + 0.35 * e["n"], alpha=0.85, solid_capstyle="round")
for k, r in enumerate(nodes):
    x = pos(r)
    ax.plot([x], [0], "s", color="#222", ms=6)
    if r in gene and (k == 0 or gene.get(nodes[k - 1]) != gene[r]):
        ax.text(x, 0.03, gene[r], rotation=90, ha="center", va="bottom", fontsize=10.5, fontweight="bold")
ax.set_xlim(lo, hi); ax.set_ylim(-0.03, max(0.25, ax.get_ylim()[1]) * 1.12)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_position(("data", -0.02)); ax.set_yticks([])
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e6:.2f}"))
ax.tick_params(axis="x", labelsize=10)
ax.text(lo, ax.get_ylim()[1], f"{chrom}:{lo / 1e6:.2f}–{hi / 1e6:.2f} Mb · {len(nodes)} elements · {H.number_of_edges()} shared edges >25 kb (≥{MINT} TFs each) · {len(tfs)} TFs",
        ha="left", va="top", fontsize=12)
ax.plot([], [], color="#4A6FA5", lw=3, label="shared clique edge"); ax.plot([], [], color="#C0392B", lw=3, label="… also on a 0.7 Hi-C loop")
ax.plot([], [], color="#888", lw=0.8 + 0.35 * 3, label="width: 3 TFs"); ax.plot([], [], color="#888", lw=0.8 + 0.35 * 12, label="width: 12 TFs")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False, fontsize=10.5)
# lower panel: one row per TF, a tick at each element (genome position) that TF's shared edges touch
for k, t in enumerate(M.index):
    xs = [pos(r) for r in nodes if M.loc[t, r] > 0]
    axm.scatter(xs, [k] * len(xs), marker="|", s=170, linewidths=2.4, color="#2E7D6B")
for r in nodes:
    axm.axvline(pos(r), color="#E6E6E6", lw=0.8, zorder=0)
axm.set_ylim(len(M) - 0.5, -0.5); axm.set_xlim(lo, hi)
axm.set_yticks(range(len(M))); axm.set_yticklabels(M.index, fontsize=10)
axm.tick_params(axis="x", labelsize=11)
axm.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e6:.2f}"))
axm.set_xlabel(f"{chrom} position (Mb, hg38)", fontsize=12)
for sp in ("top", "right"):
    axm.spines[sp].set_visible(False)

ax.tick_params(axis="x", bottom=False, labelbottom=False); ax.spines["bottom"].set_visible(False)
ax.axhline(0, color="#999", lw=0.8)
fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(f"{out}.svg", bbox_inches="tight", facecolor="white")
rows = [{"region1": a, "region2": b, "gene1": gene.get(a, "distal"), "gene2": gene.get(b, "distal"), "kb": e["kb"], "n_tfs": e["n"],
         "on_loop_0.7": e["loop"], "tfs": ",".join(e["tfs"])} for a, b, e in H.edges(data=True)]
pd.DataFrame(rows).sort_values("n_tfs", ascending=False).to_csv(f"{out}_edges.tsv", sep="\t", index=False)
print(len(nodes), "elements", H.number_of_edges(), "edges", len(tfs), "TFs; loops", sum(e["loop"] for *_, e in H.edges(data=True)))
print(pd.DataFrame(rows).sort_values("n_tfs", ascending=False).head(12).drop(columns="tfs").to_string(index=False))
