#!/usr/bin/env python3
"""Overlay Peakachu loops on the RBBP4 clique network (does not touch gold results)."""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

ROOT = Path("/work/users/d/y/dyy12/XuLab")
sys.path.insert(0, str(ROOT / "cobinding" / "scripts"))
import cluster_peaks as CP  # noqa: E402

EDGES = ROOT / "cobinding/work/RBBP4.peak_edges.tsv"
CLIQUES = ROOT / "cobinding/results/RBBP4/cliques.tsv"
NODES = ROOT / "cobinding/results/RBBP4/nodes.tsv"
# Span containment: both peaks inside the same Peakachu loop interval.
# (both-anchors table kept as RBBP4_pairs_as_loops.0.7.pad10kb.tsv — only 25 pink edges)
LOOP_PAIRS = ROOT / "hic/work/loops/union_5k_10k_0.7/RBBP4_pairs_in_loop_span.0.7.pad10kb.tsv"
OUT = ROOT / "hic/work/loops/union_5k_10k_0.7"


def load_graph() -> nx.Graph:
    df = pd.read_csv(EDGES, sep="\t")
    df = df[(df["qval"] <= 0.05) & (df["coaccess"] >= 0.0) & (df["n_links"] >= 1)].copy()
    dist_ok = []
    for a, b in zip(df["peak1"], df["peak2"]):
        d = CP.pair_distance_bp(str(a), str(b))
        dist_ok.append(d is not None and d <= 1_000_000)
    df = df.loc[np.asarray(dist_ok)]
    G = nx.Graph()
    for r in df.itertuples(index=False):
        G.add_edge(str(r.peak1), str(r.peak2))
    return G


def load_cliques() -> list[frozenset]:
    out = []
    with CLIQUES.open() as fh:
        h = fh.readline().rstrip("\n").split("\t")
        i = {x: k for k, x in enumerate(h)}
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            regs = frozenset(p[i["regions"]].split(";"))
            out.append(regs)
    return out


def load_genes() -> dict[str, set]:
    g: dict[str, set] = {}
    with NODES.open() as fh:
        h = fh.readline().rstrip("\n").split("\t")
        i = {x: k for k, x in enumerate(h)}
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            genes = p[i["genes"]]
            g[p[i["peak"]]] = set() if genes == "." else set(genes.split(","))
    return g


def load_loops() -> list[tuple[str, str, str]]:
    rows = []
    with LOOP_PAIRS.open() as fh:
        h = fh.readline().rstrip("\n").split("\t")
        i = {x: k for k, x in enumerate(h)}
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            a, b = p[i["peak1"]], p[i["peak2"]]
            res = p[i["resolution"]]
            if a > b:
                a, b = b, a
            rows.append((a, b, res))
    return rows


def bow(p, q, n: int = 14, amp: float = 0.14, max_off: float = 0.28):
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    length = float(np.linalg.norm(d)) or 1.0
    nrm = np.array([-d[1], d[0]], float) / length
    t = np.linspace(0.0, 1.0, n)
    scale = min(amp * length, max_off)
    off = scale * 4.0 * t * (1.0 - t)
    return p[None, :] + t[:, None] * d + off[:, None] * nrm


def pack_layout(H, comps, node_sep: float = 0.42):
    radii, locals_ = [], []
    for c in comps:
        loc = CP._component_layout(H.subgraph(c))
        loc = CP._spread_local(loc, min_d=node_sep)
        r = max((x * x + y * y) ** 0.5 for x, y in loc.values()) + node_sep * 0.8
        radii.append(r)
        locals_.append(loc)
    cx, cy = CP._pack_discs(radii, pad=0.85)
    pos = {}
    for i, loc in enumerate(locals_):
        for v, (x, y) in loc.items():
            pos[v] = (x + cx[i], y + cy[i])
    return pos, radii, cx, cy


def plot_network(G, cliques, genes, loops, comps, out_png: Path, title: str, subtitle: str,
                 label: bool = True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    keep = set().union(*comps) if comps else set()
    H = G.subgraph(keep)
    pos, radii, cx, cy = pack_layout(H, comps)

    live, cmap = CP._clique_colors(cliques, keep)
    node_col, ncl = {}, Counter()
    for i, c in sorted(enumerate(live), key=lambda kv: len(kv[1])):
        for r in c:
            if r in keep:
                node_col[r] = cmap[i]
                ncl[r] += 1

    fig_bg = "#f7f7f5"
    fig, ax = plt.subplots(figsize=(24, 24), facecolor=fig_bg)
    ax.set_facecolor(fig_bg)
    grey, black = "#c8ccd2", "#111111"
    clique_nodes = set().union(*cliques) if cliques else set()
    clique_edges = set()
    for c in cliques:
        for a, b in combinations(c, 2):
            if H.has_edge(a, b):
                clique_edges.add(frozenset((a, b)))

    rest_e = [(pos[u], pos[v]) for u, v in H.edges()
              if frozenset((u, v)) not in clique_edges and u in pos and v in pos]
    if rest_e:
        ax.add_collection(LineCollection(rest_e, colors=grey, linewidths=0.55,
                                         alpha=0.85, zorder=1))
    rest_n = [n for n in H if n not in clique_nodes]
    cliq_n = [n for n in H if n in clique_nodes]
    if rest_n:
        xy = np.array([pos[v] for v in rest_n])
        ax.scatter(xy[:, 0], xy[:, 1], s=16, c=grey, linewidths=0, zorder=2)
    for i, c in enumerate(live):
        segs = [(pos[a], pos[b]) for a, b in combinations(c, 2)
                if H.has_edge(a, b) and a in pos and b in pos]
        if segs:
            ax.add_collection(LineCollection(segs, colors=[cmap[i]],
                                             linewidths=2.2, alpha=1.0, zorder=3))
    if cliq_n:
        xy = np.array([pos[v] for v in cliq_n])
        hinge = [ncl.get(v, 1) > 1 for v in cliq_n]
        ax.scatter(xy[:, 0], xy[:, 1], s=28,
                   c=[node_col.get(v, black) for v in cliq_n],
                   linewidths=[0.9 if h else 0 for h in hinge],
                   edgecolors=["#111" if h else "none" for h in hinge],
                   zorder=4)

    other5, other10, cliq5, cliq10 = [], [], [], []
    other_nodes, cliq_loop_nodes = set(), set()
    n_drawn = n_skip = n_cross = n_cliq_loop = 0
    node_comp = {}
    for i, c in enumerate(comps):
        for v in c:
            node_comp[v] = i
    for a, b, res in loops:
        if a not in pos or b not in pos:
            n_skip += 1
            continue
        if node_comp.get(a) != node_comp.get(b):
            n_cross += 1
            continue
        n_drawn += 1
        poly = bow(pos[a], pos[b])
        if frozenset((a, b)) in clique_edges:
            n_cliq_loop += 1
            cliq_loop_nodes.update((a, b))
            (cliq5 if res == "5000" else cliq10).append(poly)
        else:
            other_nodes.update((a, b))
            (other5 if res == "5000" else other10).append(poly)

    if other5:
        ax.add_collection(LineCollection(other5, colors="#111111", linewidths=3.6, zorder=5))
        ax.add_collection(LineCollection(other5, colors="#F5C518", linewidths=1.9, zorder=6))
    if other10:
        ax.add_collection(LineCollection(other10, colors="#111111", linewidths=3.6, zorder=5))
        ax.add_collection(LineCollection(other10, colors="#1A6B8A", linewidths=1.9,
                                         linestyles="dashed", zorder=6))
    rest_loop = [v for v in other_nodes if v not in cliq_loop_nodes]
    if rest_loop:
        xy = np.array([pos[v] for v in rest_loop])
        ax.scatter(xy[:, 0], xy[:, 1], s=55, facecolors="none", edgecolors="#111",
                   linewidths=1.15, zorder=7)

    cliq_polys = cliq5 + cliq10
    if cliq_polys:
        ax.add_collection(LineCollection(cliq_polys, colors="#111111", linewidths=8.0, zorder=8))
        ax.add_collection(LineCollection(cliq_polys, colors="#ffffff", linewidths=5.6, zorder=9))
    if cliq5:
        ax.add_collection(LineCollection(cliq5, colors="#FF1493", linewidths=3.4, zorder=10))
    if cliq10:
        ax.add_collection(LineCollection(cliq10, colors="#FF1493", linewidths=3.4,
                                         linestyles="dashed", zorder=10))
    if cliq_loop_nodes:
        xy = np.array([pos[v] for v in cliq_loop_nodes])
        ax.scatter(xy[:, 0], xy[:, 1], s=130, facecolors="none", edgecolors="#111111",
                   linewidths=2.6, zorder=11)
        ax.scatter(xy[:, 0], xy[:, 1], s=88, facecolors="none", edgecolors="#FF1493",
                   linewidths=1.8, zorder=12)

    ink = "#222222"
    if label:
        CP._label_center_clusters(ax, comps, radii, cx, cy, genes, ink, OUT, min_n=10)
    ax.set_title(title + "\n" + subtitle, fontsize=15, color=ink, loc="left", pad=14)
    handles = [
        Line2D([0], [0], color="#e8433f", lw=2.4, label="cobinding clique"),
        Line2D([0], [0], color=grey, lw=1.2, label="surrounding pair"),
        Line2D([0], [0], color="#FF1493", lw=3.2, label="clique edge inside a loop span"),
        Line2D([0], [0], color="#F5C518", lw=2.6, label="Peakachu loop 5 kb"),
        Line2D([0], [0], color="#1A6B8A", lw=2.6, linestyle="--", label="Peakachu loop 10 kb"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="#FF1493", markeredgewidth=1.6, markersize=10,
               label="clique peak on that loop"),
    ]
    ax.set_aspect("equal")
    ax.axis("off")
    ax.autoscale_view()
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=11, labelcolor=ink)
    fig.savefig(out_png, dpi=170, bbox_inches="tight", facecolor=fig_bg)
    fig.savefig(out_png.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor=fig_bg)
    plt.close(fig)
    print(f"wrote {out_png}  loops drawn={n_drawn} clique-loops={n_cliq_loop} "
          f"skipped={n_skip} cross={n_cross}")
    return n_drawn, n_skip


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    G = load_graph()
    cliques = load_cliques()
    genes = load_genes()
    loops = load_loops()
    print(f"graph {G.number_of_nodes()} nodes {G.number_of_edges()} edges; "
          f"{len(cliques)} gold cliques; {len(loops)} loop-pairs")

    allc = sorted(nx.connected_components(G), key=len, reverse=True)
    comps10 = [c for c in allc if len(c) >= 10]
    loop_set = {frozenset((a, b)) for a, b, _ in loops}
    comps_loop = []
    for c in allc:
        nodes = set(c)
        if any(a in nodes and b in nodes for a, b, _ in loops):
            comps_loop.append(c)
    comps_loop.sort(key=len, reverse=True)

    plot_network(
        G, cliques, genes, loops, comps10,
        OUT / "tf_peak_network_cliques_loops.png",
        "RBBP4 co-binding clusters with Peakachu loops (5 kb ∪ 10 kb, 0.7, ±10 kb)",
        f"Pink = clique edge whose peaks both sit inside one loop span. "
        f"Gold/teal bows = other Peakachu-overlapping pairs. "
        f"{len(comps10)} components ≥10 shown.",
    )
    plot_network(
        G, cliques, genes, loops, comps_loop,
        OUT / "tf_peak_network_loop_components.png",
        "Only components that contain a Peakachu loop-span edge",
        f"{len(comps_loop)} components · pink = clique edge inside a loop span · "
        f"5 kb gold, 10 kb dashed teal",
        label=False,
    )


if __name__ == "__main__":
    main()
