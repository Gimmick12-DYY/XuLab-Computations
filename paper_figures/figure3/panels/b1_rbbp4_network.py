"""Figure 3b (option 1): RBBP4 coordinated binding network (original colour code, no Hi-C loop overlay; PI request 2026-11-21).

Graph: RBBP4 first-step Cicero pairs (q <= 0.05, co-accessibility >= 0, ends <= 1 Mb apart); connected components with
>= 10 regions are drawn, packed largest-first in a spiral. Higher-order coordinated binding modules (Cicero cliques) get one
colour each (hue wheel, overlapping modules pushed apart); hinge regions shared by >= 2 modules are outlined in black;
all other pairs light grey. Labels: chromosome span and number of regions of the 24 largest components.

Inputs (source_data/):  rbbp4_peak_edges.tsv.gz, rbbp4_modules.tsv
Output: b1_rbbp4_network.{pdf,svg,png}. Drawn on a 24 x 24 in canvas (large original; scale it down when placing).
Layout: Fruchterman-Reingold per component (python-igraph, fixed seed, nodes in sorted order), so the layout is reproducible.
"""
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from common import style  # noqa: E402

SEED = 2026
GREY, INK = "#C8CCD2", "#222222"


def parse(p):
    c, xy = p.split(":")
    s, e = xy.split("-")
    return c, int(s), int(e)


def load_graph(src):
    df = pd.read_csv(src / "rbbp4_peak_edges.tsv.gz", sep="\t")
    df = df[(df.qval <= 0.05) & (df.coaccess >= 0) & (df.n_links >= 1)]
    keep = []
    for a, b in zip(df.peak1, df.peak2):
        ca, sa, ea = parse(a)
        cb, sb, eb = parse(b)
        keep.append(ca == cb and abs((sa + ea) // 2 - (sb + eb) // 2) <= 1_000_000)
    G = nx.Graph()
    G.add_edges_from(sorted(tuple(sorted(e)) for e in zip(df.peak1[keep], df.peak2[keep])))
    return G


def component_layout(sub):
    nodes = sorted(sub)
    n = len(nodes)
    if n <= 2:
        return {v: (float(i) * 2 - 1, 0.0) for i, v in enumerate(nodes)} if n == 2 else {nodes[0]: (0.0, 0.0)}
    idx = {v: i for i, v in enumerate(nodes)}
    g = ig.Graph(n=n, edges=sorted((idx[u], idx[v]) if idx[u] < idx[v] else (idx[v], idx[u]) for u, v in sub.edges()))
    xy = np.asarray(g.layout_fruchterman_reingold(niter=400 if n < 400 else 200).coords, float)
    xy -= xy.mean(axis=0)
    xy /= float(np.abs(xy).max()) or 1.0
    return {v: (float(xy[i, 0]), float(xy[i, 1])) for i, v in enumerate(nodes)}


def spread(loc, min_d, iters=60):
    """Push nodes apart until nearest neighbours are >= min_d apart (no re-normalisation)."""
    nodes = list(loc)
    if len(nodes) < 2:
        return loc
    xy = np.array([loc[v] for v in nodes], float)
    for _ in range(iters):
        delta = xy[:, None, :] - xy[None, :, :]
        dist = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(dist, np.inf)
        too = dist < min_d
        if not too.any():
            break
        push = np.zeros_like(xy)
        for i in range(len(xy)):
            js = np.flatnonzero(too[i])
            if js.size:
                push[i] += ((min_d - dist[i, js][:, None]) * delta[i, js] / np.maximum(dist[i, js][:, None], 1e-9)).sum(axis=0)
        xy += 0.4 * push
    xy -= xy.mean(axis=0)
    return {v: (float(xy[i, 0]), float(xy[i, 1])) for i, v in enumerate(nodes)}


def pack_discs(radii, pad=0.85):
    """Greedy spiral packing, largest disc first."""
    order = sorted(range(len(radii)), key=lambda i: -radii[i])
    cx, cy = np.empty(len(radii)), np.empty(len(radii))
    px, py, pr, r_start = [], [], [], 0.0
    for k, i in enumerate(order):
        r = radii[i]
        if not px:
            cx[i] = cy[i] = 0.0
            px.append(0.0); py.append(0.0); pr.append(r)
            continue
        ax_, ay_, ar_ = np.array(px), np.array(py), np.array(pr)
        R, step = max(r_start, 0.0), max(r * 0.7, 0.35)
        while True:
            nth = max(8, int(2 * np.pi * R / step))
            for t in (np.arange(nth) / nth * 2 * np.pi + 0.61803 * k) % (2 * np.pi):
                x, y = R * np.cos(t), R * np.sin(t)
                if np.all((x - ax_) ** 2 + (y - ay_) ** 2 >= (ar_ + r + pad) ** 2):
                    cx[i], cy[i] = x, y
                    px.append(x); py.append(y); pr.append(r)
                    r_start = max(0.0, R - 3 * r)
                    break
            else:
                R += step
                continue
            break
    return cx, cy


def module_colours(modules, keep):
    """One hue per module touching the drawn components; overlapping modules pushed apart on the hue wheel."""
    import matplotlib.pyplot as plt
    live = [c for c in modules if c & keep]
    hues = [(0.02 + i * 0.61803398875) % 0.92 for i in range(len(live))]
    hit = defaultdict(list)
    for i, c in enumerate(live):
        for r in c:
            hit[r].append(i)
    for ids in hit.values():
        for i, j in combinations(ids, 2):
            if min(abs(hues[i] - hues[j]), 0.92 - abs(hues[i] - hues[j])) < 0.08:
                hues[j] = (hues[i] + 0.5) % 0.92
    return live, {i: plt.cm.hsv(h) for i, h in enumerate(hues)}


def main(src, out):
    random.seed(SEED)
    ig.set_random_number_generator(random)
    style.use()
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    G = load_graph(src)
    M = pd.read_csv(src / "rbbp4_modules.tsv", sep="\t")
    modules = [frozenset(r.split(";")) for r in M.regions]

    comps = sorted((sorted(c) for c in nx.connected_components(G) if len(c) >= 10), key=lambda c: (-len(c), c[0]))
    keep = set().union(*map(set, comps))
    H = G.subgraph(keep)
    locs, radii = [], []
    for c in comps:
        loc = spread(component_layout(H.subgraph(c)), 0.42)
        locs.append(loc)
        radii.append(max((x * x + y * y) ** 0.5 for x, y in loc.values()) + 0.42 * 0.8)
    cx, cy = pack_discs(radii)
    pos = {v: (x + cx[i], y + cy[i]) for i, loc in enumerate(locs) for v, (x, y) in loc.items()}

    live, cmap = module_colours(modules, keep)
    node_col, n_mod = {}, Counter()
    for i, c in sorted(enumerate(live), key=lambda kv: len(kv[1])):
        for r in c:
            if r in keep:
                node_col[r] = cmap[i]
                n_mod[r] += 1
    module_nodes = set().union(*modules)
    module_edges = {frozenset((a, b)) for c in modules for a, b in combinations(c, 2) if H.has_edge(a, b)}

    fig, ax = plt.subplots(figsize=(24, 24))
    ax.add_collection(LineCollection([(pos[u], pos[v]) for u, v in sorted(H.edges()) if frozenset((u, v)) not in module_edges],
                                     colors=GREY, linewidths=0.6, alpha=0.9, zorder=1))
    rest = sorted(n for n in H if n not in module_nodes)
    xy = np.array([pos[v] for v in rest])
    ax.scatter(xy[:, 0], xy[:, 1], s=16, c=GREY, linewidths=0, zorder=2)
    for i, c in enumerate(live):
        segs = [(pos[a], pos[b]) for a, b in combinations(sorted(c), 2) if H.has_edge(a, b)]
        if segs:
            ax.add_collection(LineCollection(segs, colors=[cmap[i]], linewidths=2.3, zorder=3))
    mn = sorted(n for n in H if n in module_nodes)
    xy = np.array([pos[v] for v in mn])
    hinge = [n_mod.get(v, 1) > 1 for v in mn]
    ax.scatter(xy[:, 0], xy[:, 1], s=28, c=[node_col.get(v, "#111111") for v in mn],
               linewidths=[0.9 if h else 0 for h in hinge], edgecolors=["#111111" if h else "none" for h in hinge], zorder=4)

    # label every component: chromosome span and number of regions
    rows = []
    for i, c in enumerate(comps):
        ch = {parse(r)[0] for r in c}
        s, e = min(parse(r)[1] for r in c), max(parse(r)[2] for r in c)
        lab = (f"{ch.pop()}:{s / 1e6:.1f}-{e / 1e6:.1f} Mb" if len(ch) == 1 else f"{len(ch)} chromosomes") + f"\n{len(c)} reg"
        rows.append((i + 1, len(c), lab.split("\n")[0]))
        if i < 24:
            ax.text(cx[i], cy[i] + radii[i] + 0.12, lab, ha="center", va="bottom", fontsize=6.5, color=INK, zorder=6, linespacing=1.15,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75))
    pd.DataFrame(rows, columns=["rank", "n_regions", "label"]).to_csv(Path(out) / "b1_rbbp4_network_components.tsv", sep="\t", index=False)

    handles = [Line2D([0], [0], color="#E8433F", lw=2.4, label="one higher-order coordinated binding module"),
               Line2D([0], [0], color="#33B5E5", lw=2.4, label="another module"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor="#EDA320", markeredgecolor="#111111", markeredgewidth=0.8,
                      markersize=8, label="hinge (shared by ≥2 modules)"),
               Line2D([0], [0], color=GREY, lw=1.2, label="surrounding pair")]
    P_ = np.array(list(pos.values()))
    ax.set_xlim(P_[:, 0].min() - 0.6, P_[:, 0].max() + 0.6)
    ax.set_ylim(P_[:, 1].min() - 0.6, P_[:, 1].max() + 1.2)   # room for the component labels on top
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(handles=handles, loc="lower right", fontsize=11, labelcolor=INK)
    print(f"b1: {G.number_of_nodes()} nodes, {len(comps)} components >= 10, {len(live)} modules")
    style.save(fig, out, "b1_rbbp4_network", bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
