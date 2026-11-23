"""Shared drawing code for the coordinated binding network panels (rbbp4_network.py, ets2_network.py).

Graph of one TF: first-step Cicero pairs (q <= 0.05, co-accessibility >= 0, ends on one chromosome <= 1 Mb apart); connected
components with >= 10 regions are drawn. Each component is laid out with Fruchterman-Reingold (python-igraph, fixed seed,
nodes in sorted order) and the components are packed largest-first in a spiral. Higher-order coordinated binding modules
(cluster_peaks.py cliques, >= 3 regions, every pair linked) get one colour each (hue wheel, overlapping modules pushed apart);
hinge regions shared by >= 2 modules are outlined in black; all other pairs light grey.

draw_all()  : every component on one large canvas (original RBBP4 panel; 24 x 24 in, scale down when placing).
draw_zoom() : grey overview of every component with one boxed region, and the boxed region redrawn in colour at print size,
              each component labelled by its chromosome span.
"""
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd

SEED = 2026
GREY, INK = "#C8CCD2", "#222222"
MAX_DIST, MIN_COMP, NODE_SEP = 1_000_000, 10, 0.42


def parse(p):
    c, xy = p.split(":")
    s, e = xy.split("-")
    return c, int(s), int(e)


def load_graph(edges_path):
    df = pd.read_csv(edges_path, sep="\t")
    df = df[(df.qval <= 0.05) & (df.coaccess >= 0) & (df.n_links >= 1)]
    keep = []
    for a, b in zip(df.peak1, df.peak2):
        ca, sa, ea = parse(a)
        cb, sb, eb = parse(b)
        keep.append(ca == cb and abs((sa + ea) // 2 - (sb + eb) // 2) <= MAX_DIST)
    G = nx.Graph()
    G.add_edges_from(sorted(tuple(sorted(e)) for e in zip(df.peak1[keep], df.peak2[keep])))
    return G


def load_modules(modules_path):
    M = pd.read_csv(modules_path, sep="\t")
    return [frozenset(r.split(";")) for r in M.regions]


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


def span_label(c):
    ch = {parse(r)[0] for r in c}
    s, e = min(parse(r)[1] for r in c), max(parse(r)[2] for r in c)
    return f"{ch.pop()}:{s / 1e6:.1f}–{e / 1e6:.1f} Mb" if len(ch) == 1 else f"{len(ch)} chromosomes"


def load_genes(tss_path, regions):
    """Region -> RefSeq Select genes whose TSS lies within the region +-1 kb (promoter regions only)."""
    T = pd.read_csv(tss_path, sep="\t")
    by = {c: (g.tss.to_numpy(), g.gene.to_numpy()) for c, g in T.sort_values(["chrom", "tss"]).groupby("chrom")}
    out = {}
    for r in regions:
        c, s, e = parse(r)
        if c in by:
            pos, gn = by[c]
            i, j = np.searchsorted(pos, s - 1000), np.searchsorted(pos, e + 1000, side="right")
            if j > i:
                out[r] = sorted(set(gn[i:j]))
    return out


def gene_label(c, genes, H, k=2):
    """Genes of the k best-connected promoter regions in the component (degree in the pair graph, ties alphabetical)."""
    prom = sorted((r for r in c if genes.get(r)), key=lambda r: (-H.degree(r), genes[r][0]))
    top = []
    for r in prom:
        for g in genes[r]:
            if g not in top:
                top.append(g)
    return ", ".join(top[:k]) + ("…" if len(top) > k else "")


def build(edges_path, modules_path, min_comp=MIN_COMP):
    """Graph, modules, components >= MIN_COMP and the packed layout (seeded, reproducible)."""
    random.seed(SEED)
    ig.set_random_number_generator(random)
    G = load_graph(edges_path)
    modules = load_modules(modules_path)
    comps = sorted((sorted(c) for c in nx.connected_components(G) if len(c) >= min_comp), key=lambda c: (-len(c), c[0]))
    keep = set().union(*map(set, comps))
    H = G.subgraph(keep)
    locs, radii = [], []
    for c in comps:
        loc = spread(component_layout(H.subgraph(c)), NODE_SEP)
        locs.append(loc)
        radii.append(max((x * x + y * y) ** 0.5 for x, y in loc.values()) + NODE_SEP * 0.8)
    cx, cy = pack_discs(radii)
    pos = {v: (x + cx[i], y + cy[i]) for i, loc in enumerate(locs) for v, (x, y) in loc.items()}
    return dict(min_comp=min_comp, G=G, H=H, modules=modules, comps=comps, keep=keep, pos=pos, radii=np.array(radii), cx=cx, cy=cy)


def draw_coloured(ax, B, nodes, s_node, lw_mod, lw_grey, s_hinge_lw):
    """Module colouring of the subgraph on `nodes` (grey pairs, coloured module edges, hinge outlines)."""
    from matplotlib.collections import LineCollection
    H, pos = B["H"].subgraph(nodes), B["pos"]
    mods = [c for c in B["modules"] if c <= nodes]
    live, cmap = module_colours(mods, set(nodes))
    node_col, n_mod = {}, Counter()
    for i, c in sorted(enumerate(live), key=lambda kv: len(kv[1])):
        for r in c:
            node_col[r] = cmap[i]
            n_mod[r] += 1
    module_nodes = set().union(*live) if live else set()
    module_edges = {frozenset((a, b)) for c in live for a, b in combinations(c, 2) if H.has_edge(a, b)}
    ax.add_collection(LineCollection([(pos[u], pos[v]) for u, v in sorted(H.edges()) if frozenset((u, v)) not in module_edges],
                                     colors=GREY, linewidths=lw_grey, alpha=0.9, zorder=1))
    rest = sorted(n for n in H if n not in module_nodes)
    if rest:
        xy = np.array([pos[v] for v in rest])
        ax.scatter(xy[:, 0], xy[:, 1], s=s_node * 0.6, c=GREY, linewidths=0, zorder=2)
    for i, c in enumerate(live):
        segs = [(pos[a], pos[b]) for a, b in combinations(sorted(c), 2) if H.has_edge(a, b)]
        if segs:
            ax.add_collection(LineCollection(segs, colors=[cmap[i]], linewidths=lw_mod, zorder=3))
    mn = sorted(module_nodes)
    if mn:
        xy = np.array([pos[v] for v in mn])
        hinge = [n_mod[v] > 1 for v in mn]
        ax.scatter(xy[:, 0], xy[:, 1], s=s_node, c=[node_col[v] for v in mn],
                   linewidths=[s_hinge_lw if h else 0 for h in hinge], edgecolors=["#111111" if h else "none" for h in hinge], zorder=4)
    return len(live), sum(n_mod[v] > 1 for v in mn)


def draw_all(edges_path, modules_path, out, name, min_comp=MIN_COMP, tss_path=None, tf=None, top=24):
    """Every component >= min_comp regions on one 24 x 24 in canvas, the `top` largest labelled (original RBBP4 panel).
    tss_path: label with the genes of the two best-connected promoter regions + span + region count; tf: one-line summary on top."""
    from common import style
    style.use()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    B = build(edges_path, modules_path, min_comp)
    genes = load_genes(tss_path, B["keep"]) if tss_path else None
    fig, ax = plt.subplots(figsize=(24, 24))
    n_live, n_hinge = draw_coloured(ax, B, B["keep"], s_node=28, lw_mod=2.3, lw_grey=0.6, s_hinge_lw=0.9)
    rows = []
    for i, c in enumerate(B["comps"]):
        g = gene_label(c, genes, B["H"]) if genes is not None else ""
        lab = (f"{g}\n" if g else "") + span_label(c) + f"\n{len(c)} regions" if genes is not None else span_label(c).replace("–", "-") + f"\n{len(c)} reg"
        rows.append((i + 1, len(c), span_label(c), g))
        if i < top:
            ax.text(B["cx"][i], B["cy"][i] + B["radii"][i] + 0.12, lab, ha="center", va="bottom", fontsize=6.5 if genes is None else 9, color=INK,
                    zorder=6, linespacing=1.15, bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75))
    pd.DataFrame(rows, columns=["rank", "n_regions", "label", "genes"]).to_csv(Path(out) / f"{name}_components.tsv", sep="\t", index=False)
    handles = [Line2D([0], [0], color="#E8433F", lw=2.4, label="one higher-order coordinated binding module"),
               Line2D([0], [0], color="#33B5E5", lw=2.4, label="another module"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor="#EDA320", markeredgecolor="#111111", markeredgewidth=0.8,
                      markersize=8, label="hinge (shared by ≥2 modules)"),
               Line2D([0], [0], color=GREY, lw=1.2, label="surrounding pair")]
    P_ = np.array(list(B["pos"].values()))
    ax.set_xlim(P_[:, 0].min() - 0.6, P_[:, 0].max() + 0.6)
    ax.set_ylim(P_[:, 1].min() - 0.6, P_[:, 1].max() + 1.2)   # room for the component labels on top
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(handles=handles, loc="lower right", fontsize=11, labelcolor=INK)
    H = B["H"]
    if tf:
        ax.set_title(f"{tf}: {len(H):,} regions · {H.number_of_edges():,} pairs · {len(B['comps']):,} networks ≥{min_comp} regions"
                     f" (of {B['G'].number_of_nodes():,} regions in the pair graph) · {n_live:,} higher-order coordinated binding modules"
                     f" · {n_hinge:,} hinge regions · labels = {top} largest networks", fontsize=13, color=INK, loc="left")
    print(f"{name}: {B['G'].number_of_nodes()} nodes, {len(B['comps'])} components >= {min_comp} ({len(H)} regions, {H.number_of_edges()} pairs), "
          f"{n_live} modules, {n_hinge} hinge regions")
    style.save(fig, out, name, bbox_inches="tight", pad_inches=0.05)


def pick_zoom_box(B, n_target=(8, 12), size_range=(20, 80), strict=True):
    """Square box holding n_target components entirely, every one of them a mid-sized component (size_range regions) so
    modules stay readable; among candidate boxes (one per eligible component centre, grown until it holds n_target[0]
    components) the one with the most module regions inside wins. Returns (x0, y0, side, component indices)."""
    comps, cx, cy, rad = B["comps"], B["cx"], B["cy"], B["radii"]
    n = np.array([len(c) for c in comps])
    in_mod = set().union(*B["modules"])
    n_mod = np.array([sum(v in in_mod for v in c) for c in comps])
    lo, hi = size_range
    best = None
    for i in np.flatnonzero((n >= lo) & (n <= hi)):
        d = np.maximum(np.abs(cx - cx[i]), np.abs(cy - cy[i])) + rad   # half-side needed to hold component j entirely
        order = np.argsort(d)
        for k in range(n_target[0], n_target[1] + 1):
            h = d[order[k - 1]]
            inside = order[:k]
            touched = np.flatnonzero(np.maximum(np.abs(cx - cx[i]), np.abs(cy - cy[i])) - rad < h)   # any part in the box
            if (strict and len(touched) > k) or not np.all((n[inside] >= lo) & (n[inside] <= hi)):
                continue
            score = n_mod[inside].sum() / (2 * h) ** 2 * k   # module regions per area, favouring more components
            if best is None or score > best[0]:
                best = (score, cx[i] - h, cy[i] - h, 2 * h, sorted(inside.tolist()))
    if best is None:
        return pick_zoom_box(B, n_target, size_range, strict=False)
    _, x0, y0, side, idx = best
    return x0, y0, side, idx


def draw_zoom(edges_path, modules_path, out, name, tf, w_mm=183.0, h_mm=92.0):
    """Grey overview of every component + one boxed region redrawn in colour, components labelled by chromosome span."""
    from common import style
    style.use()
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import ConnectionPatch, Rectangle
    B = build(edges_path, modules_path)
    pos, H = B["pos"], B["H"]
    x0, y0, side, idx = pick_zoom_box(B)
    zoom_nodes = set().union(*(set(B["comps"][i]) for i in idx))

    fig = style.figure_mm(w_mm, h_mm)
    mm = lambda x, y, w, h: [x / w_mm, y / h_mm, w / w_mm, h / h_mm]
    ov = fig.add_axes(mm(1, 10, 80, 80))
    zm = fig.add_axes(mm(100, 10, 81, 81))

    # overview: everything grey, module edges a darker grey so the structure shows
    in_mod = {frozenset(e) for c in B["modules"] for e in combinations(c, 2)}
    E = sorted(H.edges())
    ov.add_collection(LineCollection([(pos[u], pos[v]) for u, v in E if frozenset((u, v)) not in in_mod], colors="#DADDE2",
                                     linewidths=0.12, zorder=1, rasterized=True))
    ov.add_collection(LineCollection([(pos[u], pos[v]) for u, v in E if frozenset((u, v)) in in_mod], colors="#8C8C8C",
                                     linewidths=0.15, zorder=2, rasterized=True))
    xy = np.array([pos[v] for v in sorted(H)])
    ov.scatter(xy[:, 0], xy[:, 1], s=0.04, c="#6E6E6E", linewidths=0, zorder=3, rasterized=True)
    ov.add_patch(Rectangle((x0, y0), side, side, fill=False, edgecolor="#111111", lw=0.8, zorder=5))
    P_ = np.array(list(pos.values()))
    lo_, hi_ = P_.min(axis=0) - 0.5, P_.max(axis=0) + 0.5
    c_, h_ = (lo_ + hi_) / 2, (hi_ - lo_).max() / 2
    ov.set_xlim(c_[0] - h_, c_[0] + h_); ov.set_ylim(c_[1] - h_, c_[1] + h_)
    ov.set_aspect("equal"); ov.axis("off")
    ov.text(0.5, -0.02, f"{tf}: {len(B['comps']):,} networks (≥{MIN_COMP} regions), {len(H):,} regions", transform=ov.transAxes,
            ha="center", va="top", fontsize=6, color=INK)

    # zoom: units -> points so node size and line widths are physical
    pt_per_unit = 81 / 25.4 * 72 / side
    m = 9 / pt_per_unit   # headroom on top of the zoom for the uppermost labels (box stays square)
    pt_per_unit = 81 / 25.4 * 72 / (side + m)
    m = 9 / pt_per_unit
    nd = min(4.0, 0.26 * pt_per_unit)   # node diameter in pt
    n_live, n_hinge = draw_coloured(zm, B, zoom_nodes, s_node=nd ** 2, lw_mod=min(0.9, max(0.4, 0.06 * pt_per_unit)),
                                    lw_grey=0.3, s_hinge_lw=0.3)
    for i in idx:   # label above each component, kept inside the box (6 pt Arial ~ 0.55 em per character)
        lab = span_label(B["comps"][i])
        half = len(lab) * 6 * 0.55 / 2 / pt_per_unit
        xy_c = np.array([pos[v] for v in B["comps"][i]])
        x = min(max((xy_c[:, 0].min() + xy_c[:, 0].max()) / 2, x0 - m / 2 + half + 0.1), x0 + side + m / 2 - half - 0.1)
        pad = 3 / pt_per_unit
        if xy_c[:, 1].max() + pad + 8 / pt_per_unit < y0 + side + m:   # just above the drawn nodes, else just below
            zm.text(x, xy_c[:, 1].max() + pad, lab, ha="center", va="bottom", fontsize=6, color=INK, zorder=6)
        else:
            zm.text(x, xy_c[:, 1].min() - pad, lab, ha="center", va="top", fontsize=6, color=INK, zorder=6)
    zm.set_xlim(x0 - m / 2, x0 + side + m / 2); zm.set_ylim(y0, y0 + side + m)
    zm.set_aspect("equal"); zm.axis("off")
    zm.add_patch(Rectangle((0, 0), 1, 1, transform=zm.transAxes, fill=False, edgecolor="#111111", lw=0.8, zorder=7, clip_on=False))
    for (xa, ya), (xb, yb) in (((x0 + side, y0 + side), (0, 1)), ((x0 + side, y0), (0, 0))):
        fig.add_artist(ConnectionPatch((xa, ya), (xb, yb), coordsA=ov.transData, coordsB=zm.transAxes, color="#8C8C8C", lw=0.5))

    handles = [Line2D([0], [0], color="#E8433F", lw=1.2, label="one higher-order coordinated binding module"),
               Line2D([0], [0], color="#33B5E5", lw=1.2, label="another module"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor="#EDA320", markeredgecolor="#111111", markeredgewidth=0.4,
                      markersize=3.5, label="hinge (shared by ≥2 modules)"),
               Line2D([0], [0], color=GREY, lw=0.8, label="surrounding pair")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(140.5 / w_mm, 0.0), ncol=2, frameon=False, fontsize=6,
               handlelength=1.4, columnspacing=1.0, labelcolor=INK)

    rows = [(i + 1, len(B["comps"][i]), span_label(B["comps"][i]), i in idx) for i in range(len(B["comps"]))]
    pd.DataFrame(rows, columns=["rank", "n_regions", "label", "in_zoom"]).to_csv(Path(out) / f"{name}_components.tsv", sep="\t", index=False)
    print(f"{name}: {B['G'].number_of_nodes()} nodes in graph, {len(B['comps'])} components >= {MIN_COMP} ({len(H)} regions); zoom "
          f"{len(idx)} components ranks {[i + 1 for i in idx]} sizes {[len(B['comps'][i]) for i in idx]}, {n_live} modules, "
          f"{n_hinge} hinge regions")
    style.save(fig, out, name)
