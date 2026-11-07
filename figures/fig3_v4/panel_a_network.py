"""Fig 3a (92 x 75 mm): RBBP4 coordinated-binding network, redrawn in palette C at final size.
Graph = RBBP4 Cicero pairs (FDR <= 0.05, coaccess >= 0, <= 1 Mb; same filters as cluster_peaks.py), components with
>= 10 regions, each force-laid-out and disc-packed largest first (cluster_peaks.py layout helpers). Grey = surrounding
pairs. Higher-order co-binding module = maximal clique of >= 3 regions. Module edges > 25 kb are tested against WT
Peakachu 0.7 loops (5 kb U 10 kb, both ends within +-30 kb of the two anchors; compare_union_pairs_and_clique_edges.assign_loop,
as in S1D): red = on a loop, blue = Cicero only; module edges <= 25 kb are not tested (dark grey). Modules whose every
tested edge is on a loop get a dark-red node outline (same rule as the 100% stratum in panel b). The 8 largest components
are labelled by locus.
--run gold (default; results/RBBP4, the network shown on the earlier slides) or --run ourrun (RBBP4_sweep_k50_i10000_s555).
Step 1 (--cache) computes the layout + loop flags into inputs/network_<run>.npz; step 2 draws."""
import argparse, json, sys
from itertools import combinations
import numpy as np, pandas as pd
from fig3common import P, INP, XULAB, fig_mm, save, plt

ap = argparse.ArgumentParser(); ap.add_argument("--run", default="gold", choices=["gold", "ourrun"]); ap.add_argument("--cache", action="store_true")
a = ap.parse_args()
COB = XULAB / "cobinding"
SRC = dict(gold=(COB / "work/RBBP4.peak_edges.tsv", COB / "results/RBBP4/cliques.tsv"),
           ourrun=(COB / "work/RBBP4_sweep_k50_i10000_s555/RBBP4.peak_edges.tsv", COB / "results/RBBP4_sweep_k50_i10000_s555/cliques.tsv"))[a.run]
CACHE = INP / f"network_{a.run}.json"
MIN_COMP, MIN_SPAN, SLOP = 10, 25_000, 30_000
ASPECT = 1.75   # the network area is ~92 x 52 mm, so discs are packed on an elliptical spiral instead of a circle


def pack_ellipse(radii, pad, aspect):
    """cluster_peaks._pack_discs (greedy spiral, largest first) with candidate points on ellipses x = aspect * R cos t."""
    order = sorted(range(len(radii)), key=lambda i: -radii[i]); cx = np.empty(len(radii)); cy = np.empty(len(radii)); px, py, pr = [], [], []; r0 = 0.0
    for k, i in enumerate(order):
        r = radii[i]
        if not px: cx[i] = cy[i] = 0.0; px.append(0.0); py.append(0.0); pr.append(r); continue
        X_, Y_, R_ = map(np.array, (px, py, pr)); R = r0; step = max(r * 0.5, 0.25); ok = False
        while not ok:
            nt = max(12, int(2 * np.pi * R * aspect / max(step, 1e-6))); th = (np.arange(nt) / nt * 2 * np.pi + 0.61803 * k) % (2 * np.pi)
            for t in th:
                x, y = aspect * R * np.cos(t), R * np.sin(t)
                if np.all((x - X_) ** 2 + (y - Y_) ** 2 >= (R_ + r + pad) ** 2):
                    cx[i], cy[i] = x, y; px.append(x); py.append(y); pr.append(r); r0 = max(0.0, R - 3 * r); ok = True; break
            R += step
    return cx, cy


def parse(r):
    c, se = r.split(":"); s, e = se.split("-"); return c, int(s), int(e)


def mid(r):
    c, s, e = parse(r); return (s + e) // 2


if a.cache:
    import networkx as nx
    sys.path.insert(0, str(COB / "scripts")); sys.path.insert(0, str(XULAB / "hic/scripts"))
    import cluster_peaks as CP
    import compare_union_pairs_and_clique_edges as C
    import union_5k_10k_loops_vs_pairs as U
    df = pd.read_csv(SRC[0], sep="\t")
    df = df[(df.qval <= 0.05) & (df.coaccess >= 0) & (df.n_links >= 1)]
    df = df[[(d := CP.pair_distance_bp(str(x), str(y))) is not None and d <= 1_000_000 for x, y in zip(df.peak1, df.peak2)]]
    G = nx.Graph(); G.add_edges_from(zip(df.peak1.astype(str), df.peak2.astype(str)))
    comps = [c for c in sorted(nx.connected_components(G), key=len, reverse=True) if len(c) >= MIN_COMP]
    keep = set().union(*comps); H = G.subgraph(keep)
    radii, locs = [], []
    for c in comps:
        loc = CP._spread_local(CP._component_layout(H.subgraph(c)), min_d=0.42)
        radii.append(max((x * x + y * y) ** 0.5 for x, y in loc.values()) + 0.42 * 0.8); locs.append(loc)
    cx, cy = pack_ellipse(radii, pad=0.85, aspect=ASPECT)
    pos = {v: (x + cx[i], y + cy[i]) for i, loc in enumerate(locs) for v, (x, y) in loc.items()}
    L = XULAB / "hic/work_wt293t/loops"
    fine = U.load_bedpe(L / "peakachu_5000.loops.0.7.bedpe", "5000"); coarse = U.load_bedpe(L / "peakachu_10000.loops.0.7.bedpe", "10000")
    ix5, ix10 = U.index_anchors(fine, 5000), U.index_anchors(coarse, 10000)
    Cq = pd.read_csv(SRC[1], sep="\t"); mods, medges = [], {}
    for cid, regs in zip(Cq.clique_id, Cq.regions):
        rr = regs.split(";")
        if not set(rr) & keep: continue
        assert set(rr) <= keep, cid
        es = []
        for x, y in combinations(sorted(rr, key=lambda r: (parse(r)[0], parse(r)[1])), 2):
            if not H.has_edge(x, y): continue
            sp = abs(mid(x) - mid(y))
            if sp <= MIN_SPAN: st = "short"
            else:
                pr = {"p1": U.parse_peak(x), "p2": U.parse_peak(y)}
                st = "loop" if C.assign_loop(pr, fine, ix5, coarse, ix10, SLOP) is not None else "cicero"
            medges[tuple(sorted((x, y)))] = st; es.append(st)
        tested = [s for s in es if s != "short"]
        mods.append(dict(id=cid, regions=rr, all_loop=bool(tested) and all(s == "loop" for s in tested)))
    lab = []
    for i, c in enumerate(comps[:8]):
        ch = {parse(r)[0] for r in c}; assert len(ch) == 1
        ps = [mid(r) for r in c]; lab.append(dict(i=i, chrom=ch.pop(), lo=min(ps), hi=max(ps), n=len(c), cx=cx[i], cy=cy[i], r=radii[i]))
    json.dump(dict(pos=pos, edges=[list(e) for e in H.edges()], medges=[[x, y, s] for (x, y), s in medges.items()], mods=mods, labels=lab,
                   n_comp=len(comps), n_reg=H.number_of_nodes(), n_edge=H.number_of_edges()), open(CACHE, "w"))
    st = pd.Series(list(medges.values())).value_counts()
    print(f"{a.run}: {len(comps)} components >= {MIN_COMP}, {H.number_of_nodes()} regions, {H.number_of_edges()} edges; modules {len(mods)} "
          f"(all tested edges on a loop: {sum(m['all_loop'] for m in mods)}); module edges {st.to_dict()}")
    raise SystemExit

D = json.load(open(CACHE)); pos = {k: np.array(v) for k, v in D["pos"].items()}
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
W, Hh = 92.0, 75.0; LG = 17.0
N = P.neutral; RED, BLUE = P.evidence.hic_loop, P.evidence.cicero_only; DARK = P.loop_strata[-1]; SHORT = N.dark
fig = fig_mm(W, Hh); ax = fig.add_axes([0, LG / Hh, 1, 1 - LG / Hh]); ax.set_axis_off(); ax.set_aspect("equal", adjustable="datalim")
xy = np.array(list(pos.values())); x0, x1, y0, y1 = xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max()
pad = 0.9; ax.set_xlim(x0 - pad, x1 + pad); ax.set_ylim(y0 - pad, y1 + pad)
mset = {tuple(sorted(e[:2])): e[2] for e in D["medges"]}
grey = [(pos[u], pos[v]) for u, v in D["edges"] if tuple(sorted((u, v))) not in mset]
ax.add_collection(LineCollection(grey, colors=N.light, linewidths=0.25, zorder=1))
mnodes = {r for m in D["mods"] for r in m["regions"]}
rest = np.array([pos[v] for v in pos if v not in mnodes]); ax.scatter(rest[:, 0], rest[:, 1], s=0.4, c=N.light, lw=0, zorder=2)
for st, col, lw, z in (("short", SHORT, 0.55, 3), ("cicero", BLUE, 0.8, 4), ("loop", RED, 0.9, 5)):
    seg = [(pos[x], pos[y]) for (x, y), s in mset.items() if s == st]
    ax.add_collection(LineCollection(seg, colors=col, linewidths=lw, zorder=z, capstyle="round"))
full = {r for m in D["mods"] if m["all_loop"] for r in m["regions"]}
mn = np.array([pos[v] for v in mnodes - full]); ax.scatter(mn[:, 0], mn[:, 1], s=1.3, c=N.ink, lw=0, zorder=6)
fn = np.array([pos[v] for v in full]); ax.scatter(fn[:, 0], fn[:, 1], s=3.0, c=RED, edgecolors=DARK, linewidths=0.5, zorder=7)
for k, l in enumerate(D["labels"]):
    ax.text(l["cx"] - l["r"] * 0.72, l["cy"] + l["r"] * 0.72, str(k + 1), ha="right", va="bottom", fontsize=6, fontweight="bold", color=N.ink, zorder=8)
lg = fig.add_axes([0, 0, 1, LG / Hh]); lg.set_xlim(0, W); lg.set_ylim(0, LG); lg.set_axis_off()
def item(x, y, kind, col, txt, **kw):
    if kind == "line": lg.add_line(Line2D([x, x + 4], [y, y], color=col, lw=kw.get("lw", 1.0), solid_capstyle="round"))
    else: lg.scatter([x + 2], [y], s=kw.get("s", 6), c=col, edgecolors=kw.get("ec", "none"), linewidths=kw.get("lw", 0))
    lg.text(x + 5, y, txt, va="center", fontsize=6)
item(2, 15.2, "line", RED, "Module edge on a Hi-C loop", lw=1.0)
item(2, 12.2, "line", BLUE, "Module edge, Cicero only", lw=1.0)
item(2, 9.2, "line", SHORT, "Module edge ≤25 kb (not tested)", lw=0.8)
item(48, 15.2, "line", N.light, "Other co-accessible pair", lw=0.8)
item(48, 12.2, "dot", RED, "Module with every edge on a loop", ec=DARK, lw=0.5, s=6)
lg.text(48, 9.2, "Module = higher-order co-binding module", va="center", fontsize=6, color=N.dark)
for k, l in enumerate(D["labels"]):
    x, y = [1.5, 23.5, 46.0, 70.0][k % 4], 5.2 - (k // 4) * 3.0
    lg.text(x, y, str(k + 1), fontweight="bold", va="center", fontsize=6); lg.text(x + 2.0, y, f"{l['chrom']}:{l['lo'] / 1e6:.1f}–{l['hi'] / 1e6:.1f} Mb", va="center", fontsize=6, color=N.dark)
print(f"{a.run}: {D['n_comp']} components, {D['n_reg']} regions, {D['n_edge']} edges, modules {len(D['mods'])}, all-loop modules {sum(m['all_loop'] for m in D['mods'])}")
save(fig, "fig3a_network")
