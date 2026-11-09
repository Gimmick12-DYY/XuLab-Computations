"""Fig 3a v5 (64 x 98 mm): RBBP4 coordinated-binding network, the 12 largest components only, one per grid cell.
Input: inputs/network_gold.json (written by fig3_v4/panel_a_network.py --cache: RBBP4 Cicero pairs FDR <= 0.05,
coaccess >= 0, <= 1 Mb, components >= 10 regions, cluster_peaks.py component layouts; module edges > 25 kb tested against
WT Peakachu 0.7 loops 5 kb U 10 kb, +-30 kb). Each component keeps its own layout, re-centred and scaled to fill its cell.
Red = module edge on a loop, blue = module edge Cicero only, light grey = every other co-accessible pair (module edges
<= 25 kb, not tested, are drawn grey too). Nodes small black. Locus under each component."""
import json
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from fig3common import INP, RED, BLUE, INK, PALE, GREY, FS, fig_mm, mm_axes, save, plt

NCOMP, NCOL = 12, 3
D = json.load(open(INP / "network_gold.json")); pos = {k: np.array(v) for k, v in D["pos"].items()}
mset = {tuple(sorted(e[:2])): e[2] for e in D["medges"]}
par = {v: v for v in pos}
def find(v):
    while par[v] != v: par[v] = par[par[v]]; v = par[v]
    return v
for u, v in D["edges"]: par[find(u)] = find(v)
comps = {}
for v in pos: comps.setdefault(find(v), []).append(v)
comps = sorted(comps.values(), key=len, reverse=True)[:NCOMP]
parse = lambda r: (r.split(":")[0], *map(int, r.split(":")[1].split("-")))

W, H = 64.0, 98.0; LEG = 5.0; cw = W / NCOL; ch = (H - LEG) / (NCOMP // NCOL); LAB = 3.0
fig = fig_mm(W, H)
for i, c in enumerate(comps):
    cs = set(c); col, row = i % NCOL, i // NCOL
    ax = mm_axes(fig, W, H, col * cw + 0.6, H - LEG - (row + 1) * ch + LAB, cw - 1.2, ch - LAB - 0.6); ax.set_axis_off()
    xy = np.array([pos[v] for v in c]); ctr = (xy.max(0) + xy.min(0)) / 2; half = (xy.max(0) - xy.min(0)).max() / 2 + 0.5
    E = [(u, v) for u, v in D["edges"] if u in cs]
    for st, colr, lw, z in (("grey", PALE, 0.35, 1), ("cicero", BLUE, 0.7, 3), ("loop", RED, 0.8, 4)):
        seg = [(pos[u], pos[v]) for u, v in E if mset.get(tuple(sorted((u, v))), "grey").replace("short", "grey") == st]
        ax.add_collection(LineCollection(seg, colors=colr, linewidths=lw, zorder=z, capstyle="round"))
    mnodes = {v for e in mset for v in e if v in cs and mset[e] != "short"}
    ax.scatter(xy[:, 0], xy[:, 1], s=0.35, c=GREY, lw=0, zorder=2)
    mx = np.array([pos[v] for v in mnodes]) if mnodes else np.empty((0, 2))
    ax.scatter(mx[:, 0], mx[:, 1], s=1.6, c=INK, lw=0, zorder=5)
    ax.set_xlim(ctr[0] - half, ctr[0] + half); ax.set_ylim(ctr[1] - half, ctr[1] + half); ax.set_aspect("equal", adjustable="box")
    ps = [(parse(v)[1] + parse(v)[2]) / 2 for v in c]; chrom = parse(c[0])[0]
    fig.text((col + 0.5) * cw / W, (H - LEG - (row + 1) * ch + 0.9) / H, f"{chrom}:{min(ps) / 1e6:.1f}–{max(ps) / 1e6:.1f} Mb",
             ha="center", va="bottom", fontsize=FS, color=INK)
    print(i + 1, chrom, len(c), "regions", len(mnodes), "module regions",
          {s: sum(1 for u, v in E if mset.get(tuple(sorted((u, v)))) == s) for s in ("loop", "cicero", "short")})
lg = fig.add_axes([0, (H - LEG) / H, 1, LEG / H]); lg.set_xlim(0, W); lg.set_ylim(0, LEG); lg.set_axis_off()
x = 4.5
for colr, txt, lw in ((RED, "Module edge, loop", 1.0), (BLUE, "Cicero only", 1.0), (PALE, "Other pair", 1.0)):
    lg.add_line(Line2D([x, x + 3.5], [2.6, 2.6], color=colr, lw=lw, solid_capstyle="round"))
    t = lg.text(x + 4.3, 2.6, txt, va="center", fontsize=FS)
    fig.canvas.draw(); x = lg.transData.inverted().transform(t.get_window_extent().get_points())[1, 0] + 2.2
save(fig, "fig3a_network")
