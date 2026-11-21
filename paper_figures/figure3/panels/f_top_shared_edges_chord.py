"""Figure 3f: chord diagram of the 10 most widely shared module edges (> 25 kb, one per locus: edges whose first region lies
within 200 kb of an already picked edge are skipped) and the TFs that have each exact edge in one of their higher-order
coordinated binding modules. Left: TFs (shade = how many of the 10 edges the TF has); right: the edges (label = ends,
promoter gene if a TSS lies within +-1 kb, length, number of TFs, and whether the edge is on a Peakachu 0.7 loop).

Inputs (source_data/): f_top_shared_module_edges.tsv (all module edges shared by >= 2 TFs, from edge_overlap.py),
  f_gene_symbols.tsv (GENCODE gene id -> symbol)
Output: f_top_shared_edges_chord.{pdf,svg,png}; drawn on a 16 x 13 in canvas (large original; scale it down when placing).
"""
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from common import style  # noqa: E402

HUBCOL = ["#4C8BC9", "#F2B705", "#7B3F9E", "#E0475A", "#59C1C4", "#E8743B", "#5E7183", "#2E9E5B", "#B5838D", "#8C6D31"]


def pol(t, r=1.0):
    return r * math.cos(t), r * math.sin(t)


def main(src, out):
    style.use(base_pt=12, small_pt=12)
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch, Wedge
    from matplotlib.path import Path as MP
    sym = pd.read_csv(src / "f_gene_symbols.tsv", sep="\t").set_index("gene_id").symbol.to_dict()
    d = pd.read_csv(src / "f_top_shared_module_edges.tsv", sep="\t")
    d["tfl"] = d.tfs.str.split(",")
    d["n"] = d.tfl.str.len()
    d = d[d.distance_kb > 25].sort_values(["n", "distance_kb"], ascending=[False, False])
    pick, seen = [], []
    for r in d.itertuples():
        c, p = r.region1.split(":")[0], int(r.region1.split(":")[1].split("-")[0])
        if any(c == c2 and abs(p - p2) < 200_000 for c2, p2 in seen):
            continue
        seen.append((c, p))
        pick.append(r)
        if len(pick) == 10:
            break
    pd.DataFrame([dict(region1=r.region1, region2=r.region2, distance_kb=r.distance_kb, n_tfs=r.n, tfs=r.tfs,
                       on_loop_0_7=getattr(r, "_6")) for r in pick]).to_csv(Path(out) / "f_top_shared_edges_chord_edges.tsv", sep="\t", index=False)

    def lab(reg, g):
        if isinstance(g, str) and g:
            return "/".join(sym.get(x, x) for x in g.split(",")) + " prom."
        c, se = reg.split(":")
        return f"distal {c}:{int(se.split('-')[0]) / 1e6:.2f} Mb"

    def ribbon(ax, a1, a2, b1, b2, col, alpha, r=0.94):
        v = [pol(a1, r), pol((a1 + a2) / 2, r), pol(a2, r), (0, 0), pol(b2, r), pol((b1 + b2) / 2, r), pol(b1, r), (0, 0), pol(a1, r)]
        ax.add_patch(PathPatch(MP(v, [MP.MOVETO] + [MP.CURVE3] * 8), fc=col, ec="none", alpha=alpha))

    links = [(t, i) for i, r in enumerate(pick) for t in r.tfl]
    L = pd.DataFrame(links, columns=["tf", "e"])
    first, cnt, nE = L.groupby("tf").e.min(), L.groupby("tf").e.count(), L.groupby("e").tf.count()
    tfb = sorted(first.index, key=lambda t: (first[t], -cnt[t], t))
    gap = math.radians(0.6)
    R0, R1 = math.radians(80), math.radians(-80)
    spanR = (R0 - R1) - gap * (len(pick) - 1)
    ep, cur = {}, R0
    for i in range(len(pick)):
        w = spanR * nE[i] / nE.sum()
        ep[i] = [cur, cur - w, cur]
        cur -= w + gap
    T0, T1 = math.radians(100), math.radians(260)
    spanL = (T1 - T0) - gap * 0.5 * (len(tfb) - 1)
    tp, cur = {}, T0
    for t in tfb:
        w = spanL / len(tfb)
        tp[t] = [cur, cur + w, cur]
        cur += w + gap * 0.5

    fig, ax = plt.subplots(figsize=(16, 13))
    ax.set_aspect("equal")
    ax.axis("off")
    ylast = 9
    for i, r in enumerate(pick):
        s, e, _ = ep[i]
        ax.add_patch(Wedge((0, 0), 1.0, math.degrees(e), math.degrees(s), width=0.07, color=HUBCOL[i]))
        x, y = pol((s + e) / 2, 1.06)
        y = min(y, ylast - 0.15)
        ylast = y
        ax.text(x, y, f"{lab(r.region1, r.region1_promoter_gene_ids)} ↔ {lab(r.region2, r.region2_promoter_gene_ids)}\n"
                      f"{r.distance_kb:g} kb · {r.n} TFs" + (" · on a 0.7 loop" if getattr(r, "_6") else ""), ha="left", va="center", fontsize=11)
    for t in tfb:
        s, e, _ = tp[t]
        ax.add_patch(Wedge((0, 0), 1.0, math.degrees(s), math.degrees(e), width=0.05, color=plt.get_cmap("Greys")(0.25 + 0.75 * cnt[t] / len(pick))))
        m = (s + e) / 2
        ax.text(*pol(m, 1.03), t, ha="right", va="center", rotation=math.degrees(m) - 180, rotation_mode="anchor", fontsize=10.5)
    for t, i in sorted(links, key=lambda z: z[1], reverse=True):
        ts, te, tc = tp[t]
        a1, a2 = tc, tc + (te - ts) / cnt[t]
        tp[t][2] = a2
        hs, he, hc = ep[i]
        b1, b2 = hc, hc - (hs - he) / nE[i]
        ep[i][2] = b2
        ribbon(ax, a1, a2, b1, b2, HUBCOL[i], alpha=0.6)
    ax.set_xlim(-1.35, 2.15)
    ax.set_ylim(-1.2, 1.2)
    style.save(fig, out, "f_top_shared_edges_chord", bbox_inches=style.content_bbox(fig, 0.05))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
