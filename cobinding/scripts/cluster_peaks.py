#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# cluster_peaks.py
#
# Higher-order co-binding structure from a per-TF peak co-accessibility graph
# (build_peak_graph.py). Nodes are peaks/regions; edges are significant
# co-accessible pairs. Two nested levels:
#   * Strict maximal cliques (>= --min-clique): every region co-binds every other
#     (density = 1) -- the fully-coordinated cores.
#   * Overlap modules: strict cliques merged only when they SHARE >= --share
#     regions (default 2 = an edge). A partially-complete higher-order
#     architecture still anchored by complete subgraphs.
# (Louvain community detection was removed -- overlap modules are the cluster unit.)
#
# Each clique/module is annotated with region composition (proximal/distal),
# genes, chromatin states, genomic span, pair scores (coaccess) and FDR.
#
# Outputs (--out-dir): cliques.tsv, modules.tsv, nodes.tsv, summary.txt,
#   + clique_sizes.png / module_sizes.png / tf_peak_network.png /
#     clique_chromosomes.png (--plot).
#   The network figure uses one node per region (peak), keeps only regions that
#   sit in a clique/module, and draws co-accessibility edges among them.
#   The chromosome figure places each big clique (K4+ / multi-clique module)
#   on hg38 chr1–22 + X.
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd

_COORD = re.compile(r"^(chr[0-9A-Za-z]+):(\d+)-(\d+)$")

# hg38 primary assembly lengths; 23 chromosomes = chr1–22 + X
HG38_LEN = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555,
    "chr5": 181538259, "chr6": 170805979, "chr7": 159345973, "chr8": 145138636,
    "chr9": 138394717, "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189, "chr16": 90338345,
    "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
    "chr21": 46709983, "chr22": 50818468, "chrX": 156040895,
}
CHROMS_23 = [f"chr{i}" for i in range(1, 23)] + ["chrX"]


def pair_distance_bp(p1: str, p2: str) -> int | None:
    """Mid-to-mid distance if same chromosome, else None (trans / unparsable)."""
    m1, m2 = _COORD.match(p1), _COORD.match(p2)
    if not m1 or not m2 or m1.group(1) != m2.group(1):
        return None
    mid1 = (int(m1.group(2)) + int(m1.group(3))) // 2
    mid2 = (int(m2.group(2)) + int(m2.group(3))) // 2
    return abs(mid1 - mid2)


def span(peaks) -> tuple[str, int]:
    by_chrom = defaultdict(lambda: [10**12, 0])
    for p in peaks:
        m = _COORD.match(p)
        if not m:
            continue
        c, s, e = m.group(1), int(m.group(2)), int(m.group(3))
        by_chrom[c][0] = min(by_chrom[c][0], s)
        by_chrom[c][1] = max(by_chrom[c][1], e)
    if len(by_chrom) == 1:
        c, (s, e) = next(iter(by_chrom.items()))
        return c, e - s
    return f"{len(by_chrom)}chroms", 0


def annotate(members, types, genes, states) -> dict:
    n_prox = sum(1 for m in members if "proximal" in types.get(m, ()))
    n_dist = sum(1 for m in members if "distal" in types.get(m, ()))
    g = sorted({x for m in members for x in genes.get(m, ()) if x != "."})
    st = Counter(x for m in members for x in states.get(m, ()) if x != ".")
    chrom, sp = span(members)
    return {
        "n_proximal": n_prox, "n_distal": n_dist, "chromosome": chrom, "span_bp": sp,
        "region_types": f"distal:{n_dist};proximal:{n_prox}",
        "genes": ";".join(g) or ".", "n_genes": len(g),
        "states": ";".join(f"{k}:{v}" for k, v in st.most_common()) or ".",
    }


def pair_stats(members, G) -> dict:
    """Coaccess / p / fdr summary over the edges present among `members`."""
    ca, pv, qv = [], [], []
    for a, b in combinations(sorted(members), 2):
        if G.has_edge(a, b):
            d = G[a][b]
            ca.append(d["coaccess"]); pv.append(d["pval"]); qv.append(d["qval"])
    n_edges = len(ca)
    n = len(members)
    dens = n_edges / (n * (n - 1) / 2) if n > 1 else 0.0
    return {
        "n_edges": n_edges, "density": round(dens, 4),
        "min_pair_score": round(min(ca), 6) if ca else 0.0,
        "mean_pair_score": round(sum(ca) / len(ca), 6) if ca else 0.0,
        "max_pair_p": max(pv) if pv else 1.0,
        "max_pair_fdr": max(qv) if qv else 1.0,
    }


def overlap_modules(cliques: list[frozenset], share: int) -> list[set]:
    """Merge maximal cliques that share >= `share` regions; components = modules."""
    CG = nx.Graph()
    CG.add_nodes_from(range(len(cliques)))
    reg2cl = defaultdict(list)
    for i, c in enumerate(cliques):
        for r in c:
            reg2cl[r].append(i)
    for cl in reg2cl.values():
        for i, j in combinations(cl, 2):
            if len(cliques[i] & cliques[j]) >= share:
                CG.add_edge(i, j)
    mods = []
    for comp in nx.connected_components(CG):
        regs = set().union(*[cliques[i] for i in comp])
        mods.append((regs, len(comp)))
    return mods


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edges", type=Path, required=True, help="<tf>.peak_edges.tsv")
    ap.add_argument("--annot", type=Path, default=None, help="<tf>.peak_annot.tsv")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--qval", type=float, default=0.05, help="edge FDR cutoff (<=)")
    ap.add_argument("--coaccess-min", type=float, default=0.0)
    ap.add_argument("--min-support", type=int, default=1)
    ap.add_argument("--min-clique", type=int, default=3)
    ap.add_argument("--share", type=int, default=2,
                    help="cliques merge into a module when they share >= this many regions (default 2)")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--net-min-comp", type=int, default=1,
                    help="network figure: smallest component to draw (1 = the whole graph)")
    ap.add_argument("--net-big", type=int, default=10,
                    help="network figure: components with >= this many regions get their own colour")
    ap.add_argument("--max-dist", type=int, default=1_000_000,
                    help="keep cis pairs with mid-to-mid distance <= this many bp "
                         "(default 1000000 = Cicero 1 Mb window). 0 = no distance filter.")
    args = ap.parse_args()

    annot_path = args.annot or Path(str(args.edges).replace(".peak_edges.tsv", ".peak_annot.tsv"))
    types: dict[str, set] = {}; genes: dict[str, set] = {}; states: dict[str, set] = {}
    if annot_path.is_file():
        for r in pd.read_csv(annot_path, sep="\t").itertuples(index=False):
            types[r.peak] = set(str(r.types).split(",")) if r.types != "." else set()
            genes[r.peak] = set(str(r.genes).split(",")) if r.genes != "." else set()
            states[r.peak] = set(str(r.states).split(",")) if r.states != "." else set()

    df = pd.read_csv(args.edges, sep="\t")
    n0 = len(df)
    df = df[(df["qval"] <= args.qval) & (df["coaccess"] >= args.coaccess_min)
            & (df["n_links"] >= args.min_support)]
    if args.max_dist and args.max_dist > 0:
        keep = []
        for a, b in zip(df["peak1"], df["peak2"]):
            d = pair_distance_bp(str(a), str(b))
            keep.append(d is not None and d <= args.max_dist)
        df = df.iloc[[i for i, ok in enumerate(keep) if ok]]
    print(f"[filter] {n0:,} -> {len(df):,} edges (FDR<={args.qval}, coaccess>={args.coaccess_min}, "
          f"support>={args.min_support}, max_dist={args.max_dist or 'off'})", flush=True)
    if df.empty:
        raise SystemExit("no edges pass filters; loosen thresholds")

    G = nx.Graph()
    has_p = "pval" in df.columns
    for r in df.itertuples(index=False):
        G.add_edge(r.peak1, r.peak2, coaccess=float(r.coaccess),
                   pval=float(r.pval) if has_p else float(r.qval),
                   qval=float(r.qval))
    print(f"[graph] regions={G.number_of_nodes():,} edges={G.number_of_edges():,}", flush=True)

    # strict maximal cliques
    cliques = [frozenset(c) for c in nx.find_cliques(G) if len(c) >= args.min_clique]
    cliques.sort(key=len, reverse=True)
    csize = Counter(len(c) for c in cliques)
    print(f"[cliques] {len(cliques)} (>= {args.min_clique}); sizes={dict(sorted(csize.items()))}", flush=True)

    # overlap modules
    mods = overlap_modules(cliques, args.share)
    mods.sort(key=lambda m: -len(m[0]))
    print(f"[modules] {len(mods)} overlap modules (share>={args.share}); "
          f"largest={len(mods[0][0]) if mods else 0}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    kcore = nx.core_number(G); deg = dict(G.degree())
    reg_ncliques = Counter(r for c in cliques for r in c)

    # cliques.tsv
    with (args.out_dir / "cliques.tsv").open("w") as fh:
        cols = ["clique_id", "n_regions", "chromosome", "span_bp", "region_types", "genes",
                "min_pair_score", "mean_pair_score", "max_pair_p", "max_pair_fdr", "density",
                "states", "regions"]
        fh.write("\t".join(cols) + "\n")
        for i, c in enumerate(cliques):
            a = annotate(c, types, genes, states); ps = pair_stats(c, G)
            fh.write(f"C{i+1:05d}\t{len(c)}\t{a['chromosome']}\t{a['span_bp']}\t{a['region_types']}\t"
                     f"{a['genes']}\t{ps['min_pair_score']}\t{ps['mean_pair_score']}\t{ps['max_pair_p']:.6g}\t"
                     f"{ps['max_pair_fdr']:.6g}\t{ps['density']}\t{a['states']}\t{';'.join(sorted(c))}\n")

    # modules.tsv
    reg_nmodules = Counter()
    with (args.out_dir / "modules.tsv").open("w") as fh:
        cols = ["module_id", "n_regions", "n_source_cliques", "n_edges", "density", "chromosome",
                "span_bp", "region_types", "genes", "min_pair_score", "mean_pair_score",
                "max_pair_p", "max_pair_fdr", "states", "regions"]
        fh.write("\t".join(cols) + "\n")
        for i, (regs, nsrc) in enumerate(mods):
            for r in regs:
                reg_nmodules[r] += 1
            a = annotate(regs, types, genes, states); ps = pair_stats(regs, G)
            fh.write(f"M{i+1:05d}\t{len(regs)}\t{nsrc}\t{ps['n_edges']}\t{ps['density']}\t{a['chromosome']}\t"
                     f"{a['span_bp']}\t{a['region_types']}\t{a['genes']}\t{ps['min_pair_score']}\t"
                     f"{ps['mean_pair_score']}\t{ps['max_pair_p']:.6g}\t{ps['max_pair_fdr']:.6g}\t"
                     f"{a['states']}\t{';'.join(sorted(regs))}\n")

    # nodes.tsv
    with (args.out_dir / "nodes.tsv").open("w") as fh:
        fh.write("peak\tdegree\tkcore\tn_cliques\tn_modules\ttypes\tgenes\tstates\n")
        for n in sorted(G.nodes()):
            fh.write(f"{n}\t{deg[n]}\t{kcore[n]}\t{reg_ncliques.get(n,0)}\t{reg_nmodules.get(n,0)}\t"
                     f"{','.join(sorted(types.get(n, []))) or '.'}\t"
                     f"{','.join(sorted(genes.get(n, []))) or '.'}\t"
                     f"{','.join(sorted(states.get(n, []))) or '.'}\n")

    msize = Counter(len(r) for r, _ in mods)
    multi = sum(1 for _, nsrc in mods if nsrc > 1)
    with (args.out_dir / "summary.txt").open("w") as fh:
        fh.write(f"regions\t{G.number_of_nodes()}\nedges\t{G.number_of_edges()}\n")
        fh.write(f"cliques(>= {args.min_clique})\t{len(cliques)}  sizes={dict(sorted(csize.items()))}\n")
        fh.write(f"overlap_modules\t{len(mods)}  sizes={dict(sorted(msize.items()))}\n")
        fh.write(f"multi_clique_modules(n_source>1)\t{multi}\n")
        fh.write(f"largest_module\t{len(mods[0][0]) if mods else 0}\n")
        fh.write(f"filters\tFDR<={args.qval} coaccess>={args.coaccess_min} support>={args.min_support} "
                 f"min_clique={args.min_clique} share={args.share} max_dist={args.max_dist}\n")
    print(f"[done] {multi} multi-clique modules; outputs under {args.out_dir}")

    if args.plot:
        _plot(G, cliques, mods, types, genes, args.out_dir, args.out_dir.name,
              args.net_min_comp, args.net_big)
    return 0


def _plot(G, cliques, mods, types, genes, out_dir: Path, tf: str,
          net_min_comp: int = 1, net_big: int = 10) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] skipped: {e}"); return
    for name, sizes, xl in [("clique_sizes", [len(c) for c in cliques], "clique size"),
                            ("module_sizes", [len(r) for r, _ in mods], "module size (regions)")]:
        if not sizes:
            continue
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(sizes, bins=range(min(sizes), max(sizes) + 2))
        ax.set_xlabel(xl); ax.set_ylabel("count"); ax.set_title(f"{tf} {xl}")
        fig.tight_layout(); fig.savefig(out_dir / f"{name}.png", dpi=150); plt.close(fig)
    _plot_network(G, cliques, mods, types, genes, out_dir, tf, net_min_comp, net_big)
    _plot_network(G, cliques, mods, types, genes, out_dir, tf, max(net_big, 10), net_big,
                  clique_contrast=True, out_name="tf_peak_network_clusters.png",
                  node_sep=0.42)
    _plot_network(G, cliques, mods, types, genes, out_dir, tf, max(net_big, 10), net_big,
                  clique_contrast=True, color_cliques=True,
                  out_name="tf_peak_network_cliques.png", node_sep=0.42)
    _plot_chromosomes(mods, genes, out_dir, tf)
    print(f"[plot] wrote {out_dir/'clique_sizes.png'}, {out_dir/'module_sizes.png'}, "
          f"{out_dir/'tf_peak_network.png'}, {out_dir/'tf_peak_network_clusters.png'}, "
          f"{out_dir/'tf_peak_network_cliques.png'}, {out_dir/'clique_chromosomes.png'}")


def _locus(regs) -> tuple[str, int, int] | None:
    """Single-chromosome span (chrom, start, end) from region ids, or None."""
    by = defaultdict(lambda: [10**12, 0])
    for p in regs:
        m = _COORD.match(p)
        if not m:
            continue
        c, s, e = m.group(1), int(m.group(2)), int(m.group(3))
        by[c][0] = min(by[c][0], s)
        by[c][1] = max(by[c][1], e)
    if len(by) != 1:
        return None
    c, (s, e) = next(iter(by.items()))
    return c, s, e


def _cluster_ann(comp, genes) -> tuple[str, str, str, int, int]:
    """(short label, chrom, start, end) for a connected component."""
    loc = _locus(comp)
    g = _gene_short(comp, genes)
    if loc:
        chrom, s, e = loc
    else:
        chroms = sorted({r.split(":")[0] for r in comp})
        chrom, s, e = (chroms[0] if len(chroms) == 1 else f"{len(chroms)}chroms"), 0, 0
    if g:
        lab = f"{g}\n{chrom}  {len(comp)} reg"
    elif s or e:
        lab = f"{chrom}:{s/1e6:.1f}-{e/1e6:.1f} Mb\n{len(comp)} reg"
    else:
        lab = f"{chrom}\n{len(comp)} reg"
    return lab, chrom, s, e


def _gene_short(regs, genes) -> str:
    g = sorted({x for r in regs for x in genes.get(r, ()) if x and x != "."})
    g = [x for x in g if not x.startswith("ENSG")] + [x for x in g if x.startswith("ENSG")]
    if not g:
        return ""
    return ";".join(g[:2]) + ("…" if len(g) > 2 else "")


def _plot_chromosomes(mods, genes, out_dir: Path, tf: str) -> None:
    """Ideogram of chr1–22 + X with big-clique (K4+ / multi-clique) spans."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    big = []
    for i, (regs, nsrc) in enumerate(mods):
        if len(regs) < 4 and nsrc <= 1:
            continue
        loc = _locus(regs)
        if loc is None or loc[0] not in HG38_LEN:
            continue
        chrom, start, end = loc
        big.append({
            "i": i, "regs": regs, "nsrc": nsrc, "n": len(regs),
            "chrom": chrom, "start": start, "end": end,
            "label": _gene_short(regs, genes),
        })
    if not big:
        print("[plot] no big cliques on chr1–22/X; skip chromosomes"); return

    tab = plt.cm.tab20.colors
    by_chr: dict[str, list] = defaultdict(list)
    for b in big:
        by_chr[b["chrom"]].append(b)

    lanes: dict[str, dict[int, int]] = {}
    n_lanes: dict[str, int] = {}
    for chrom, items in by_chr.items():
        items = sorted(items, key=lambda x: (x["start"], -x["end"]))
        ends: list[int] = []
        lanes[chrom] = {}
        for b in items:
            placed = False
            for li, le in enumerate(ends):
                if b["start"] >= le + 200_000:
                    ends[li] = b["end"]
                    lanes[chrom][b["i"]] = li
                    placed = True
                    break
            if not placed:
                ends.append(b["end"])
                lanes[chrom][b["i"]] = len(ends) - 1
        n_lanes[chrom] = max(len(ends), 1)

    row_h = 1.0
    fig_h = max(9.5, 0.42 * len(CHROMS_23) + 1.8)
    fig, ax = plt.subplots(figsize=(13.5, fig_h))
    xmax = max(HG38_LEN.values()) / 1e6

    for yi, chrom in enumerate(CHROMS_23):
        y = -yi * row_h
        clen = HG38_LEN[chrom] / 1e6
        ax.add_patch(FancyBboxPatch(
            (0, y - 0.16), clen, 0.32,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor="#e6e8eb", edgecolor="#9aa0a6", linewidth=0.7,
            mutation_aspect=0.4))
        ax.text(-2.2, y, chrom.replace("chr", ""), ha="right", va="center",
                fontsize=10, fontweight="medium", color="#222")
        nl = n_lanes.get(chrom, 1)
        for b in by_chr.get(chrom, []):
            li = lanes[chrom][b["i"]]
            yoff = (li - (nl - 1) / 2.0) * 0.22
            x0, x1 = b["start"] / 1e6, b["end"] / 1e6
            w = max(x1 - x0, 0.55)
            color = tab[b["i"] % len(tab)] if b["nsrc"] > 1 else (0.35, 0.42, 0.55)
            ax.add_patch(Rectangle(
                (x0, y + yoff - 0.11), w, 0.22,
                facecolor=color, edgecolor="0.15", linewidth=0.5, zorder=3,
                alpha=0.92 if b["nsrc"] > 1 else 0.75))
            for p in b["regs"]:
                m = _COORD.match(p)
                if m:
                    mid = (int(m.group(2)) + int(m.group(3))) / 2 / 1e6
                    ax.plot([mid, mid], [y + yoff - 0.11, y + yoff + 0.11],
                            color="0.1", lw=0.7, zorder=4)
            lab = b["label"] or f"{b['n']} reg"
            side = 1 if (b["i"] + yi) % 2 == 0 else -1
            ax.annotate(
                lab, xy=(x0 + w / 2, y + yoff),
                xytext=(0, side * 10), textcoords="offset points",
                ha="center", va="bottom" if side > 0 else "top",
                fontsize=7, color="#111",
                arrowprops=dict(arrowstyle="-", color="#888", lw=0.5))

    ax.set_xlim(-8, xmax + 8)
    ax.set_ylim(-row_h * (len(CHROMS_23) - 0.3), 0.7)
    ax.set_xlabel("Position (Mb, hg38)", fontsize=11)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    n_multi = sum(1 for b in big if b["nsrc"] > 1)
    ax.set_title(
        f"{tf} big cliques on 23 chromosomes\n"
        f"{len(big)} loci (≥4 regions or multi-clique) · "
        f"{n_multi} multi-clique modules (colored) · "
        f"K4 single cliques (slate) · bars = genomic span",
        fontsize=13, loc="left", pad=8)
    fig.tight_layout()
    fig.savefig(out_dir / "clique_chromosomes.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _component_layout(sub, seed: int = 2026):
    """Force-directed coordinates for one component, unit-normalized. igraph if present."""
    import numpy as np

    nodes = list(sub)
    n = len(nodes)
    if n == 1:
        return {nodes[0]: (0.0, 0.0)}
    if n == 2:
        return {nodes[0]: (-1.0, 0.0), nodes[1]: (1.0, 0.0)}
    xy = None
    try:
        import igraph as ig
        idx = {v: i for i, v in enumerate(nodes)}
        g = ig.Graph(n=n, edges=[(idx[u], idx[v]) for u, v in sub.edges()])
        xy = np.asarray(g.layout_fruchterman_reingold(niter=400 if n < 400 else 200).coords, float)
    except Exception:                                    # noqa: BLE001 - fall back to networkx
        loc = (nx.circular_layout(sub) if n <= 6 else
               nx.spring_layout(sub, seed=seed, k=1.3 / max(n ** 0.5, 1.0)))
        xy = np.array([loc[v] for v in nodes], float)
    xy = xy - xy.mean(axis=0)
    m = float(np.abs(xy).max()) or 1.0
    xy = xy / m                                          # radius 1 before scaling
    return {v: (float(xy[i, 0]), float(xy[i, 1])) for i, v in enumerate(nodes)}


def _spread_local(loc, min_d: float = 0.38, iters: int = 60):
    """Push nodes apart until nearest-neighbour distance is at least `min_d`.
    Does NOT re-normalize, so the component grows and the extra space is kept."""
    import numpy as np

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
            if not js.size:
                continue
            vec, d = delta[i, js], dist[i, js][:, None]
            push[i] += ((min_d - d) * vec / np.maximum(d, 1e-9)).sum(axis=0)
        xy += 0.4 * push
    xy -= xy.mean(axis=0)
    return {v: (float(xy[i, 0]), float(xy[i, 1])) for i, v in enumerate(nodes)}


def _pack_discs(radii, pad: float = 0.45):
    """Greedy spiral packing of discs, largest first -> centres. Big clusters land in
    the middle and the many two-node pairs form the surrounding halo."""
    import numpy as np

    order = sorted(range(len(radii)), key=lambda i: -radii[i])
    cx = np.empty(len(radii)); cy = np.empty(len(radii))
    px, py, pr = [], [], []                              # already-placed discs
    r_start = 0.0
    for k, i in enumerate(order):
        r = radii[i]
        if not px:
            cx[i] = cy[i] = 0.0
            px.append(0.0); py.append(0.0); pr.append(r)
            continue
        ax_, ay_, ar_ = np.array(px), np.array(py), np.array(pr)
        R = max(r_start, 0.0)
        step = max(r * 0.7, 0.35)
        placed = False
        while not placed:
            ntheta = max(8, int(2 * np.pi * R / max(step, 1e-6)))
            th = (np.arange(ntheta) / ntheta * 2 * np.pi + 0.61803 * k) % (2 * np.pi)
            for t in th:
                x, y = R * np.cos(t), R * np.sin(t)
                if np.all((x - ax_) ** 2 + (y - ay_) ** 2 >= (ar_ + r + pad) ** 2):
                    cx[i], cy[i] = x, y
                    px.append(x); py.append(y); pr.append(r)
                    r_start = max(0.0, R - 3 * r)        # next disc starts near this ring
                    placed = True
                    break
            R += step
    return cx, cy


def _comp_label(comp, genes) -> str:
    g = sorted({x for r in comp for x in genes.get(r, ()) if x != "."})
    g = [x for x in g if not x.startswith("ENSG")] + [x for x in g if x.startswith("ENSG")]
    if g:
        return ";".join(g[:2]) + ("…" if len(g) > 2 else "")
    chroms = {r.split(":")[0] for r in comp}
    return next(iter(chroms)) if len(chroms) == 1 else f"{len(comp)} regions"


def _clique_colors(cliques, keep):
    """One distinct hue per clique that touches `keep`. Overlapping cliques are
    placed far apart on the hue wheel so they cannot look like the same colour."""
    import matplotlib.pyplot as plt

    live = [c for c in cliques if c & keep]
    n = len(live)
    if not n:
        return live, {}
    # unique hues, golden-ratio stepped so neighbours on the list are not neighbours on the wheel
    step = 0.61803398875
    hues = [(0.02 + i * step) % 0.92 for i in range(n)]
    hit = defaultdict(list)
    for i, c in enumerate(live):
        for r in c:
            hit[r].append(i)
    # if two overlapping cliques landed close in hue, swap one to the farthest free slot
    for ids in hit.values():
        for i, j in combinations(ids, 2):
            if min(abs(hues[i] - hues[j]), 0.92 - abs(hues[i] - hues[j])) < 0.08:
                hues[j] = (hues[i] + 0.5) % 0.92
    hsv = plt.cm.hsv
    return live, {i: hsv(hues[i]) for i in range(n)}


def _label_center_clusters(ax, comps, radii, cx, cy, genes, ink, out_dir,
                           min_n: int = 10, top: int = 24) -> int:
    """Label the largest components (packed into the centre) and write a TSV."""
    rows = []
    n_lab = 0
    for i, comp in enumerate(comps):
        if len(comp) < min_n:
            continue
        lab, chrom, s, e = _cluster_ann(comp, genes)
        genes_s = _gene_short(comp, genes) or "."
        rows.append((i + 1, len(comp), chrom, s, e, genes_s.replace("\n", " ")))
        if n_lab < top:
            ax.text(cx[i], cy[i] + radii[i] + 0.12, lab,
                    ha="center", va="bottom", fontsize=6.5, color=ink,
                    zorder=6, linespacing=1.15,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.75))
            n_lab += 1
    tsv = out_dir / "cluster_annotations.tsv"
    with tsv.open("w") as fh:
        fh.write("rank\tn_regions\tchrom\tstart\tend\tgenes\n")
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
    print(f"[plot] labelled {n_lab} centre clusters; {len(rows)} ≥{min_n} in {tsv.name}")
    return n_lab


def _plot_network(G, cliques, mods, types, genes, out_dir: Path, tf: str,
                  min_comp: int = 1, big: int = 10, clique_contrast: bool = False,
                  out_name: str = "tf_peak_network.png", node_sep: float = 0.0,
                  color_cliques: bool = False) -> None:
    """One canvas, one node per region, every edge drawn.

    Each connected component is force-laid-out on its own, then disc-packed
    largest-first. Default colouring: components >= `big` get their own colour.
    `clique_contrast=True` (the clusters figure): clique cores are bold black,
    every surrounding pair is light grey, on a light background so the cores pop.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    allc = sorted(nx.connected_components(G), key=len, reverse=True)
    comps = [c for c in allc if len(c) >= min_comp]
    if not comps:
        print("[plot] no components; skip network"); return
    drop = [c for c in allc if len(c) < min_comp]
    n_drop_c, n_drop_r = len(drop), sum(len(c) for c in drop)
    keep = set().union(*comps)
    H = G.subgraph(keep)
    n_pair = sum(1 for c in comps if len(c) == 2)

    # local force layout per component; node_sep > 0 pushes dots apart (clusters figure)
    radii, locals_ = [], []
    for c in comps:
        loc = _component_layout(H.subgraph(c))
        if node_sep > 0:
            loc = _spread_local(loc, min_d=node_sep)
            r = max((x * x + y * y) ** 0.5 for x, y in loc.values()) + node_sep * 0.8
        else:
            r = 0.5 + 0.62 * len(c) ** 0.5
        radii.append(r)
        locals_.append(loc)
    cx, cy = _pack_discs(radii, pad=0.85 if node_sep > 0 else 0.45)
    pos = {}
    fill = 1.0 if node_sep > 0 else 0.82
    for i, loc in enumerate(locals_):
        for v, (x, y) in loc.items():
            pos[v] = (x * fill + cx[i], y * fill + cy[i]) if node_sep > 0 else \
                     (x * radii[i] * fill + cx[i], y * radii[i] * fill + cy[i])

    clique_nodes = set().union(*cliques) if cliques else set()
    clique_edges = set()
    for c in cliques:
        for a, b in combinations(c, 2):
            if H.has_edge(a, b):
                clique_edges.add(frozenset((a, b)))

    fig_bg = "#f7f7f5" if clique_contrast else "#23272e"
    side = 24 if node_sep > 0 else 19
    fig, ax = plt.subplots(figsize=(side, side), facecolor=fig_bg)
    ax.set_facecolor(fig_bg)

    if clique_contrast:
        grey, black = "#c8ccd2", "#111111"
        rest_e = [(pos[u], pos[v]) for u, v in H.edges()
                  if frozenset((u, v)) not in clique_edges]
        if rest_e:
            ax.add_collection(LineCollection(rest_e, colors=grey, linewidths=0.6,
                                             alpha=0.9, zorder=1))
        rest_n = [n for n in H if n not in clique_nodes]
        cliq_n = [n for n in H if n in clique_nodes]
        if rest_n:
            xy = np.array([pos[v] for v in rest_n])
            ax.scatter(xy[:, 0], xy[:, 1], s=16, c=grey, linewidths=0, zorder=2)
        n_cliq_comp = sum(1 for c in comps if any(v in clique_nodes for v in c))
        ink = "#222222"
        if color_cliques:
            live, cmap = _clique_colors(cliques, keep)
            # largest clique wins when a region/edge is shared
            node_col, ncl = {}, Counter()
            for i, c in sorted(enumerate(live), key=lambda kv: len(kv[1])):
                for r in c:
                    if r in keep:
                        node_col[r] = cmap[i]
                        ncl[r] += 1
            for i, c in enumerate(live):
                segs = [(pos[a], pos[b]) for a, b in combinations(c, 2)
                        if H.has_edge(a, b) and a in pos and b in pos]
                if segs:
                    ax.add_collection(LineCollection(segs, colors=[cmap[i]],
                                                     linewidths=2.3, alpha=1.0, zorder=3))
            if cliq_n:
                xy = np.array([pos[v] for v in cliq_n])
                hinge = [ncl.get(v, 1) > 1 for v in cliq_n]
                ax.scatter(xy[:, 0], xy[:, 1], s=28,
                           c=[node_col.get(v, black) for v in cliq_n],
                           linewidths=[0.9 if h else 0 for h in hinge],
                           edgecolors=["#111" if h else "none" for h in hinge],
                           zorder=4)
            n_hinge = sum(1 for v in cliq_n if ncl.get(v, 1) > 1)
            n_lab = _label_center_clusters(ax, comps, radii, cx, cy, genes, ink,
                                           out_dir, min_n=big)
            ax.set_title(
                f"{tf} co-binding clusters — each clique its own colour\n"
                f"{H.number_of_nodes():,} regions · {H.number_of_edges():,} edges · "
                f"{len(comps):,} components ≥{min_comp} (largest {len(comps[0])})\n"
                f"{len(live)} cliques coloured · {len(cliq_n):,} clique regions · "
                f"{n_hinge} hinge regions (black outline, in >1 clique) · "
                f"labels = {n_lab} largest clusters (gene + chrom)",
                fontsize=15, color=ink, loc="left", pad=14)
            handles = [
                Line2D([0], [0], color="#e8433f", lw=2.4, label="one clique"),
                Line2D([0], [0], color="#33b5e5", lw=2.4, label="another clique"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor="#eda320",
                       markeredgecolor="#111", markeredgewidth=0.8, markersize=8,
                       label="hinge (shared by ≥2 cliques)"),
                Line2D([0], [0], color=grey, lw=1.2, label="surrounding pair"),
            ]
        else:
            cliq_e = [(pos[u], pos[v]) for u, v in H.edges()
                      if frozenset((u, v)) in clique_edges]
            if cliq_e:
                ax.add_collection(LineCollection(cliq_e, colors=black, linewidths=2.3,
                                                 alpha=1.0, zorder=3))
            if cliq_n:
                xy = np.array([pos[v] for v in cliq_n])
                ax.scatter(xy[:, 0], xy[:, 1], s=26, c=black, linewidths=0, zorder=4)
            ax.set_title(
                f"{tf} co-binding clusters — cliques in black, surrounding pairs in grey\n"
                f"{H.number_of_nodes():,} regions · {H.number_of_edges():,} edges · "
                f"{len(comps):,} components ≥{min_comp} "
                f"(largest {len(comps[0])})\n"
                f"{len(cliq_n):,} regions / {len(cliq_e):,} edges in a clique (bold black) · "
                f"{n_cliq_comp} components contain a clique · "
                f"{n_drop_c:,} smaller components hidden",
                fontsize=15, color=ink, loc="left", pad=14)
            handles = [
                Line2D([0], [0], color=black, lw=2.4, label="clique (every pair linked)"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor=black,
                       markersize=8, label="clique region"),
                Line2D([0], [0], color=grey, lw=1.2, label="surrounding pair"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor=grey,
                       markersize=6, label="non-clique region"),
            ]
    else:
        deg = dict(H.degree())
        palette = ["#f5793a", "#4fd1a5", "#e8433f", "#63d345", "#b344e8", "#33b5e5",
                   "#f0c419", "#ff7ab6", "#8ad4ff", "#c0f24a", "#ff9f5a", "#6ee7d0"]
        slate, halo, hub = "#7f8794", "#5b74d1", "#ffc61e"
        col, size = {}, {}
        big_i = 0
        for c, r in zip(comps, radii):
            if len(c) >= big:
                base = palette[big_i % len(palette)]; big_i += 1
            elif len(c) > 2:
                base = slate
            else:
                base = halo
            dmax = max(deg[v] for v in c)
            for v in c:
                is_hub = len(c) >= big and deg[v] == dmax and dmax >= 3
                col[v] = hub if is_hub else base
                size[v] = (95 if is_hub else 10 + 7 * deg[v]) if len(c) >= big else \
                          (9 if len(c) == 2 else 12)
        seg = [(pos[u], pos[v]) for u, v in H.edges()]
        ax.add_collection(LineCollection(seg, colors="#9aa2ae", linewidths=0.35,
                                         alpha=0.55, zorder=1))
        nodes = list(H)
        xy = np.array([pos[v] for v in nodes])
        ax.scatter(xy[:, 0], xy[:, 1], s=[size[v] for v in nodes],
                   c=[col[v] for v in nodes], linewidths=0, zorder=2)
        n_big = sum(1 for c in comps if len(c) >= big)
        n_mid = sum(1 for c in comps if 2 < len(c) < big)
        tail = (f"{n_pair:,} isolated pairs (blue halo)" if n_pair else
                f"{n_drop_c:,} components <{min_comp} regions hidden "
                f"({n_drop_r:,} regions, {100*n_drop_r/G.number_of_nodes():.0f}% of the graph)")
        ink = "#e8eaed"
        ax.set_title(
            f"{tf} co-binding region network — every edge drawn\n"
            f"{H.number_of_nodes():,} regions · {H.number_of_edges():,} co-accessibility edges · "
            f"{len(comps):,} connected components (largest {len(comps[0])})\n"
            f"{n_big} components ≥{big} regions (coloured) · {n_mid:,} of 3–{big-1} (grey) · {tail}",
            fontsize=15, color=ink, loc="left", pad=14)
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=hub,
                   markersize=11, label="cluster hub (highest degree)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=palette[1],
                   markersize=8, label=f"component ≥{big} regions"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=slate,
                   markersize=7, label=f"component 3–{big-1} regions"),
        ]
        if n_pair:
            handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=halo,
                                  markersize=6, label="isolated pair"))

    ax.set_aspect("equal"); ax.axis("off")
    ax.autoscale_view()
    leg = ax.legend(handles=handles, loc="lower right", frameon=False,
                    fontsize=11, labelcolor=ink)
    for t in leg.get_texts():
        t.set_color(ink)
    out_png = out_dir / out_name
    fig.savefig(out_png, dpi=170, bbox_inches="tight", facecolor=fig_bg)
    # Vector sibling for Illustrator / Inkscape editing (same stem, .svg).
    out_svg = out_png.with_suffix(".svg")
    fig.savefig(out_svg, format="svg", bbox_inches="tight", facecolor=fig_bg)
    plt.close(fig)
    tag = " (per-clique colour)" if color_cliques else (" (clique contrast)" if clique_contrast else "")
    print(f"[plot] {out_name} + {out_svg.name}: {H.number_of_nodes():,} regions, "
          f"{H.number_of_edges():,} edges, {len(comps):,} components{tag}")


if __name__ == "__main__":
    raise SystemExit(main())
