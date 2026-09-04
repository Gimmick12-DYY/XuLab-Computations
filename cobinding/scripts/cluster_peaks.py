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
    print(f"[filter] {n0:,} -> {len(df):,} edges (FDR<={args.qval}, coaccess>={args.coaccess_min}, "
          f"support>={args.min_support})", flush=True)
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
                 f"min_clique={args.min_clique} share={args.share}\n")
    print(f"[done] {multi} multi-clique modules; outputs under {args.out_dir}")

    if args.plot:
        _plot(G, cliques, mods, types, genes, args.out_dir, args.out_dir.name)
    return 0


def _plot(G, cliques, mods, types, genes, out_dir: Path, tf: str) -> None:
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
    _plot_network(G, cliques, mods, types, genes, out_dir, tf)
    _plot_chromosomes(mods, genes, out_dir, tf)
    print(f"[plot] wrote {out_dir/'clique_sizes.png'}, {out_dir/'module_sizes.png'}, "
          f"{out_dir/'tf_peak_network.png'}, {out_dir/'clique_chromosomes.png'}")


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

    # dodge overlapping spans within a chromosome
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
            # span can be <1 Mb; keep a visible minimum width
            w = max(x1 - x0, 0.55)
            color = tab[b["i"] % len(tab)] if b["nsrc"] > 1 else (0.35, 0.42, 0.55)
            ax.add_patch(Rectangle(
                (x0, y + yoff - 0.11), w, 0.22,
                facecolor=color, edgecolor="0.15", linewidth=0.5, zorder=3,
                alpha=0.92 if b["nsrc"] > 1 else 0.75))
            # individual peaks as ticks
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


def _pack_components(H, seed: int = 2026, gap: float = 3.15):
    """Layout each connected component locally, then tile them on a grid."""
    import numpy as np

    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    n = len(comps)
    cols = max(1, int(np.ceil(np.sqrt(n * 1.2))))
    pos: dict = {}
    centroids: list[tuple[set, tuple[float, float], int]] = []
    for i, comp in enumerate(comps):
        sub = H.subgraph(comp)
        if len(comp) == 1:
            local = {next(iter(comp)): (0.0, 0.0)}
        elif len(comp) <= 6:
            local = nx.circular_layout(sub, scale=0.85)
        else:
            local = nx.spring_layout(sub, seed=seed, weight="coaccess",
                                     k=1.3 / max(len(comp) ** 0.5, 1.0))
            xy = np.array(list(local.values()))
            xy -= xy.mean(axis=0)
            m = float(np.abs(xy).max()) or 1.0
            xy = xy / m * 0.9
            local = {nd: tuple(xy[j]) for j, nd in enumerate(local)}
        row, col = divmod(i, cols)
        ox, oy = col * gap, -row * gap
        for nd, xy in local.items():
            pos[nd] = (float(xy[0]) + ox, float(xy[1]) + oy)
        centroids.append((comp, (ox, oy), len(comp)))
    return pos, centroids


def _draw_region_graph(ax, H, pos, types, color_of, size_of,
                       edge_alpha: float = 0.55) -> None:
    if H.number_of_nodes() == 0:
        return
    widths = [0.6 + 2.6 * float(H[u][v].get("coaccess", 0.05)) for u, v in H.edges()]
    nx.draw_networkx_edges(H, pos, ax=ax, width=widths, alpha=edge_alpha, edge_color="#444444")
    prox = [n for n in H if "proximal" in types.get(n, ())]
    dist = [n for n in H if n not in prox]
    if prox:
        nx.draw_networkx_nodes(
            H, pos, nodelist=prox, node_shape="o", ax=ax,
            node_color=[color_of(n) for n in prox],
            node_size=[size_of(n) for n in prox],
            edgecolors="0.15", linewidths=0.5)
    if dist:
        nx.draw_networkx_nodes(
            H, pos, nodelist=dist, node_shape="s", ax=ax,
            node_color=[color_of(n) for n in dist],
            node_size=[size_of(n) for n in dist],
            edgecolors="0.15", linewidths=0.5)


def _comp_label(comp, genes) -> str:
    g = sorted({x for r in comp for x in genes.get(r, ()) if x != "."})
    g = [x for x in g if not x.startswith("ENSG")] + [x for x in g if x.startswith("ENSG")]
    if g:
        return ";".join(g[:2]) + ("…" if len(g) > 2 else "")
    chroms = {r.split(":")[0] for r in comp}
    return next(iter(chroms)) if len(chroms) == 1 else f"{len(comp)} regions"


def _plot_network(G, cliques, mods, types, genes, out_dir: Path, tf: str) -> None:
    """One node per region; only clique/module members; edges = co-accessibility."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    clique_regs = set().union(*cliques) if cliques else set()
    if not clique_regs:
        print("[plot] no clique regions; skip network"); return
    H = G.subgraph(clique_regs).copy()
    H.remove_nodes_from(list(nx.isolates(H)))
    if H.number_of_nodes() == 0:
        print("[plot] clique subgraph empty; skip network"); return

    reg2mod: dict[str, int] = {}
    for i, (regs, nsrc) in enumerate(mods):
        for r in regs:
            prev = reg2mod.get(r)
            if prev is None:
                reg2mod[r] = i
            else:
                pr, pn = mods[prev]
                if (nsrc, len(regs)) > (pn, len(pr)):
                    reg2mod[r] = i

    tab = plt.cm.tab20.colors
    multi_ids = [i for i, (_, nsrc) in enumerate(mods) if nsrc > 1]
    multi_color = {mid: tab[j % len(tab)] for j, mid in enumerate(multi_ids)}
    single_col = (0.62, 0.66, 0.72)
    multi_regs = {r for r, mid in reg2mod.items() if mid in multi_color}

    def color_of(n):
        mid = reg2mod.get(n)
        if mid is None:
            return single_col
        return multi_color.get(mid, single_col)

    n_clique = Counter(r for c in cliques for r in c)
    multi_comps, single_comps = [], []
    for comp in nx.connected_components(H):
        (multi_comps if any(n in multi_regs for n in comp) else single_comps).append(comp)
    H_multi = H.subgraph(set().union(*multi_comps) if multi_comps else []).copy()
    H_single = H.subgraph(set().union(*single_comps) if single_comps else []).copy()
    pos_m, cents_m = _pack_components(H_multi, gap=3.6)
    pos_s, _ = _pack_components(H_single, gap=2.55)

    fig, axes = plt.subplots(
        2, 1, figsize=(16, 15),
        gridspec_kw={"height_ratios": [1.15, 1.55]})
    ax_m, ax_s = axes
    _draw_region_graph(ax_m, H_multi, pos_m, types, color_of,
                       size_of=lambda n: 200 + 40 * max(n_clique.get(n, 1) - 1, 0))
    for comp, (ox, oy), nsz in cents_m:
        if nsz < 4 and not any(mods[reg2mod[r]][1] > 1 for r in comp if r in reg2mod):
            continue
        ax_m.text(ox, oy - 1.45, _comp_label(comp, genes), ha="center", va="top",
                  fontsize=8.5, color="#222222", clip_on=False)
    ax_m.set_title(
        f"Multi-clique modules  ({H_multi.number_of_nodes()} regions, "
        f"{H_multi.number_of_edges()} links, {len(multi_ids)} modules)",
        fontsize=12, loc="left", pad=6)
    ax_m.axis("off")

    _draw_region_graph(ax_s, H_single, pos_s, types, color_of,
                       size_of=lambda n: 55, edge_alpha=0.45)
    ax_s.set_title(
        f"Single cliques  ({H_single.number_of_nodes()} regions, "
        f"{H_single.number_of_edges()} links, {len(single_comps)} triangles/K4s)",
        fontsize=12, loc="left", pad=6)
    ax_s.axis("off")

    fig.suptitle(
        f"{tf} co-binding regions in cliques/modules\n"
        f"nodes = peaks · edges = co-accessibility · "
        f"{H.number_of_nodes()} regions in {len(cliques)} cliques / {len(mods)} modules · "
        f"isolated pairs omitted",
        fontsize=14, y=0.995)
    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=single_col,
               markeredgecolor="0.15", markersize=8, label="proximal region"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=single_col,
               markeredgecolor="0.15", markersize=7, label="distal region"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=tab[0],
               markeredgecolor="0.15", markersize=8, label="multi-clique module"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=single_col,
               markeredgecolor="0.15", markersize=8, label="single clique"),
    ], loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out_dir / "tf_peak_network.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
