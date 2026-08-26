#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# cluster_peaks.py
#
# Higher-order co-binding clusters from a per-TF peak co-accessibility graph
# (build_peak_graph.py). Nodes are peaks; a cluster = a set of RBBP4 regions that
# mutually co-bind. Answers "a-b, b-c, a-c -> a hub" via:
#   * community detection (Louvain; networkx fallback)
#   * maximal cliques (fully co-accessible peak sets; needs the full pp+dd+pd net)
#   * k-core + connected components
# Each cluster is annotated with its proximal genes, distal chromatin states,
# proximal/distal composition, and genomic span.
#
# Outputs (--out-dir): clusters.tsv, cliques.tsv, nodes.tsv, summary.txt,
#   + tf_peak_network.png / clique_sizes.png (--plot).
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd

_COORD = re.compile(r"^(chr[0-9A-Za-z]+):(\d+)-(\d+)$")


def louvain(G: nx.Graph) -> dict:
    try:
        import community as community_louvain
        return community_louvain.best_partition(G, weight="weight", random_state=2026)
    except ImportError:
        print("[communities] python-louvain not found; networkx greedy modularity", flush=True)
        comms = nx.community.greedy_modularity_communities(G, weight="weight")
        return {n: i for i, c in enumerate(comms) for n in c}


def leiden(G: nx.Graph) -> dict:
    import igraph as ig
    import leidenalg
    nodes = list(G.nodes()); idx = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=[(idx[u], idx[v]) for u, v in G.edges()],
                 edge_attrs={"weight": [G[u][v]["weight"] for u, v in G.edges()]})
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                    weights="weight", seed=2026)
    return {nodes[i]: part.membership[i] for i in range(len(nodes))}


def span(peaks) -> str:
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
        return f"{c}:{s}-{e} ({(e - s) / 1e3:.0f}kb)"
    return f"{len(by_chrom)} chroms"


def annotate(members, types, genes, states) -> dict:
    n_prox = sum(1 for m in members if "proximal" in types.get(m, ()))
    n_dist = sum(1 for m in members if "distal" in types.get(m, ()))
    g = sorted({x for m in members for x in genes.get(m, ()) if x != "."})
    st = Counter(x for m in members for x in states.get(m, ()) if x != ".")
    return {
        "n_peaks": len(members), "n_proximal": n_prox, "n_distal": n_dist,
        "genes": ",".join(g[:30]) or ".", "n_genes": len(g),
        "states": ";".join(f"{k}:{v}" for k, v in st.most_common()) or ".",
        "span": span(members),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edges", type=Path, required=True, help="<tf>.peak_edges.tsv")
    ap.add_argument("--annot", type=Path, default=None, help="<tf>.peak_annot.tsv")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--qval", type=float, default=0.05)
    ap.add_argument("--coaccess-min", type=float, default=0.0)
    ap.add_argument("--min-support", type=int, default=1)
    ap.add_argument("--weight", choices=["coaccess"], default="coaccess")
    ap.add_argument("--method", choices=["louvain", "leiden"], default="louvain")
    ap.add_argument("--min-clique", type=int, default=3)
    ap.add_argument("--min-cluster", type=int, default=3, help="report clusters with >= this many peaks")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--max-plot-nodes", type=int, default=1500)
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
    df = df[(df["qval"] < args.qval) & (df["coaccess"] >= args.coaccess_min)
            & (df["n_links"] >= args.min_support)]
    print(f"[filter] {n0:,} -> {len(df):,} edges (q<{args.qval}, coaccess>={args.coaccess_min}, "
          f"support>={args.min_support})", flush=True)
    if df.empty:
        raise SystemExit("no edges pass filters; loosen thresholds")

    G = nx.Graph()
    for r in df.itertuples(index=False):
        G.add_edge(r.peak1, r.peak2, weight=float(r.coaccess))
    print(f"[graph] peaks={G.number_of_nodes():,} edges={G.number_of_edges():,}", flush=True)

    part = leiden(G) if args.method == "leiden" else louvain(G)
    kcore = nx.core_number(G)
    deg = dict(G.degree())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # nodes
    with (args.out_dir / "nodes.tsv").open("w") as fh:
        fh.write("peak\tcommunity\tkcore\tdegree\ttypes\tgenes\tstates\n")
        for n in sorted(G.nodes()):
            fh.write(f"{n}\t{part.get(n, -1)}\t{kcore[n]}\t{deg[n]}\t"
                     f"{','.join(sorted(types.get(n, []))) or '.'}\t"
                     f"{','.join(sorted(genes.get(n, []))) or '.'}\t"
                     f"{','.join(sorted(states.get(n, []))) or '.'}\n")

    # communities (higher-order clusters)
    members = defaultdict(list)
    for n, c in part.items():
        members[c].append(n)
    clusters = sorted(members.items(), key=lambda kv: -len(kv[1]))
    with (args.out_dir / "clusters.tsv").open("w") as fh:
        fh.write("cluster\tn_peaks\tn_proximal\tn_distal\tn_genes\tspan\tstates\tgenes\n")
        n_reported = 0
        for c, mem in clusters:
            if len(mem) < args.min_cluster:
                continue
            a = annotate(mem, types, genes, states)
            fh.write(f"{c}\t{a['n_peaks']}\t{a['n_proximal']}\t{a['n_distal']}\t{a['n_genes']}\t"
                     f"{a['span']}\t{a['states']}\t{a['genes']}\n")
            n_reported += 1
    print(f"[clusters] {len(clusters)} communities ({n_reported} with >= {args.min_cluster} peaks)", flush=True)

    # maximal cliques (fully co-accessible peak sets)
    cliques = [c for c in nx.find_cliques(G) if len(c) >= args.min_clique]
    cliques.sort(key=len, reverse=True)
    with (args.out_dir / "cliques.tsv").open("w") as fh:
        fh.write("clique_id\tsize\tn_genes\tstates\tspan\tmembers\n")
        for i, c in enumerate(cliques):
            a = annotate(c, types, genes, states)
            fh.write(f"{i}\t{len(c)}\t{a['n_genes']}\t{a['states']}\t{a['span']}\t{','.join(sorted(c))}\n")
    top = cliques[0] if cliques else []
    print(f"[cliques] {len(cliques)} cliques (>= {args.min_clique}); largest={len(top)}", flush=True)
    if not cliques:
        print("[cliques] none >= 3 -- if the input is proximal-distal ONLY (.pdc), the graph is "
              "bipartite (no triangles). Include the full .sel (pp+dd+pd) for cliques.", flush=True)

    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    with (args.out_dir / "summary.txt").open("w") as fh:
        fh.write(f"peaks\t{G.number_of_nodes()}\nedges\t{G.number_of_edges()}\n")
        fh.write(f"communities\t{len(clusters)}\n")
        fh.write(f"connected_components\t{len(comps)} (largest={len(comps[0]) if comps else 0})\n")
        fh.write(f"cliques(>= {args.min_clique})\t{len(cliques)} (largest={len(top)})\n")
        fh.write(f"filters\tq<{args.qval} coaccess>={args.coaccess_min} support>={args.min_support}\n")
    print(f"[done] outputs under {args.out_dir}")

    if args.plot:
        _plot(G, part, cliques, args.out_dir, args.max_plot_nodes)
    return 0


def _plot(G, part, cliques, out_dir: Path, max_nodes: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] skipped: {e}"); return
    if cliques:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sizes = [len(c) for c in cliques]
        ax.hist(sizes, bins=range(min(sizes), max(sizes) + 2))
        ax.set_xlabel("clique size (co-binding peaks)"); ax.set_ylabel("count")
        fig.tight_layout(); fig.savefig(out_dir / "clique_sizes.png", dpi=150); plt.close(fig)
    H = G
    if H.number_of_nodes() > max_nodes:
        keep = [n for n, _ in sorted(H.degree(), key=lambda kv: -kv[1])[:max_nodes]]
        H = H.subgraph(keep).copy()
        print(f"[plot] network capped to top {max_nodes} peaks by degree")
    if H.number_of_nodes() == 0:
        return
    pos = nx.spring_layout(H, seed=2026, weight="weight")
    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw_networkx_edges(H, pos, alpha=0.1, ax=ax)
    nx.draw_networkx_nodes(H, pos, node_size=15,
                           node_color=[part.get(n, -1) for n in H.nodes()], cmap="tab20", ax=ax)
    ax.set_title("RBBP4 peak co-accessibility (nodes colored by higher-order cluster)")
    ax.axis("off"); fig.tight_layout()
    fig.savefig(out_dir / "tf_peak_network.png", dpi=150); plt.close(fig)
    print(f"[plot] wrote {out_dir/'tf_peak_network.png'}, {out_dir/'clique_sizes.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
