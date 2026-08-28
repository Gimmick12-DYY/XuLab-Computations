#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# build_peak_graph.py
#
# Build a per-TF peak co-accessibility graph from Cicero connection files, for
# higher-order co-binding clustering. Nodes are PEAKS (RBBP4 binding regions);
# edges are co-accessible pairs. Merging the full network (proximal-proximal +
# distal-distal + proximal-distal, i.e. TF.<TF>.fitConns.res.sel) makes the graph
# non-bipartite so true a-b-c cliques can form. (The .pdc alone is proximal-distal
# only = bipartite = no triangles.)
#
# Parsing is layout-agnostic (works on .sel and .pdc): per row it finds the two
# genomic-coordinate tokens (chr:start-end or chr_start_end -> normalized) as the
# two peaks, and reads coaccess + q from the trailing numeric columns. Peak type
# (proximal/distal) and annotation (gene/state = the token after the type) are
# captured best-effort.
#
# Outputs (under --out-dir):
#   <tf>.peak_edges.tsv   peak1  peak2  coaccess  qval  type1  type2  n_links
#   <tf>.peak_annot.tsv   peak   types   genes   states   n_edges
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import glob
import re
from collections import defaultdict
from pathlib import Path

_COORD = re.compile(r"^(chr[0-9A-Za-z]+)[:_](\d+)[-_](\d+)$")
_TYPES = {"proximal", "distal"}


def norm_coord(tok: str) -> str | None:
    m = _COORD.match(tok)
    return f"{m.group(1)}:{m.group(2)}-{m.group(3)}" if m else None


def is_float(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def parse_row(toks: list[str]):
    """-> (peak1, peak2, coaccess, q, type1, type2, annot1, annot2) or None."""
    # peaks: the two distinct normalized coordinates (id and coord forms collapse).
    peaks: list[str] = []
    coord_idx: list[int] = []
    for i, t in enumerate(toks):
        c = norm_coord(t)
        if c is not None and c not in peaks:
            peaks.append(c)
            coord_idx.append(i)
    if len(peaks) != 2:
        return None
    # coaccess / q from trailing floats (…, coaccess, p, q) or (…, coaccess).
    tail = []
    for t in reversed(toks):
        if is_float(t):
            tail.append(float(t))
        else:
            break
    tail.reverse()
    if len(tail) >= 3:
        coaccess, pval, q = tail[-3], tail[-2], tail[-1]
    elif tail:
        coaccess, pval, q = tail[-1], None, None
    else:
        return None
    # types + annotation (token right after each 'proximal'/'distal').
    types: list[str] = []
    annots: list[str] = []
    for i, t in enumerate(toks):
        if t in _TYPES:
            types.append(t)
            a = toks[i + 1].strip() if i + 1 < len(toks) else ""
            annots.append("" if (not a or a.lower() == "nan" or norm_coord(a) or is_float(a)) else a)
    t1 = types[0] if len(types) >= 1 else ""
    t2 = types[1] if len(types) >= 2 else ""
    a1 = annots[0] if len(annots) >= 1 else ""
    a2 = annots[1] if len(annots) >= 2 else ""
    return peaks[0], peaks[1], coaccess, pval, q, t1, t2, a1, a2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--tf", required=True, help="TF id, e.g. RBBP4")
    ap.add_argument("--glob", default="TF.{tf}.fitConns.res.*",
                    help="input files for this TF (default: .sel + .pdc). '{tf}' is substituted.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--q-keep", type=float, default=1.0,
                    help="retain links with q < this at build time (default 1.0 = keep all)")
    args = ap.parse_args()

    pattern = str(args.data_dir / args.glob.replace("{tf}", args.tf))
    files = sorted(Path(p) for p in glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files matching {pattern}")
    print(f"[input] {len(files)} file(s): " + ", ".join(f.name for f in files), flush=True)

    # edge (unordered peak pair) -> [max_coaccess, min_q, n_links, type1, type2]
    edges: dict[frozenset, list] = {}
    node_types: dict[str, set] = defaultdict(set)
    node_genes: dict[str, set] = defaultdict(set)
    node_states: dict[str, set] = defaultdict(set)
    node_edges: dict[str, int] = defaultdict(int)
    n_rows = n_kept = 0
    for path in files:
        with path.open() as fh:
            for line in fh:
                n_rows += 1
                r = parse_row(line.rstrip("\n").split("\t"))
                if r is None:
                    continue
                p1, p2, ca, pv, q, t1, t2, a1, a2 = r
                if p1 == p2:
                    continue
                if q is not None and (q != q or q >= args.q_keep):
                    continue
                n_kept += 1
                # node annotations by type
                for pk, ty, an in ((p1, t1, a1), (p2, t2, a2)):
                    if ty:
                        node_types[pk].add(ty)
                    if an:
                        (node_genes if ty == "proximal" else node_states)[pk].add(an)
                    node_edges[pk] += 1
                key = frozenset((p1, p2))
                pval = pv if pv is not None else 1.0
                qv = q if q is not None else 1.0
                e = edges.get(key)
                if e is None:
                    edges[key] = [ca, pval, qv, 1, t1, t2]
                else:
                    if ca > e[0]:
                        e[0] = ca
                    if pval < e[1]:
                        e[1] = pval
                    if qv < e[2]:
                        e[2] = qv
                    e[3] += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    edges_path = args.out_dir / f"{args.tf}.peak_edges.tsv"
    with edges_path.open("w") as out:
        out.write("peak1\tpeak2\tcoaccess\tpval\tqval\ttype1\ttype2\tn_links\n")
        for key, (ca, pv, qv, n, t1, t2) in edges.items():
            p1, p2 = sorted(key)
            out.write(f"{p1}\t{p2}\t{ca:.6g}\t{pv:.6g}\t{qv:.6g}\t{t1}\t{t2}\t{n}\n")
    annot_path = args.out_dir / f"{args.tf}.peak_annot.tsv"
    with annot_path.open("w") as out:
        out.write("peak\ttypes\tgenes\tstates\tn_edges\n")
        for pk in sorted(node_edges):
            out.write(f"{pk}\t{','.join(sorted(node_types[pk])) or '.'}\t"
                      f"{','.join(sorted(node_genes[pk])) or '.'}\t"
                      f"{','.join(sorted(node_states[pk])) or '.'}\t{node_edges[pk]}\n")

    n_prox = sum(1 for pk in node_types if "proximal" in node_types[pk])
    n_dist = sum(1 for pk in node_types if "distal" in node_types[pk])
    print(f"[done] rows={n_rows:,} kept={n_kept:,} peaks={len(node_edges):,} "
          f"(proximal~{n_prox:,} distal~{n_dist:,}) edges={len(edges):,}")
    print(f"[done] wrote {edges_path}\n[done] wrote {annot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
