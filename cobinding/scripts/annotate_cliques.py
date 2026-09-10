#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# annotate_cliques.py
#
# Independent annotation of co-binding CLIQUES (not the whole graph). Cicero
# proximal/distal/gene tokens are ignored. Each clique region is labelled with:
#   ChromHMM-18 (max bp overlap) + broad class, A/B compartment, nearest TSS.
# The CENTRAL GENE is the nearest protein-coding (NM_) TSS to the hub peak
# (highest degree in the clique; promoter-state breaks ties).
#
#   python cobinding/scripts/annotate_cliques.py \
#     --cliques cobinding/results/RBBP4/cliques.tsv \
#     --nodes   cobinding/results/RBBP4/nodes.tsv \
#     --out-dir cobinding/results/RBBP4
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from distal_chromhmm_composition import (  # noqa: E402
    BROAD, STATE2BROAD, _COORD, assign, load_segments,
)

_DEFAULT_CHROMHMM = Path(__file__).resolve().parents[2] / "data" / "HEK293T_chromHMM18.bed.gz"
_DEFAULT_AB = Path(__file__).resolve().parents[2] / "hic" / "work" / "compartments" / "compartments_25000.AB.bed"
_DEFAULT_TSS = Path("/nas/longleaf/rhel9/apps/homer/5.1/data/genomes/hg38/hg38.tss")
_DEFAULT_MAP = Path("/nas/longleaf/rhel9/apps/homer/5.1/data/accession/human2gene.tsv")


def parse_peak(p: str):
    m = _COORD.match(p)
    if not m:
        return None
    c, s, e = m.group(1), int(m.group(2)), int(m.group(3))
    return c, s, e, (s + e) // 2


def load_ab(path: Path):
    by = {}
    if not path.is_file():
        return by
    with open(path) as fh:
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 4 or p[3] not in ("A", "B"):
                continue
            by.setdefault(p[0], []).append((int(p[1]), int(p[2]), p[3]))
    idx = {}
    for c, segs in by.items():
        segs.sort()
        idx[c] = (np.fromiter((x[0] for x in segs), int, len(segs)),
                  np.fromiter((x[1] for x in segs), int, len(segs)),
                  [x[2] for x in segs])
    return idx


def ab_of(chrom, mid, ab_idx) -> str:
    if chrom not in ab_idx:
        return "."
    starts, ends, labs = ab_idx[chrom]
    k = int(np.searchsorted(starts, mid, side="right")) - 1
    if k >= 0 and mid < ends[k]:
        return labs[k]
    return "."


def load_symbols(path: Path, wanted: set[str]) -> dict[str, str]:
    """RefSeq (NM_/NR_…) -> gene symbol from HOMER human2gene.tsv (last column)."""
    out = {}
    if not path.is_file() or not wanted:
        return out
    with open(path) as fh:
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            sym = p[-1].strip()
            if not sym or sym.startswith("LOC") and sym[3:].isdigit():
                # keep LOC* only if nothing else maps later
                pass
            for tok in p:
                if tok in wanted:
                    if tok not in out or (out[tok].startswith("LOC") and not sym.startswith("LOC")):
                        out[tok] = sym
            if len(out) >= len(wanted):
                break
    return out


def load_tss(path: Path, symbols: dict[str, str]):
    """Per-chrom arrays of TSS midpoints + gene symbol + NM/NR flag."""
    rec = defaultdict(list)
    ids = []
    with open(path) as fh:
        for ln in fh:
            p = ln.split()
            if len(p) < 4:
                continue
            rid, c, s, e = p[0], p[1], int(p[2]), int(p[3])
            ids.append(rid)
            rec[c].append(((s + e) // 2, rid, rid.startswith("NM_")))
    # symbols filled in a second pass by caller; we just store rids for now
    idx = {}
    for c, rows in rec.items():
        rows.sort()
        idx[c] = (np.fromiter((x[0] for x in rows), int, len(rows)),
                  [x[1] for x in rows],
                  np.fromiter((1 if x[2] else 0 for x in rows), np.int8, len(rows)))
    return idx, set(ids)


def nearest_tss(chrom, mid, tss_idx, symbols, coding_only=False, max_dist=None):
    if chrom not in tss_idx:
        return ".", None, None
    pos, rids, coding = tss_idx[chrom]
    k = int(np.searchsorted(pos, mid))
    cand = []
    for j in (k - 1, k):
        if 0 <= j < len(pos):
            if coding_only and not coding[j]:
                continue
            d = abs(int(pos[j]) - mid)
            if max_dist is not None and d > max_dist:
                continue
            cand.append((d, j))
    # also walk outward for coding_only if the two neighbors were NR_
    if coding_only and not cand:
        for j in range(max(0, k - 20), min(len(pos), k + 21)):
            if coding[j]:
                d = abs(int(pos[j]) - mid)
                if max_dist is None or d <= max_dist:
                    cand.append((d, j))
    if not cand:
        return ".", None, None
    d, j = min(cand)
    rid = rids[j]
    gene = symbols.get(rid, rid)
    return gene, d, "NM" if coding[j] else "NR"


def promoter_genes(chrom, s, e, tss_idx, symbols, window: int):
    if chrom not in tss_idx:
        return []
    pos, rids, _ = tss_idx[chrom]
    lo, hi = s - window, e + window
    i0 = int(np.searchsorted(pos, lo, side="left"))
    i1 = int(np.searchsorted(pos, hi, side="right"))
    out = []
    seen = set()
    for j in range(i0, i1):
        g = symbols.get(rids[j], rids[j])
        if g not in seen and not g.startswith("ENSG"):
            seen.add(g)
            out.append(g)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cliques", type=Path, required=True)
    ap.add_argument("--nodes", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--chromhmm", type=Path, default=_DEFAULT_CHROMHMM)
    ap.add_argument("--ab-bed", type=Path, default=_DEFAULT_AB)
    ap.add_argument("--tss", type=Path, default=_DEFAULT_TSS)
    ap.add_argument("--gene-map", type=Path, default=_DEFAULT_MAP)
    ap.add_argument("--promoter-bp", type=int, default=2000)
    ap.add_argument("--max-gene-dist", type=int, default=100_000)
    args = ap.parse_args()

    degree, kcore = {}, {}
    with open(args.nodes) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            degree[r["peak"]] = int(r["degree"])
            kcore[r["peak"]] = int(r.get("kcore") or 0)

    cliques = []
    with open(args.cliques) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            regs = [x for x in r["regions"].split(";") if x]
            cliques.append(r | {"_regs": regs})
    print(f"[in] {len(cliques)} cliques, {sum(len(c['_regs']) for c in cliques)} region-slots",
          flush=True)

    seg_idx, states = load_segments(args.chromhmm)
    ab_idx = load_ab(args.ab_bed)
    print(f"[chromhmm] {len(states)} states  [AB] {sum(len(v[0]) for v in ab_idx.values()):,} bins",
          flush=True)

    tss_idx, wanted = load_tss(args.tss, {})
    print(f"[tss] {sum(len(v[0]) for v in tss_idx.values()):,} TSS; mapping symbols…", flush=True)
    symbols = load_symbols(args.gene_map, wanted)
    print(f"[map] {len(symbols):,}/{len(wanted):,} RefSeq IDs -> symbols", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_out = []
    gene_hits = Counter()
    gene_cliques = defaultdict(list)

    for c in cliques:
        regs = c["_regs"]
        ann = []
        for p in regs:
            loc = parse_peak(p)
            if loc is None:
                continue
            chrom, s, e, mid = loc
            si = assign(p, seg_idx, len(states))
            st = "Unassigned" if si is None else states[si]
            br = "Unassigned" if si is None else STATE2BROAD.get(st, "Unassigned")
            gene, dist, kind = nearest_tss(chrom, mid, tss_idx, symbols,
                                           coding_only=True, max_dist=args.max_gene_dist)
            if gene == ".":
                gene, dist, kind = nearest_tss(chrom, mid, tss_idx, symbols,
                                               coding_only=False, max_dist=args.max_gene_dist)
            prom = promoter_genes(chrom, s, e, tss_idx, symbols, args.promoter_bp)
            ann.append({
                "peak": p, "chrom": chrom, "mid": mid,
                "degree": degree.get(p, 0), "kcore": kcore.get(p, 0),
                "state": st, "broad": br, "ab": ab_of(chrom, mid, ab_idx),
                "nearest_gene": gene, "dist": dist, "kind": kind or ".",
                "promoter_genes": prom,
            })
        if not ann:
            continue
        # hub: highest degree, then promoter chromatin, then closest TSS
        def hub_key(a):
            return (-a["degree"],
                    0 if a["broad"] == "Promoter" else 1,
                    a["dist"] if a["dist"] is not None else 10**12)
        hub = min(ann, key=hub_key)
        central = hub["nearest_gene"]
        prom_all = []
        seen = set()
        for a in ann:
            for g in a["promoter_genes"]:
                if g not in seen:
                    seen.add(g)
                    prom_all.append(g)
        # if hub has no gene but a promoter gene exists, use that
        if (not central or central == ".") and prom_all:
            central = prom_all[0]
        br_counts = Counter(a["broad"] for a in ann)
        ab_counts = Counter(a["ab"] for a in ann)
        if central and central != ".":
            gene_hits[central] += 1
            gene_cliques[central].append(c["clique_id"])
        rows_out.append({
            "clique_id": c["clique_id"],
            "n_regions": len(ann),
            "chromosome": c.get("chromosome", ann[0]["chrom"]),
            "span_bp": c.get("span_bp", ""),
            "central_gene": central or ".",
            "central_gene_dist_bp": hub["dist"] if hub["dist"] is not None else ".",
            "central_transcript": hub["kind"],
            "hub_peak": hub["peak"],
            "hub_degree": hub["degree"],
            "hub_chromhmm": hub["state"],
            "hub_class": hub["broad"],
            "hub_compartment": hub["ab"],
            "promoter_genes": ";".join(prom_all) or ".",
            "region_classes": ";".join(f"{k}:{v}" for k, v in br_counts.most_common()),
            "compartments": ";".join(f"{k}:{v}" for k, v in ab_counts.most_common()),
            "region_genes": ";".join(
                f"{a['peak']}={a['nearest_gene']}" +
                (f"({a['dist']})" if a["dist"] is not None else "")
                for a in ann),
            "regions": ";".join(a["peak"] for a in ann),
        })

    out_c = args.out_dir / "cliques_annotated.tsv"
    cols = ["clique_id", "n_regions", "chromosome", "span_bp", "central_gene",
            "central_gene_dist_bp", "central_transcript", "hub_peak", "hub_degree",
            "hub_chromhmm", "hub_class", "hub_compartment", "promoter_genes",
            "region_classes", "compartments", "region_genes", "regions"]
    with out_c.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows_out:
            fh.write("\t".join(str(r[k]) for k in cols) + "\n")

    out_g = args.out_dir / "clique_central_genes.tsv"
    with out_g.open("w") as fh:
        fh.write("central_gene\tn_cliques\tclique_ids\n")
        for g, n in gene_hits.most_common():
            fh.write(f"{g}\t{n}\t{';'.join(gene_cliques[g])}\n")

    n_named = sum(1 for r in rows_out if r["central_gene"] not in (".", ""))
    print(f"[done] {len(rows_out)} cliques, {n_named} with a central gene, "
          f"{len(gene_hits)} unique genes -> {out_c.name} + {out_g.name}")
    print("top central genes: " +
          ", ".join(f"{g}×{n}" for g, n in gene_hits.most_common(12)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
