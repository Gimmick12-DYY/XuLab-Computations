#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Compare a sidecar RBBP4 Cicero rebuild against the gold-standard fitConns
# graph (cobinding/work/RBBP4.peak_edges.tsv + results/RBBP4/summary.txt).
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def unordered_pairs(df: pd.DataFrame) -> set[frozenset[str]]:
    return {frozenset((str(a), str(b))) for a, b in zip(df["peak1"], df["peak2"])}


def pair_map(df: pd.DataFrame) -> dict[frozenset[str], float]:
    out: dict[frozenset[str], float] = {}
    for a, b, ca in zip(df["peak1"], df["peak2"], df["coaccess"]):
        key = frozenset((str(a), str(b)))
        ca = float(ca)
        prev = out.get(key)
        if prev is None or ca > prev:
            out[key] = ca
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def read_summary(path: Path) -> dict[str, str]:
    out = {}
    if not path.is_file():
        return out
    for ln in path.read_text().splitlines():
        if "\t" in ln:
            k, v = ln.split("\t", 1)
            out[k.strip()] = v.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold-edges", type=Path, required=True)
    ap.add_argument("--repro-edges", type=Path, required=True)
    ap.add_argument("--gold-summary", type=Path, default=None)
    ap.add_argument("--repro-summary", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--qval", type=float, default=0.05)
    ap.add_argument("--coaccess-min", type=float, default=0.05)
    args = ap.parse_args()

    gold = pd.read_csv(args.gold_edges, sep="\t")
    repro = pd.read_csv(args.repro_edges, sep="\t")
    gold_all = pair_map(gold)
    repro_all = pair_map(repro)

    gold_q = pair_map(gold[gold["qval"] <= args.qval])
    gold_ca = pair_map(gold[gold["coaccess"] >= args.coaccess_min])
    repro_ca = pair_map(repro[repro["coaccess"] >= args.coaccess_min])

    g_all, r_all = set(gold_all), set(repro_all)
    g_q, r_ca = set(gold_q), set(repro_ca)
    shared_all = g_all & r_all
    shared_filt = g_q & r_ca

    lines = []
    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit(f"gold edges file:  {args.gold_edges}  n={len(gold_all):,}")
    emit(f"repro edges file: {args.repro_edges}  n={len(repro_all):,}")
    emit(f"gold unique peaks:  {len({p for k in g_all for p in k}):,}")
    emit(f"repro unique peaks: {len({p for k in r_all for p in k}):,}")
    emit("")
    emit("=== unordered pair overlap ===")
    emit(f"all gold vs all repro:           intersection={len(shared_all):,}  "
         f"jaccard={jaccard(g_all, r_all):.4f}")
    emit(f"gold q<={args.qval} vs repro coaccess>={args.coaccess_min}:  "
         f"intersection={len(shared_filt):,}  jaccard={jaccard(g_q, r_ca):.4f}")
    emit(f"gold n q<={args.qval}: {len(g_q):,}   gold n coaccess>={args.coaccess_min}: {len(gold_ca):,}")
    emit(f"repro n coaccess>={args.coaccess_min}: {len(r_ca):,}")
    emit(f"repro recovered of gold-FDR pairs: "
         f"{len(g_q & r_ca):,} / {len(g_q):,} = {len(g_q & r_ca) / len(g_q) if g_q else 0:.3f}")
    emit("")

    if shared_all:
        g = np.array([gold_all[k] for k in shared_all])
        r = np.array([repro_all[k] for k in shared_all])
        spe = float(pd.Series(g).corr(pd.Series(r), method="spearman"))
        pea = float(pd.Series(g).corr(pd.Series(r), method="pearson"))
        emit("=== coaccess on shared pairs (all gold ∩ all repro) ===")
        emit(f"n={len(shared_all):,}  spearman={spe:.4f}  pearson={pea:.4f}")
        emit(f"gold coaccess:  median={np.median(g):.4g} mean={np.mean(g):.4g}")
        emit(f"repro coaccess: median={np.median(r):.4g} mean={np.mean(r):.4g}")
        emit("")

    gs = read_summary(args.gold_summary) if args.gold_summary else {}
    rs = read_summary(args.repro_summary) if args.repro_summary else {}
    if gs or rs:
        emit("=== clustering summary (gold vs repro) ===")
        keys = ["regions", "edges", "cliques(>= 3)", "overlap_modules",
                "multi_clique_modules(n_source>1)", "largest_module"]
        for k in keys:
            emit(f"{k:40s}  gold={gs.get(k, '-'):20s}  repro={rs.get(k, '-')}")
        emit("")

    emit("gold target (FDR<=0.05): 15,732 regions, 11,342 edges, 355 cliques "
         "(344 K3 + 11 K4), 326 modules, 25 multi-clique, largest=5")

    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        emit(f"[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
