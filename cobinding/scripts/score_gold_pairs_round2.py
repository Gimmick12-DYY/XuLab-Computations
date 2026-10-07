#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# score_gold_pairs_round2.py
#
# Round-2 Cicero reproducibility for ALREADY-CALLED pairs (e.g. gold RBBP4).
# Look up each gold pair in a second-round coaccess table, score with the same
# shuffle-Gaussian null used for calling, and BH-correct ONLY among the gold
# pairs (not among every peak-peak link in the restricted run).
#
# Missing from round-2 conns -> coaccess_r2 = NA, p=1, q=1 (not reproduced).
#
#   python score_gold_pairs_round2.py \
#     --edges cobinding/work/RBBP4.peak_edges.tsv --qval 0.05 \
#     --conns work/RBBP4/cicero_stage2/cicero/coaccess.tsv.gz \
#     --shuffle-para work/RBBP4/cicero_stage2/cicero_shuf/shuffle.para.txt \
#     --out results/RBBP4/stage2/RBBP4.gold_scored_round2.tsv
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from cicero_conns_to_edges import (  # noqa: E402
    bh_qvalues,
    gaussian_upper_p,
    parse_shuffle_para,
    to_colon,
)


def load_conns(path: Path) -> dict[frozenset, float]:
    opener = gzip.open if str(path).endswith(".gz") else open
    out: dict[frozenset, float] = {}
    with opener(path, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        # Peak1 Peak2 coaccess
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            a, b = to_colon(p[0]), to_colon(p[1])
            if a is None or b is None or a == b:
                continue
            try:
                ca = float(p[2])
            except ValueError:
                continue
            if ca != ca:
                continue
            key = frozenset((a, b))
            prev = out.get(key)
            if prev is None or ca > prev:
                out[key] = ca
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edges", type=Path, required=True, help="gold / round-1 peak_edges.tsv")
    ap.add_argument("--conns", type=Path, required=True, help="round-2 cicero coaccess.tsv.gz")
    ap.add_argument("--shuffle-para", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--qval", type=float, default=0.05,
                    help="keep gold pairs with round-1 qval <= this")
    ap.add_argument("--fdr", type=float, default=0.05,
                    help="round-2 BH FDR cutoff among gold pairs")
    args = ap.parse_args()

    mu, sd = parse_shuffle_para(args.shuffle_para)
    conns = load_conns(args.conns)
    print(f"[r2] conns unique pairs={len(conns):,}  shuffle mu={mu:.6g} sd={sd:.6g}",
          flush=True)

    rows = []
    with args.edges.open() as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        col = {h: i for i, h in enumerate(hdr)}
        i1, i2 = col["peak1"], col["peak2"]
        iq = col.get("qval", col.get("fdr"))
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            if iq is not None and float(p[iq]) > args.qval:
                continue
            a, b = p[i1], p[i2]
            key = frozenset((a, b))
            ca2 = conns.get(key)
            rows.append((p, key, ca2))

    n = len(rows)
    n_hit = sum(1 for _, _, ca in rows if ca is not None)
    ca = np.array([0.0 if ca is None else ca for _, _, ca in rows], dtype=float)
    present = np.array([ca is not None for _, _, ca in rows], dtype=bool)
    # missing from round-2 conns: treat as not reproduced (p=1)
    pv = np.ones(n, dtype=float)
    pv[present] = gaussian_upper_p(ca[present], mu, sd)
    qv = bh_qvalues(pv)
    keep = qv <= args.fdr

    print(f"[r2] gold pairs (r1 q<={args.qval}): {n:,}", flush=True)
    print(f"[r2] present in round-2 conns: {n_hit:,} ({100 * n_hit / max(n, 1):.1f}%)",
          flush=True)
    print(f"[r2] significant at BH FDR<={args.fdr} (among gold only): "
          f"{int(keep.sum()):,} ({100 * keep.mean():.1f}%)", flush=True)
    if present.any():
        print(f"[r2] coaccess_r2 median (present)={np.median(ca[present]):.4g}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_all = args.out
    out_sig = args.out.with_name(args.out.stem + f".fdr{args.fdr:g}.tsv")
    with out_all.open("w") as fo, out_sig.open("w") as fs:
        extra = "\tcoaccess_r2\tpval_r2\tqval_r2\tin_r2_conns\treproduced"
        fo.write("\t".join(hdr) + extra + "\n")
        fs.write("\t".join(hdr) + extra + "\n")
        for i, (p, key, ca2) in enumerate(rows):
            in_c = "1" if present[i] else "0"
            ca_s = f"{ca2:.6g}" if ca2 is not None else "NA"
            line = ("\t".join(p)
                    + f"\t{ca_s}\t{pv[i]:.6g}\t{qv[i]:.6g}\t{in_c}\t"
                    + ("1" if keep[i] else "0") + "\n")
            fo.write(line)
            if keep[i]:
                fs.write(line)
    print(f"[r2] wrote all scored gold pairs -> {out_all}", flush=True)
    print(f"[r2] wrote reproduced (FDR<={args.fdr}) -> {out_sig}  n={int(keep.sum()):,}",
          flush=True)

    summary = args.out.with_name("round2_on_gold_summary.txt")
    summary.write_text(
        f"gold_pairs_r1_qle_{args.qval}\t{n}\n"
        f"present_in_r2_conns\t{n_hit}\n"
        f"present_pct\t{100 * n_hit / max(n, 1):.4f}\n"
        f"reproduced_bh_fdr_le_{args.fdr}\t{int(keep.sum())}\n"
        f"reproduced_pct_of_gold\t{100 * keep.mean():.4f}\n"
        f"shuffle_mu\t{mu}\n"
        f"shuffle_sd\t{sd}\n"
        f"note\tBH FDR is among the {n} gold pairs only (not genome-wide pair calls)\n"
    )
    print(f"[r2] wrote {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
