#!/usr/bin/env python3
"""Exon vs intron read density, and read ends at exon boundaries, per gene (all TFs pooled and per TF).
Reads: single-end; + strand rows give the full read span, - strand rows are 1 bp (5' end only)."""
import gzip, sys
from pathlib import Path
import numpy as np, pandas as pd
G = pd.read_csv(sys.argv[1], sep="\t"); FR = Path(sys.argv[2]); OUT = Path(sys.argv[3])
G["ex"] = [list(zip(map(int, s.strip(",").split(",")), map(int, e.strip(",").split(",")))) for s, e in zip(G.exonStarts, G.exonEnds)]
win = {c: (d.txStart.min() - 1000, d.txEnd.max() + 1000) for c, d in G.groupby("chrom")}
rows, bnd = [], []
for f in sorted(FR.glob("TF.*.bed.gz")):
    tf = f.name[3:-7]
    reads = {c: [] for c in win}
    with gzip.open(f, "rt") as fh:
        for ln in fh:
            x = ln.split("\t", 6)
            if x[0] in win and win[x[0]][0] <= int(x[1]) <= win[x[0]][1]:
                reads[x[0]].append((int(x[1]), int(x[2]), x[5][0]))
    for g in G.itertuples():
        R = [r for r in reads[g.chrom] if g.txStart <= r[0] < g.txEnd]
        exlen = sum(e - s for s, e in g.ex); inlen = (g.txEnd - g.txStart) - exlen
        p5 = [r[0] if r[2] == "+" else r[1] - 1 for r in R]
        nex = sum(any(s <= p < e for s, e in g.ex) for p in p5)
        # + strand reads with full span: does the read stop exactly at an internal exon boundary?
        inner = [b for s, e in g.ex for b in (s, e)][1:-1]
        plus = [r for r in R if r[2] == "+" and r[1] - r[0] > 1]
        at_b = sum(any(abs(r[0] - b) <= 1 or abs(r[1] - b) <= 1 for b in inner) for r in plus)
        rows.append({"tf": tf, "gene": g.name, "reads": len(R), "exon_reads": nex, "intron_reads": len(R) - nex,
                     "exon_bp": exlen, "intron_bp": inlen, "plus_reads": len(plus), "plus_end_at_exon_boundary": at_b,
                     "n_internal_boundaries": len(inner), "median_plus_len": float(np.median([r[1] - r[0] for r in plus])) if plus else np.nan})
    print(tf, flush=True)
D = pd.DataFrame(rows); D.to_csv(OUT / "exon_check_per_tf.tsv", sep="\t", index=False)
S = D.groupby("gene").sum(numeric_only=True)
S["exon_bp"] = D.groupby("gene").exon_bp.first(); S["intron_bp"] = D.groupby("gene").intron_bp.first()
S["n_internal_boundaries"] = D.groupby("gene").n_internal_boundaries.first()
S["exon_per_kb"] = S.exon_reads / S.exon_bp * 1e3; S["intron_per_kb"] = S.intron_reads / S.intron_bp.clip(lower=1) * 1e3
S["exon_over_intron"] = S.exon_per_kb / S.intron_per_kb.replace(0, np.nan)
S["pct_plus_reads_ending_at_boundary"] = 100 * S.plus_end_at_exon_boundary / S.plus_reads
# expected share if read ends were uniform: each read has 2 ends, each within ±1 bp (3 bp) of a boundary
S["pct_expected_uniform"] = 100 * 2 * 3 * S.n_internal_boundaries / (S.exon_bp + S.intron_bp)
S.to_csv(OUT / "exon_check_by_gene.tsv", sep="\t")
print(S[["reads", "exon_per_kb", "intron_per_kb", "exon_over_intron", "pct_plus_reads_ending_at_boundary", "pct_expected_uniform"]].round(2).to_string())
