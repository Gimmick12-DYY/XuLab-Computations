"""Pseudobulk TF read coverage for the chr19 super-enhancer browser (Figure 3g).

For each TF, all single-cell reads of that TF are pooled (per-TF read BED, hg38, single-end: + strand entries span the read,
- strand entries are the 1 bp 5' end). Each read is counted once at its 5' end (+: start, -: end - 1) in fixed bins of the
browser window chr19:430,000-2,250,000; the total number of reads genome-wide is written as bin "total" (CPM denominator).
Output columns (no header): tf, bin index (or "total"), reads; bins with zero reads are omitted.
  python se_tf_coverage.py <out.tsv> [bin_bp, default 1000]
Read BEDs: $TF_READS/TF.<TF>.bed.gz (default /vast/som/xujie_lab/TAGATG-293T/bigwig/fragment, read-only lab share).
"""
import gzip
import os
import sys
from collections import Counter
from pathlib import Path

READS = Path(os.environ.get("TF_READS", "/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment"))
TFS = ["SOX4", "NFYA", "MBD3", "E4F1", "ZNF507", "MLX", "NME2", "DR1", "GLYR1", "MTF2"]
CHROM, L0, L1 = "chr19", 430000, 2250000


def main(out, bin_bp):
    with open(out, "w") as fo:
        for tf in TFS:
            cnt, total = Counter(), 0
            with gzip.open(READS / f"TF.{tf}.bed.gz", "rt") as fh:
                for ln in fh:
                    total += 1
                    if not ln.startswith(CHROM + "\t"):
                        continue
                    c, s, e, *rest = ln.split("\t")
                    strand = next((x.strip() for x in rest if x.strip() in ("+", "-")), "+")
                    p = int(s) if strand == "+" else int(e) - 1
                    if L0 <= p < L1:
                        cnt[(p - L0) // bin_bp] += 1
            for b in sorted(cnt):
                fo.write(f"{tf}\t{b}\t{cnt[b]}\n")
            fo.write(f"{tf}\ttotal\t{total}\n")
            print(tf, total, sum(cnt.values()), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
