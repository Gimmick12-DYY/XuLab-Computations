#!/bin/bash
# Motif-centred read profiles (2026-11-12), prep: FASTA of eligible, non-HOT union-ATAC 1 kb bins (names chr_start_end; FIMO cuts names at ':').
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite; G=$O/motif_centred; mkdir -p $G/{fimo,counts,logs}
python3 - <<'PY'
import numpy as np, pandas as pd
O = "/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite"; G = f"{O}/motif_centred"
B = pd.read_csv(f"{O}/ref/bins1kb.bed", sep="\t", header=None, names=["c", "s", "e", "i"]); F = np.load(f"{O}/recall/ref.npz"); ok = F["elig"] & ~F["hot"]
b = B[ok[B.i.values]]; b.assign(n=b.c + "_" + b.s.astype(str) + "_" + b.e.astype(str))[["c", "s", "e", "n"]].to_csv(f"{G}/bins.bed", sep="\t", header=False, index=False); print(len(b), "bins")
PY
module load bedtools >/dev/null 2>&1
bedtools getfasta -nameOnly -fi /work/users/d/y/dyy12/XuLab/downstream/cache/hg38.fa -bed $G/bins.bed -fo $G/bins.fa
grep -c ">" $G/bins.fa
