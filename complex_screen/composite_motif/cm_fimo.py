# Pair-test motif scans on union-ATAC elements. FIMO p <= 1e-4 (stricter tiers filtered later) for half-sites, 3 column-shuffled copies of each, and the CAP-SELEX MLX_FOXA1 / MLX_FOXD2
# composites (FOXK1-MLX has no composite of its own in Supp Table S3/S4, so these paralog composites set the spacing prior). THAP1 stands in for THAP6, which has no motif in any database here.
# Also scans a dinucleotide-shuffled copy of the sequences with the real half-sites, for the spacing control.
import re, subprocess
import numpy as np
from cm_common import *
sh = lambda c: subprocess.run(c, shell=True, check=True, executable="/bin/bash"); D = f"{O}/pairs"; subprocess.run(f"mkdir -p {D}", shell=True)
txt = open(f"{X}/motif_analysis/cache/motifdb/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt").read(); hdr = txt.split("MOTIF ")[0]; blocks = {b.split()[0]: b for b in txt.split("MOTIF ")[1:]}
IDS = {"MA0663.1": "MLX", "MA0596.1": "SREBF2", "MA0828.3": "SREBF2", "MA0852.3": "FOXK1", "MA0597.3": "THAP1_proxy"}; r = np.random.default_rng(20261102); real = hdr; shuf = ""; ic = {}
for m in IDS:
    b = blocks[m]; ls = b.strip().split("\n"); i = [k for k, l in enumerate(ls) if l.startswith("letter-probability")][0]; w = int(re.search(r"w= *(\d+)", ls[i]).group(1)); M = ls[i + 1:i + 1 + w]
    P = np.array([[float(x) for x in l.split()] for l in M]); ic[m] = float((2 + (P * np.log2(P + 1e-9)).sum(1)).sum()); real += "MOTIF " + b.rstrip() + "\n\n"
    for s in range(3): shuf += f"MOTIF {m}.shuf{s + 1} {IDS[m]}_columns_shuffled\n{ls[i]}\n" + "\n".join(M[j] for j in r.permutation(w)) + "\n\n"
open(f"{D}/real.meme", "w").write(real); open(f"{D}/all.meme", "w").write(real + shuf + open(f"{O}/capselex_mlx_fox.meme.part").read())
mb = "/nas/longleaf/rhel9/apps/meme/5.5.7"; FF = f"{mb}/bin/fimo --thresh 1e-4 --max-stored-scores 20000000 --text --bfile {U}/union.bg"
sh(f"{FF} {D}/all.meme {U}/union.fa > {D}/fimo_all.tsv 2>/dev/null")
sh(f"{mb}/libexec/meme-5.5.7/fasta-dinucleotide-shuffle -f {U}/union.fa -t _s -s 20261102 > {D}/union_dinuc.fa")
sh(f"{FF} {D}/real.meme {D}/union_dinuc.fa > {D}/fimo_dinuc.tsv 2>/dev/null")
print("information content (bits):", {k: round(v, 1) for k, v in ic.items()})
