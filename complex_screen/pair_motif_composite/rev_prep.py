# Step 4 (reversed design) prep: eligible union-ATAC bin centres (+/- 100 bp) to FASTA, a 0-order background, and the known DB split into
# chunks of 100 motifs for a FIMO array. Output in reversed/.
import os, subprocess
import pandas as pd
O = os.path.dirname(os.path.abspath(__file__)); R = f"{O}/reversed"; os.makedirs(f"{R}/chunks", exist_ok=True); os.makedirs(f"{R}/fimo", exist_ok=True); os.makedirs(f"{R}/logs", exist_ok=True)
Q = pd.read_csv(f"{O}/ref/bins_qc.tsv", sep="\t"); num = Q.chrom.str.replace("chr", "")
ok = Q.n_ok & ~Q.blacklist & (Q.exonic <= 0.5) & num.isin([str(i) for i in range(1, 23)] + ["X"])
with open(f"{R}/bins.fa", "w") as f:
    for b, s in zip(Q.bin[ok], Q.seq[ok]): f.write(f">b{b}\n{s}\n")
subprocess.run(f"module load meme/5.5.7 >/dev/null 2>&1; fasta-get-markov -m 0 {R}/bins.fa {R}/bins.bg", shell=True, executable="/bin/bash", check=True)
txt = open(f"{O}/db/known.meme").read(); head, *bl = txt.split("\nMOTIF "); bl = ["MOTIF " + b for b in bl]
for k in range(0, len(bl), 100): open(f"{R}/chunks/c{k // 100:02d}.meme", "w").write(head + "\n" + "\n".join(bl[k:k + 100]) + "\n")
print(int(ok.sum()), "bins;", len(bl), "motifs;", (len(bl) + 99) // 100, "chunks")
