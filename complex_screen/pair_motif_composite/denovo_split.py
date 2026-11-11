# Split the de novo family representatives (per_tf_matched/denovo_cmp/reps.meme) into 100-motif chunks for FIMO over the union-ATAC bin centres.
import os, glob, subprocess
B = os.environ.get("TFBASE", "/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite/per_tf_matched"); G = f"{B}/denovo_cmp"; os.makedirs(f"{G}/chunks", exist_ok=True); os.makedirs(f"{G}/fimo", exist_ok=True)
head, *bl = open(f"{G}/reps.meme").read().split("\nMOTIF "); bl = ["MOTIF " + b.rstrip() for b in bl]
for k in range(0, len(bl), 100): open(f"{G}/chunks/c{k // 100:02d}.meme", "w").write(head.rstrip() + "\n\n" + "\n\n".join(bl[k:k + 100]) + "\n")
seen = set()
with open(f"{G}/val.bed", "w") as fo:  # held-out sequences; non-bin windows renamed <TF>:<id> so names are unique
    for f in sorted(glob.glob(f"{B}/pools/*/val/*.bed")):
        t = f.split("/")[-3]
        for l in open(f):
            x = l.split("\t"); n = x[3] if x[3].startswith("b") else f"{t}:{x[3]}"
            if n not in seen: seen.add(n); fo.write(f"{x[0]}\t{x[1]}\t{x[2]}\t{n}\n")
subprocess.run(f"module load bedtools >/dev/null 2>&1; bedtools getfasta -nameOnly -fi /work/users/d/y/dyy12/XuLab/downstream/cache/hg38.fa -bed {G}/val.bed -fo {G}/val.fa", shell=True, executable="/bin/bash", check=True)
print(len(bl), "families;", (len(bl) + 99) // 100, "chunks;", len(seen), "held-out sequences")
