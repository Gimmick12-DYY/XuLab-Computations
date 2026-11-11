# Motif validation (Fig 4c), prep. Sites = lab MACS2 summits (no control; the 2026-11-11 re-call QC kept them), top 1,000 per TF inside
# eligible union-ATAC bins, HOT bins removed, Alu / Codebook non-endogenous artifact sites removed (recall/pools_noart/raw_lab), with the
# matched background (GC x CpG x promoter x ATAC quintile, 2 per site). Writes (1) one FASTA of all unique +/- 100 bp sequences (fg + bg)
# and the motif DB split into chunks, for a FIMO presence matrix; (2) per TF and part, +/- 250 bp summit FASTA for CentriMo.
# Output motif_valid/{all.fa, sites.tsv, chunks/, cm_fa/<TF>_<part>.fa, tasks_cm.tsv}
import os, glob, subprocess
import pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; R = f"{O}/recall/pools_noart/raw_lab"; G = f"{O}/motif_valid"
for d in ["chunks", "cm_fa", "fimo", "cm", "logs", "tmp"]: os.makedirs(f"{G}/{d}", exist_ok=True)
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True, executable="/bin/bash"); assert r.returncode == 0, (c, r.stderr[-1500:]); return r.stdout
rows = []
for f in sorted(glob.glob(f"{R}/*/*/*.bed")):
    t, pt, nm = f.split("/")[-3:]
    for l in open(f): x = l.split("\t"); rows.append((t, pt, nm[:-4], x[0], int(x[1]), int(x[2])))
S = pd.DataFrame(rows, columns=["tf", "part", "set", "c", "s", "e"]); S["key"] = S.c + ":" + S.s.astype(str) + "-" + S.e.astype(str); S.to_csv(f"{G}/sites.tsv", sep="\t", index=False)
U = S.drop_duplicates("key"); U[["c", "s", "e", "key"]].to_csv(f"{G}/tmp/u.bed", sep="\t", header=False, index=False)
sh(f"module load bedtools >/dev/null 2>&1; bedtools getfasta -nameOnly -fi {X}/downstream/cache/hg38.fa -bed {G}/tmp/u.bed -fo {G}/all.fa")
fg = S[S.set == "fg"]; tasks = []
for (t, pt), g in fg.groupby(["tf", "part"]):
    m = (g.s + 100).astype(int); b = pd.DataFrame(dict(c=g.c, s=m - 250, e=m + 250, k=g.key)); b = b[b.s > 0]; fb = f"{G}/tmp/{t}_{pt}.bed"; b.to_csv(fb, sep="\t", header=False, index=False)
    sh(f"module load bedtools >/dev/null 2>&1; bedtools getfasta -nameOnly -fi {X}/downstream/cache/hg38.fa -bed {fb} -fo {G}/cm_fa/{t}_{pt}.fa"); tasks.append(f"{t}\t{pt}")
open(f"{G}/tasks_cm.tsv", "w").write("\n".join(tasks) + "\n")
txt = open(f"{O}/db/known.meme").read(); head, *bl = txt.split("\nMOTIF "); bl = ["MOTIF " + b.rstrip() for b in bl]
for k in range(0, len(bl), 100): open(f"{G}/chunks/c{k // 100:02d}.meme", "w").write(head.rstrip() + "\n\n" + "\n\n".join(bl[k:k + 100]) + "\n")
print(len(S), "site rows;", len(U), "unique sequences;", len(tasks), "CentriMo tasks;", (len(bl) + 99) // 100, "motif chunks")
