"""HOMER .motifs library -> MEME (motif id = HOMER name before the first '/', duplicates suffixed __k). usage: homer2meme.py known.motifs out.meme"""
import sys
src, out = sys.argv[1:3]; M = []; cur = None
for l in open(src):
    if l.startswith(">"):
        cur = [l[1:].split("\t")[1].split("/")[0], []]; M.append(cur)
    elif l.strip(): cur[1].append([float(x) for x in l.split()])
with open(out, "w") as f:
    f.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\nBackground letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")
    seen = {}
    for n, rows in M:
        k = seen.get(n, 0); seen[n] = k + 1; nid = (n if k == 0 else f"{n}__{k}").replace(" ", "_")
        f.write(f"MOTIF {nid}\nletter-probability matrix: alength= 4 w= {len(rows)}\n")
        for r in rows: s = sum(r); f.write(" ".join(f"{v / s:.4f}" for v in r) + "\n")
        f.write("\n")
print(len(M), "HOMER motifs")
