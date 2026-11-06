# Weak Boltz candidates at 1 seed, same pipeline; MSAs reused where they exist.
import json, os
C = "/work/users/d/y/dyy12/XuLab/complex_screen"; O = f"{C}/af3_novel_20261103"; L = [f"{O}/msa_out", f"{C}/af3_local/msa_out", f"{C}/af3_dna/msa_out"]
rows = [l.rstrip("\n").split("\t") for l in open(f"{O}/weak.tsv")][1:]
def path(g):
    for d in L:
        f = f"{d}/{g.lower()}/{g.lower()}_data.json"
        if os.path.exists(f): return f
def seq(g): return json.load(open(f"{C}/seqs/uniprot/{g}.json"))["sequence"]["value"]
PS = [(r[0], r[1]) for r in rows]; miss = sorted({g for p in PS for g in p if not path(g)})
for g in miss: json.dump({"name": g, "modelSeeds": [1], "dialect": "alphafold3", "version": 1, "sequences": [{"protein": {"id": "A", "sequence": seq(g)}}]}, open(f"{O}/msa_in/{g}.json", "w"))
open(f"{O}/proteins_weak.txt", "w").write("".join(g + "\n" for g in miss)); open(f"{O}/pairs_weak.txt", "w").write("".join(f"{a}_{b}\n" for a, b in PS)); print("weak pairs", len(PS), "MSA needed", miss)
