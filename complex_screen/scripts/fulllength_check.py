# Full-length construct check of the strong non-STAG2 screen hits (screen chains are disorder-trimmed with GS linkers). Writes Boltz YAMLs with full-length UniProt sequences and unpaired MSAs (whole a3m).
import json, os, re
C = "/work/users/d/y/dyy12/XuLab/complex_screen"; O = f"{C}/fulllength_check"
os.makedirs(f"{O}/inputs", exist_ok=True); os.makedirs(f"{O}/msa", exist_ok=True)
def seq(g): return json.load(open(f"{C}/seqs/uniprot/{g}.json"))["sequence"]["value"]
def csv(g):
    dst = f"{O}/msa/{g}_full.csv"
    if not os.path.exists(dst):
        rows, cur = [], None
        for l in open(f"{C}/msa/full/{g}.a3m"):
            l = l.rstrip("\n").replace("\x00", "")
            if not l: continue
            if l.startswith(">"): rows.append("")
            else: rows[-1] += re.sub(r"[a-z.]", "", l)
        out, seen = [], set()
        for r in rows:
            if set(r) <= {"-"} or r in seen: continue
            seen.add(r); out.append(r)
        with open(dst, "w") as f: f.write("key,sequence\n" + "\n".join(f"-1,{x}" for x in out) + "\n")
    return dst
PAIRS = [("MLX", "RBBP7"), ("ZNF282", "ZNF746"), ("HOXB8", "RBPJ"), ("HOXA5", "RBBP4"), ("MLX", "SREBF2"), ("RBBP4", "RBBP7"),
         ("MTF2", "RBBP4"), ("HOXC4", "RBBP4"), ("EPAS1", "NEUROD4"), ("NEUROD4", "SREBF2"), ("RBBP7", "ZNF79"), ("SREBF2", "ZNF367"),
         ("NEUROD4", "ZNF746"), ("SREBF2", "ZNF746")]  # the 14 strong non-STAG2 screen hits (2026-10-30)
os.makedirs(f"{O}/inputs_seeds23", exist_ok=True)
for a, b in PAIRS:
    with open(f"{O}/inputs/{a}__{b}_full.yaml", "w") as f:
        f.write("version: 1\nsequences:\n")
        for i, g in zip("AB", (a, b)):
            f.write(f"  - protein:\n      id: {i}\n      sequence: {seq(g)}\n      msa: {csv(g)}\n")
    if (a, b) in [("MTF2", "RBBP4"), ("RBBP4", "RBBP7")]:
        os.system(f"cp {O}/inputs/{a}__{b}_full.yaml {O}/inputs_seeds23/")
    print(a, b, len(seq(a)), len(seq(b)), len(seq(a)) + len(seq(b)))
