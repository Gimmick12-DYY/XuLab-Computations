# Full-length Boltz-2 retest of the 6 complex edges backed only by trimmed Boltz hits, same settings as fulllength_check (seeds 1-3 x 5 samples); HP1 = CBX1.
import json, os, re
C = "/work/users/d/y/dyy12/XuLab/complex_screen"; O = f"{C}/fulllength_check/retest6"
os.makedirs(f"{O}/inputs", exist_ok=True); os.makedirs(f"{O}/msa", exist_ok=True); os.makedirs(f"{O}/logs", exist_ok=True)
def seq(g): return json.load(open(f"{C}/seqs/uniprot/{g}.json"))["sequence"]["value"]
def csv(g):
    dst = f"{O}/msa/{g}_full.csv"; rows = []
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
for a, b in [("DDIT3", "XPA"), ("DR1", "HOXC9"), ("DR1", "THAP6"), ("DR1", "ZNF746"), ("CBX1", "ZNF507"), ("ZNF143", "ZNF512")]:
    with open(f"{O}/inputs/{a}__{b}_full.yaml", "w") as f:
        f.write("version: 1\nsequences:\n")
        for i, g in zip("AB", (a, b)): f.write(f"  - protein:\n      id: {i}\n      sequence: {seq(g)}\n      msa: {csv(g)}\n")
    print(a, b, len(seq(a)) + len(seq(b)))
