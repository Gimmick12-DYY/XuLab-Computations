# Builds one AF3 pair input at run time from cached per-protein MSAs. usage: make_one.py A_B seeds(comma) out.json
import json, os, sys
C = "/work/users/d/y/dyy12/XuLab/complex_screen"; O = f"{C}/af3_corr060_20261104"; L = [f"{O}/msa_out", f"{C}/af3_novel_20261103/msa_out", f"{C}/af3_local/msa_out", f"{C}/af3_dna/msa_out"]
def path(g):
    for d in L:
        f = f"{d}/{g.lower()}/{g.lower()}_data.json"
        if os.path.exists(f): return f
    raise SystemExit("no MSA " + g)
n, sd, out = sys.argv[1:4]; ch = []
for cid, g in zip("AB", n.split("_")): p = json.load(open(path(g)))["sequences"][0]["protein"]; p["id"] = cid; ch.append({"protein": p})
json.dump({"name": n, "modelSeeds": [int(x) for x in sd.split(",")], "dialect": "alphafold3", "version": 1, "sequences": ch}, open(out, "w"))
