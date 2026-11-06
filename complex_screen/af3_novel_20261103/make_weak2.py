import json, os
C = "/work/users/d/y/dyy12/XuLab/complex_screen"; O = f"{C}/af3_novel_20261103"; L = [f"{O}/msa_out", f"{C}/af3_local/msa_out", f"{C}/af3_dna/msa_out"]; os.makedirs(f"{O}/pair_in_weak"); n_ = 0
def path(g):
    for d in L:
        f = f"{d}/{g.lower()}/{g.lower()}_data.json"
        if os.path.exists(f): return f
for n in [l.strip() for l in open(f"{O}/pairs_weak.txt")]:
    ch = []
    for cid, g in zip("AB", n.split("_")): p = json.load(open(path(g)))["sequences"][0]["protein"]; p["id"] = cid; ch.append({"protein": p})
    json.dump({"name": n, "modelSeeds": [1], "dialect": "alphafold3", "version": 1, "sequences": ch}, open(f"{O}/pair_in_weak/{n}.json", "w")); n_ += 1
print(n_)
