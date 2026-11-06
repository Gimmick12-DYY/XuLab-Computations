# EZH2-MTF2 dimer and EZH2-SUZ12-MTF2 trimer for the PRC2 example slides: reuses the per-protein MSAs and templates, inference only, seeds 1-5.
import json, os
C = "/work/users/d/y/dyy12/XuLab/complex_screen"; O = f"{C}/af3_local"; os.makedirs(f"{O}/jie_in", exist_ok=True)
for n, gs in (("EZH2_MTF2", ["EZH2", "MTF2"]), ("EZH2_SUZ12_MTF2", ["EZH2", "SUZ12", "MTF2"])):
    ch = []
    for cid, g in zip("ABC", gs):
        p = json.load(open(f"{O}/msa_out/{g.lower()}/{g.lower()}_data.json"))["sequences"][0]["protein"]; p["id"] = cid; ch.append({"protein": p})
    json.dump({"name": n, "modelSeeds": [1, 2, 3, 4, 5], "dialect": "alphafold3", "version": 1, "sequences": ch}, open(f"{O}/jie_in/{n}.json", "w")); print(n, sum(len(c["protein"]["sequence"]) for c in ch))
