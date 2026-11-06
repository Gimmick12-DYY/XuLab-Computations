# AF3 on novel candidates: full-length pairs, seeds 1-5, same pipeline as af3_local. Controls first, then the slide 1 Boltz-only rows, then the other Boltz-strong candidates. MSAs reused from af3_local where they exist.
import json, os
C = "/work/users/d/y/dyy12/XuLab/complex_screen"; O = f"{C}/af3_novel_20261103"; L0 = f"{C}/af3_local/msa_out"; L1 = f"{O}/msa_out"
rows = [l.rstrip("\n").split("\t") for l in open(f"{O}/list.tsv")][1:]; PAIRS = [("EZH2", "SUZ12"), ("RBBP7", "SRCAP")] + [(r[0], r[1]) for r in rows]
def have(g): return os.path.exists(f"{L0}/{g.lower()}/{g.lower()}_data.json") or os.path.exists(f"{L1}/{g.lower()}/{g.lower()}_data.json")
def seq(g): return json.load(open(f"{C}/seqs/uniprot/{g}.json"))["sequence"]["value"]
miss = sorted({g for p in PAIRS for g in p if not have(g)})
for g in miss: json.dump({"name": g, "modelSeeds": [1], "dialect": "alphafold3", "version": 1, "sequences": [{"protein": {"id": "A", "sequence": seq(g)}}]}, open(f"{O}/msa_in/{g}.json", "w"))
open(f"{O}/proteins.txt", "w").write("".join(g + "\n" for g in miss)); now, later = [], []
for a, b in PAIRS:
    n = f"{a}_{b}"
    if have(a) and have(b):
        ch = []
        for cid, g in zip("AB", (a, b)):
            f = f"{L1}/{g.lower()}/{g.lower()}_data.json"; f = f if os.path.exists(f) else f"{L0}/{g.lower()}/{g.lower()}_data.json"; p = json.load(open(f))["sequences"][0]["protein"]; p["id"] = cid; ch.append({"protein": p})
        json.dump({"name": n, "modelSeeds": [1, 2, 3, 4, 5], "dialect": "alphafold3", "version": 1, "sequences": ch}, open(f"{O}/pair_in/{n}.json", "w")); now.append(n); print(n, sum(len(c["protein"]["sequence"]) for c in ch))
    else: later.append(n); print(n, "waits for MSA")
open(f"{O}/pairs.txt", "w").write("".join(x + "\n" for x in now + later)); open(f"{O}/pairs_ready.count", "w").write(str(len(now))); print("MSA needed:", miss, "| ready", len(now), "| waiting", len(later))
