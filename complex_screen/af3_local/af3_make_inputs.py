# AlphaFold 3 local run on Longleaf (module container/alphafold/3.0.1). Step "msa": one monomer JSON per unique protein (full-length
# UniProt canonical, as in fulllength_check) for the data pipeline. Step "pairs": pair JSONs (dialect alphafold3, seeds 1-5) reusing
# each protein's *_data.json MSAs and templates, for inference only. HP1 pair = CBX1, the paralog that scored best in the Boltz screen.
import json, os, sys
C = "/work/users/d/y/dyy12/XuLab/complex_screen"; O = f"{C}/af3_local"
def seq(g): return json.load(open(f"{C}/seqs/uniprot/{g}.json"))["sequence"]["value"]
CONTROLS = [("EZH2", "SUZ12"), ("RBBP7", "SRCAP")]
HITS = [("MLX", "RBBP7"), ("ZNF282", "ZNF746"), ("HOXB8", "RBPJ"), ("HOXA5", "RBBP4"), ("MLX", "SREBF2"), ("RBBP4", "RBBP7"),
        ("MTF2", "RBBP4"), ("HOXC4", "RBBP4"), ("EPAS1", "NEUROD4"), ("NEUROD4", "SREBF2"), ("RBBP7", "ZNF79"), ("SREBF2", "ZNF367"),
        ("NEUROD4", "ZNF746"), ("SREBF2", "ZNF746")]
EDGES = [("DDIT3", "XPA"), ("DR1", "HOXC9"), ("DR1", "THAP6"), ("DR1", "ZNF746"), ("CBX1", "ZNF507"), ("ZNF143", "ZNF512")]
WEAK = [("CHAMP1", "ZNF367"), ("DDIT3", "HES1"), ("DDIT3", "NR1D2"), ("DDIT3", "PA2G4"), ("DDIT3", "RELA"), ("DDIT3", "SAFB"), ("DR1", "SREBF2"), ("DR1", "ZNF143"),
        ("EPAS1", "ZNF507"), ("ETS2", "ZUP1"), ("HOXB6", "ZUP1"), ("HOXC9", "MLX"), ("HOXC9", "ZUP1"), ("MEF2A", "ZNF746"), ("MSX2", "ZNF777"), ("POGK", "ZNF143"),
        ("SREBF2", "THAP6"), ("ZNF143", "ZNF644")]  # complex_weak class, provisional R1 2026-11-01; ZUFSP = ZUP1
ALL = CONTROLS + HITS + EDGES + WEAK
step = sys.argv[1]
if step == "msa":
    os.makedirs(f"{O}/msa_in", exist_ok=True)
    order = [g for p in CONTROLS for g in p] + sorted({g for p in HITS + EDGES for g in p} - {g for p in CONTROLS for g in p})
    order += sorted({g for p in WEAK for g in p} - set(order))  # appended so earlier array indices keep their protein
    for g in order:
        json.dump({"name": g, "modelSeeds": [1], "dialect": "alphafold3", "version": 1, "sequences": [{"protein": {"id": "A", "sequence": seq(g)}}]},
                  open(f"{O}/msa_in/{g}.json", "w"))
    open(f"{O}/proteins.txt", "w").write("\n".join(order) + "\n")
    open(f"{O}/pairs.txt", "w").write("\n".join(f"{a}_{b}" for a, b in ALL) + "\n")
    print(len(order), "proteins;", len(ALL), "pairs; first 4 are the control proteins")
else:  # pairs [only these pair names]
    os.makedirs(f"{O}/pair_in", exist_ok=True); want = set(sys.argv[2:])
    for a, b in ALL:
        n = f"{a}_{b}"
        if want and n not in want: continue
        ch = []
        for cid, g in zip("AB", (a, b)):
            p = json.load(open(f"{O}/msa_out/{g.lower()}/{g.lower()}_data.json"))["sequences"][0]["protein"]; p["id"] = cid; ch.append({"protein": p})
        json.dump({"name": n, "modelSeeds": [1, 2, 3, 4, 5], "dialect": "alphafold3", "version": 1, "sequences": ch}, open(f"{O}/pair_in/{n}.json", "w"))
        print(n, sum(len(c["protein"]["sequence"]) for c in ch))
