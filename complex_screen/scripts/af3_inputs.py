# AlphaFold Server inputs (dialect "alphafoldserver", version 1) for the 14 strong non-STAG2 Boltz hits, full-length UniProt
# canonical sequences (as in complex_screen/fulllength_check), plus controls EZH2-SUZ12 (positive; SUZ12 full length from the
# cached UniProt entry) and RBBP7-SRCAP (negative). One JSON per job and one combined file; token totals vs the 5,000 limit.
import json, os
C = "/work/users/d/y/dyy12/XuLab/complex_screen"; O = f"{C}/af3_inputs"; os.makedirs(O, exist_ok=True)
def seq(g): return json.load(open(f"{C}/seqs/uniprot/{g}.json"))["sequence"]["value"]
PAIRS = [("MLX", "RBBP7"), ("ZNF282", "ZNF746"), ("HOXB8", "RBPJ"), ("HOXA5", "RBBP4"), ("MLX", "SREBF2"), ("RBBP4", "RBBP7"),
         ("MTF2", "RBBP4"), ("HOXC4", "RBBP4"), ("EPAS1", "NEUROD4"), ("NEUROD4", "SREBF2"), ("RBBP7", "ZNF79"), ("SREBF2", "ZNF367"),
         ("NEUROD4", "ZNF746"), ("SREBF2", "ZNF746"), ("EZH2", "SUZ12"), ("RBBP7", "SRCAP")]
jobs, rows = [], []
for a, b in PAIRS:
    sa, sb = seq(a), seq(b); name = f"{a}_{b}_fulllength"
    job = {"name": name, "modelSeeds": [], "dialect": "alphafoldserver", "version": 1,
           "sequences": [{"proteinChain": {"sequence": sa, "count": 1}}, {"proteinChain": {"sequence": sb, "count": 1}}]}
    json.dump([job], open(f"{O}/{name}.json", "w"), indent=1); jobs.append(job)
    kind = "positive control" if (a, b) == ("EZH2", "SUZ12") else ("negative control" if (a, b) == ("RBBP7", "SRCAP") else "strong non-STAG2 hit")
    rows.append(f"{name}\t{a}\t{len(sa)}\t{b}\t{len(sb)}\t{len(sa) + len(sb)}\t{'ok' if len(sa) + len(sb) <= 5000 else 'OVER 5000'}\t{kind}")
json.dump(jobs, open(f"{O}/all_jobs.json", "w"), indent=1)
open(f"{O}/jobs.tsv", "w").write("job\tchain_a\tlen_a\tchain_b\tlen_b\ttokens\tvs_5000_limit\ttype\n" + "\n".join(rows) + "\n")
print("\n".join(rows))
