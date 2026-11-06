# Build the PPI pilot pair list (pairs.tsv) and Boltz YAML inputs from UniProt JSON
import json, sys, random, glob, os
import pandas as pd

up_dir, pair_scores, neg_pool, out = sys.argv[1:5]
os.makedirs(f"{out}/inputs", exist_ok=True)
seq = {}
for f in glob.glob(f"{up_dir}/*.json"):
    d = json.load(open(f))
    seq[d["genes"][0]["geneName"]["value"]] = (d["primaryAccession"], d["sequence"]["value"])

# (class, chainA spec, chainB spec, extra chains, rationale); spec = (gene, start, end)
P = []
def add(cls, a, b, why, extra=()):
    P.append((cls, a, b, extra, why))

# direct-contact positives (structure or mapped interface)
add("positive", ("CTCF", 180, 265), ("STAG2", 80, 1060), "CTCF N-term YDF motif (aa ~226) binds SA2-SCC1 (Li 2020); SA2 construct 80-1060; CTCF ZFs excluded")
add("positive", ("CTCF", 180, 265), ("STAG2", 80, 1060), "as above, with SCC1/RAD21 281-420 as third chain (composite interface)", extra=(("RAD21", 281, 420),))
add("positive", ("STAG2", 80, 1060), ("RAD21", 281, 420), "SA2-SCC1 interface; RAD21 region incl. STAG-binding 362-403")
add("positive", ("EZH2", 1, 746), ("SUZ12", 400, 739), "PRC2 core; SUZ12 VEFS + C-term contacts EZH2; SUZ12 N-term (JARID2/AEBP2 sites) dropped")
add("positive", ("MBD3", 1, 291), ("GATAD2A", 100, 250), "NuRD MBD coiled-coil (216-245) - GATAD2A CR1 (139-178); used instead of CHD4 (1912 aa, no direct MBD3 contact)")
add("positive", ("ADNP", 740, 900), ("CBX3", 100, 183), "ChAHP: ADNP PxVxL (PGVLL, aa 820) binds HP1 chromoshadow dimer; CBX3 as 2 copies", extra=(("CBX3", 100, 183),))
add("positive", ("DR1", 1, 176), ("DRAP1", 1, 205), "NC2 histone-fold heterodimer (DR1-DRAP1); off-panel partner, control only")
add("positive", ("DDIT3", 1, 169), ("CEBPB", 250, 345), "CHOP-C/EBP bZIP heterodimer; CEBPB bZIP 250-345; off-panel partner, control only")
# same complex, direct contact not established (reported separately)
add("comember", ("MTF2", 1, 593), ("RBBP4", 1, 425), "PRC2.1 co-members; PCL binds SUZ12, RBBP4 binds SUZ12 N-term; no known direct contact")
add("comember", ("EZH2", 1, 746), ("RBBP7", 1, 469), "PRC2 co-members; RBBP7 contacts SUZ12, direct EZH2 contact not established")
add("comember", ("MBD3", 1, 291), ("RBBP4", 1, 425), "NuRD co-members; RBBP4 binds MTA1, not MBD3")
# known complex, interface not mapped (exploratory fragments)
add("uncertain", ("WIZ", 1440, 1651), ("EHMT2", 600, 1210), "G9a/GLP-WIZ; WIZ C-term zinc fingers reported to bind G9a; EHMT2 ANK+SET; boundaries uncertain")
add("uncertain", ("WIZ", 1440, 1651), ("ZNF644", 900, 1327), "G9a/GLP-WIZ-ZNF644; interface unmapped; C-terminal fragments are a guess")
# co-binding candidates (top Pearson pairs not in a known complex + requested pairs)
for a, b, why in [("RBBP4", "XPA", "top co-binding pair"), ("E4F1", "MLX", "top co-binding"), ("NFYA", "MLX", "requested"),
                  ("NFYA", "SREBF2", "requested; NF-Y/SREBP promoter cooperation"), ("MLX", "SREBF2", "requested"),
                  ("E2F3", "RBBP4", "requested; E2F-RB-RBBP4 axis"), ("MAZ", "ZNF143", "co-binding; both promoter ZNFs"),
                  ("DR1", "MBD3", "co-binding"), ("DDIT3", "RBBP4", "co-binding"), ("HOXC9", "ZUP1", "co-binding (ZUFSP)"),
                  ("MBD3", "NME2", "co-binding")]:
    sa = len(seq[a][1]); sb = len(seq[b][1])
    A = (a, 1, sa); B = (b, 1, sb)
    if a == "SREBF2": A = (a, 1, 480)
    if b == "SREBF2": B = (b, 1, 480); why += "; SREBF2 1-480 = nuclear (cleaved) form with bHLH-ZIP"
    if a == "E4F1": why += "; full length 784"
    add("candidate", A, B, why)
# negatives: random panel pairs, low co-binding (score pct < 0.3 in both), no STRING, both <= 750 aa
ps = pd.read_csv(pair_scores, sep="\t")
pool = pd.read_csv(neg_pool, sep="\t", header=None, names=["acc", "gene", "len"])
pool = pool[pool.gene.isin(["HOXA5", "HOXB8", "SOX4", "FOXC1", "RELA", "NR1D2", "HES1", "MSX2", "ETS2", "ZIC2", "MEF2D",
                            "SOX12", "FOXJ3", "PA2G4", "NRF1", "ZBTB7A", "ZBTB43", "THAP6", "HOXB6", "HOXC4", "HOXA10", "RBPJ", "TBX3", "FOXK1"])]
ok = set(pool.gene.str.lower())
low = ps[(ps.pct_enrich < 0.3) & (ps.pct_spqn < 0.3) & (ps.string == 0) & ps.a.isin(ok) & ps.b.isin(ok)]
random.seed(1)
used, negs = set(), []
for i in random.sample(range(len(low)), len(low)):
    r = low.iloc[i]
    if r.a in used or r.b in used: continue
    negs.append((r.a.upper(), r.b.upper())); used |= {r.a, r.b}
    if len(negs) == 8: break
for a, b in negs:
    add("negative", (a, 1, None), (b, 1, None), "random panel pair; co-binding pct < 0.3 in both matrices; no STRING edge")
print("low-co-binding pool pairs:", len(low))

def fetch(g):
    if g in seq: return seq[g]
    import urllib.request
    acc = pool.set_index("gene").loc[g, "acc"]
    d = json.load(urllib.request.urlopen(f"https://rest.uniprot.org/uniprotkb/{acc}.json"))
    seq[g] = (d["primaryAccession"], d["sequence"]["value"]); return seq[g]

rows = []
for i, (cls, A, B, extra, why) in enumerate(P, 1):
    chains = [A, B] + list(extra)
    specs = []
    for g, s, e in chains:
        acc, sq = fetch(g); e = e or len(sq)
        specs.append((g, acc, s, e, sq[s - 1:e]))
    pid = f"p{i:02d}_{specs[0][0]}_{specs[1][0]}" + ("_" + "_".join(x[0] for x in specs[2:]) if extra else "")
    letters = "ABCD"
    with open(f"{out}/inputs/{pid}.yaml", "w") as f:
        f.write("version: 1\nsequences:\n")
        for L, sp in zip(letters, specs):
            f.write(f"  - protein:\n      id: {L}\n      sequence: {sp[4]}\n")
    rows.append(dict(pair_id=pid, cls=cls, gene_a=specs[0][0], uniprot_a=specs[0][1], start_a=specs[0][2], end_a=specs[0][3],
                     gene_b=specs[1][0], uniprot_b=specs[1][1], start_b=specs[1][2], end_b=specs[1][3],
                     extra_chains=";".join(f"{x[0]}:{x[1]}:{x[2]}-{x[3]}" for x in specs[2:]) or ".",
                     n_tokens=sum(len(x[4]) for x in specs), rationale=why))
df = pd.DataFrame(rows)
df.to_csv(f"{out}/pairs.tsv", sep="\t", index=False)
print(df[["pair_id", "cls", "n_tokens"]].to_string())
