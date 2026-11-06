# Slice the full-length per-protein MSAs to each construct and write Boltz YAML inputs.
#   calib  : the 31 pilot pairs (ppi_pilot/pairs.tsv constructs), unpaired MSAs
#   screen : all unordered panel heterodimers + homodimers from chains.tsv
import sys, os, re, itertools
import pandas as pd

mode, root = sys.argv[1:3]
LINK = 10  # GS linker length used in trim_chains.py
os.makedirs(f"{root}/msa/csv", exist_ok=True)

def read_a3m(g):
    names, rows, cur = [], [], None
    for line in open(f"{root}/msa/full/{g}.a3m"):
        line = line.rstrip("\n").replace("\x00", "")
        if not line: continue
        if line.startswith(">"):
            names.append(line); rows.append("")
        else:
            rows[-1] += re.sub(r"[a-z.]", "", line)  # drop insertions -> query-aligned columns
    return rows

A3M = {}
def msa_csv(name, gene, segs, chain_seq):
    """segs: list of (start, end) 1-based on the full-length sequence; linkers -> gap columns"""
    dst = f"{root}/msa/csv/{name}.csv"
    if os.path.exists(dst): return dst
    if gene not in A3M: A3M[gene] = read_a3m(gene)
    rows = A3M[gene]
    out, seen = [chain_seq], {chain_seq}
    for r in rows[1:]:
        s = ("-" * LINK).join(r[a - 1:b] for a, b in segs)
        if set(s) <= {"-"} or s in seen: continue
        seen.add(s); out.append(s)
    assert all(len(x) == len(chain_seq) for x in out), name
    with open(dst, "w") as f:
        f.write("key,sequence\n" + "\n".join(f"-1,{x}" for x in out) + "\n")
    return dst

def yaml(path, chains):
    """chains: list of (ids, seq, msa_path); ids is a str or list (homomer copies)"""
    with open(path, "w") as f:
        f.write("version: 1\nsequences:\n")
        for ids, seq, msa in chains:
            idv = ids if isinstance(ids, str) else "[" + ", ".join(ids) + "]"
            f.write(f"  - protein:\n      id: {idv}\n      sequence: {seq}\n      msa: {msa}\n")

full = {}
for t in ["proteins.tsv", "proteins_offpanel.tsv"]:
    for _, r in pd.read_csv(f"{root}/seqs/{t}", sep="\t").iterrows():
        import json
        full[r.gene] = json.load(open(f"{root}/seqs/uniprot/{r.gene}.json"))["sequence"]["value"]

if mode == "calib":
    pairs = pd.read_csv(sys.argv[3], sep="\t")
    os.makedirs(f"{root}/calib/inputs", exist_ok=True)
    for _, p in pairs.iterrows():
        specs = [(p.gene_a, p.start_a, p.end_a), (p.gene_b, p.start_b, p.end_b)]
        if p.extra_chains != ".":
            for x in p.extra_chains.split(";"):
                g, _, rng = x.split(":"); a, b = map(int, rng.split("-")); specs.append((g, a, b))
        ch = []
        for i, (g, a, b) in enumerate(specs):
            g = {"ZUFSP": "ZUP1"}.get(g, g)
            seq = full[g][a - 1:b]
            ch.append(("ABCD"[i], seq, msa_csv(f"{g}_{a}-{b}", g, [(a, b)], seq)))
        yaml(f"{root}/calib/inputs/{p.pair_id}.yaml", ch)
    print("calib inputs:", len(pairs))

elif mode == "screen":
    ch = pd.read_csv(f"{root}/chains/chains.tsv", sep="\t")
    seqs = {}
    for l in open(f"{root}/chains/chains.fasta").read().split(">")[1:]:
        n, s = l.split("\n", 1); seqs[n] = s.replace("\n", "")
    msa = {}
    for _, r in ch.iterrows():
        segs = [tuple(map(int, x.split("-"))) for x in r.segments.split(",")]
        msa[r.chain_id] = msa_csv(f"chain_{r.chain_id}", r.chain_id, segs, seqs[r.chain_id])
    L = dict(zip(ch.chain_id, ch.chain_length))
    # panel units: HP1 = CBX1/CBX3/CBX5 -> CBX3 represents HP1 in the 3,081-pair set; CBX1/CBX5 go to a paralog set
    main = [c for c in ch.chain_id if c not in ("CBX1", "CBX5")]
    os.makedirs(f"{root}/jobs/inputs/het", exist_ok=True); os.makedirs(f"{root}/jobs/inputs/homo", exist_ok=True)
    os.makedirs(f"{root}/jobs/inputs/hp1_paralogs", exist_ok=True)
    rows = []
    for a, b in itertools.combinations(sorted(main), 2):
        pid = f"{a}__{b}"; yaml(f"{root}/jobs/inputs/het/{pid}.yaml", [("A", seqs[a], msa[a]), ("B", seqs[b], msa[b])])
        rows.append((pid, "het", a, b, L[a] + L[b]))
    for a in sorted(ch.chain_id):
        pid = f"{a}__{a}"; yaml(f"{root}/jobs/inputs/homo/{pid}.yaml", [(["A", "B"], seqs[a], msa[a])])
        rows.append((pid, "homo", a, a, 2 * L[a]))
    for p in ("CBX1", "CBX5"):
        for b in sorted(main):
            if b == "CBX3": continue
            pid = f"{p}__{b}"; yaml(f"{root}/jobs/inputs/hp1_paralogs/{pid}.yaml", [("A", seqs[p], msa[p]), ("B", seqs[b], msa[b])])
            rows.append((pid, "hp1_paralog", p, b, L[p] + L[b]))
    for a, b in [("CBX1", "CBX3"), ("CBX1", "CBX5"), ("CBX3", "CBX5")]:
        pid = f"{a}__{b}"; yaml(f"{root}/jobs/inputs/hp1_paralogs/{pid}.yaml", [("A", seqs[a], msa[a]), ("B", seqs[b], msa[b])])
        rows.append((pid, "hp1_paralog", a, b, L[a] + L[b]))
    df = pd.DataFrame(rows, columns=["pair_id", "set", "chain_a", "chain_b", "n_tokens"])
    df.to_csv(f"{root}/jobs/joblist.tsv", sep="\t", index=False)
    print(df.groupby("set").agg(n=("pair_id", "size"), tokens_median=("n_tokens", "median"), tokens_max=("n_tokens", "max")))
