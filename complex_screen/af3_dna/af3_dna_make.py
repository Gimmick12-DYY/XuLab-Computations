# bHLH-ZIP constructs and DNA duplexes for the AF3 + DNA runs. Step "msa": one monomer JSON per construct. Step "runs": run JSONs (seeds 1-5) reusing each construct's MSAs and templates, each DNA strand its own chain.
import json, os, random, re, sys
C = "/work/users/d/y/dyy12/XuLab/complex_screen"; O = f"{C}/af3_dna"
SRC = {"MLX": f"{C}/seqs/uniprot/MLX.json", "SREBF2": f"{C}/seqs/uniprot/SREBF2.json", "MLXIP": f"{O}/uniprot/Q9HAP2.json", "MAX": f"{O}/uniprot/P61244.json"}
R = {"MLX": (129, 220), "SREBF2": (330, 401), "MLXIP": (719, 790), "MAX": (23, 102)}
def seq(g): a, b = R[g]; return json.load(open(SRC[g]))["sequence"]["value"][a - 1:b]
comp = lambda s: s[::-1].translate(str.maketrans("ACGT", "TGCA"))
def dishuffle(s, rng):
    # Altschul-Erikson: random last-edge arborescence, then an Euler walk, keeps the dinucleotide counts exactly
    nts = sorted(set(s)); edges = {n: [] for n in nts}
    for x, y in zip(s, s[1:]): edges[x].append(y)
    while True:
        last = {}; ok = True
        for n in nts:
            if n == s[-1] or not edges[n]: continue
            last[n] = rng.choice(edges[n])
        for n in last:
            x = n; seen = set()
            while x != s[-1] and x in last and x not in seen: seen.add(x); x = last[x]
            ok &= x == s[-1]
        if ok: break
    lists = {n: list(edges[n]) for n in nts}
    for n, y in last.items(): lists[n].remove(y)
    for n in nts: rng.shuffle(lists[n])
    for n, y in last.items(): lists[n].append(y)
    o = [s[0]]
    for _ in s[1:]: o.append(lists[o[-1]].pop(0))
    return "".join(o)
def mism(s, m): return min(sum(a != b for a, b in zip(s[i:i + len(m)], m)) for i in range(len(s) - len(m) + 1))
EB = "CGAGTAGCACGTGCTACTC"; SRE = ("TTGCAGTGGGGTGATCT", "CATGAGATCACCCCACTGCAA"); rng = random.Random(293)
def good(s):
    t = "C" + comp(s[1:])
    return not any(re.search("CA..TG", x) or mism(x, "ATCACCCCAC") <= 2 or mism(comp(x), "ATCACCCCAC") <= 2 for x in (s, t))
while True:
    sc = dishuffle(EB, rng)
    if good(sc) and sc != EB: break
SC = (sc, "C" + comp(sc[1:]))
DNA = {"Ebox": (EB, EB), "SRE": SRE, "scrambled": SC}
RUNS = [("MLX", "SREBF2", "Ebox"), ("MLX", "SREBF2", "SRE"), ("MLX", "SREBF2", "scrambled"), ("MLX", "MLXIP", "Ebox"), ("SREBF2", "SREBF2", "SRE"), ("MAX", "MLX", "Ebox")]
step = sys.argv[1]
if step == "msa":
    os.makedirs(f"{O}/msa_in", exist_ok=True); os.makedirs(f"{O}/msa_out", exist_ok=True)
    for g in R: json.dump({"name": f"{g}_bHLHZIP", "modelSeeds": [1], "dialect": "alphafold3", "version": 1, "sequences": [{"protein": {"id": "A", "sequence": seq(g)}}]}, open(f"{O}/msa_in/{g}.json", "w"))
    open(f"{O}/proteins.txt", "w").write("\n".join(R) + "\n"); open(f"{O}/runs.txt", "w").write("\n".join(f"{a}_{b}_{d}" for a, b, d in RUNS) + "\n")
    with open(f"{O}/constructs.tsv", "w") as f:
        f.write("name\tsource\tstart\tend\tlength\tsequence\n")
        for g in R: f.write(f"{g}\t{os.path.basename(SRC[g])}\t{R[g][0]}\t{R[g][1]}\t{len(seq(g))}\t{seq(g)}\n")
        for k, (x, y) in DNA.items(): f.write(f"{k}_strand1\t.\t.\t.\t{len(x)}\t{x}\n{k}_strand2\t.\t.\t.\t{len(y)}\t{y}\n")
    for g in R: print(g, R[g], len(seq(g)), seq(g))
    for k, v in DNA.items(): print(k, v, "paired", all(a == b for a, b in zip(v[0][1:], comp(v[1])[:len(v[0]) - 1])) if k != "SRE" else "", "dinuc same", sorted(zip(v[0], v[0][1:])) == sorted(zip(EB, EB[1:])))
else:
    os.makedirs(f"{O}/run_in", exist_ok=True)
    for a, b, d in RUNS:
        ch = []
        for cid, g in zip("AB", (a, b)):
            p = json.load(open(f"{O}/msa_out/{g.lower()}_bhlhzip/{g.lower()}_bhlhzip_data.json"))["sequences"][0]["protein"]; p["id"] = cid; ch.append({"protein": p})
        ch += [{"dna": {"id": "C", "sequence": DNA[d][0]}}, {"dna": {"id": "D", "sequence": DNA[d][1]}}]
        json.dump({"name": f"{a}_{b}_{d}", "modelSeeds": [1, 2, 3, 4, 5], "dialect": "alphafold3", "version": 1, "sequences": ch}, open(f"{O}/run_in/{a}_{b}_{d}.json", "w")); print(a, b, d)
