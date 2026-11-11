# DNA-included AF3 test, Amendment 1 (DECLARATION.md): inputs for the pairs skipped for lack of a HOCOMOCO v11 model. Motif = first available of (a) the
# workbook's Codebook sheets (Codebook_rescued_from_unknown "Codebook motif", known_motifs "Codebook"; PWM from motifdb/codebook_top1.meme),
# (b) HOCOMOCO v12 CORE (motifdb/H12CORE_meme_format.meme; UniProt short name, rank-0 model), (c) JASPAR 2024 CORE (motifdb/JASPAR2024...meme, gene
# name). The PWM is turned into an IUPAC consensus (max p >= .5 -> base, upper case; else the smallest 2-3 base code reaching .75, lower case; else n),
# then the same site rule as build_inputs.py (summits in rank order, pass 1 all positions, pass 2 upper case only, 24 bp centred, motif on strand C).
# Appends rows to inputs.tsv (adds motif_source; earlier rows = "workbook HOCOMOCO v11"), writes run_in/<PAIR>.json and pairs_amend1.txt.
import json, os, re, subprocess
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; C = f"{X}/complex_screen"; G = f"{C}/af3_dna_included_20261111"; S5 = f"{C}/af3_corr060_20261104"; MD = f"{X}/motif_analysis/cache/motifdb"
L = [f"{S5}/msa_out", f"{C}/af3_novel_20261103/msa_out", f"{C}/af3_local/msa_out", f"{C}/af3_dna/msa_out"]; FA = f"{X}/downstream/cache/hg38.fa"
IUP = dict(A="A", C="C", G="G", T="T", R="[AG]", Y="[CT]", S="[CG]", W="[AT]", K="[GT]", M="[AC]", B="[CGT]", D="[AGT]", H="[ACT]", V="[ACG]", N="[ACGT]")
CODE = {frozenset(k): v for k, v in [("AG", "r"), ("CT", "y"), ("CG", "s"), ("AT", "w"), ("GT", "k"), ("AC", "m"), ("CGT", "b"), ("AGT", "d"), ("ACT", "h"), ("ACG", "v")]}
rc = lambda s: s[::-1].translate(str.maketrans("ACGTN", "TGCAN"))
PAIRS = [("HOXB9_RBBP7", "HOXB9"), ("HOXB9_RBBP4", "HOXB9"), ("HOXB9_XPA", "HOXB9"), ("HOXC4_ZUP1", "HOXC4"), ("RBBP7_ZNF703", "ZNF703"), ("ADNP_MTF2", "ADNP"),
         ("RBBP4_ZNF644", "ZNF644"), ("RBBP7_ZNF644", "ZNF644")]
def meme(f):
    D, ls, i = {}, open(f).read().split("\n"), 0
    while i < len(ls):
        if ls[i].startswith("MOTIF "):
            x = ls[i].split(); j = i + 1
            while not ls[j].startswith("letter-probability"): j += 1
            w = int(ls[j].split("w=")[1].split()[0]); D[(x[1], x[2] if len(x) > 2 else "")] = np.array([[float(v) for v in ls[j + 1 + k].split()] for k in range(w)]); i = j + w
        i += 1
    return D
CB, H12, JA = meme(f"{MD}/codebook_top1.meme"), meme(f"{MD}/H12CORE_meme_format.meme"), meme(f"{MD}/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt")
xl = pd.ExcelFile(f"{X}/motif_analysis/TF1000cells_motifs.xlsx"); R = xl.parse("Codebook_rescued_from_unknown", header=None).iloc[3:, :3]; K = xl.parse("known_motifs")
wb_cb = {str(t).upper(): str(m) for t, m in zip(R[0], R[2]) if isinstance(m, str)}
wb_cb.update({str(t).upper(): str(m) for t, m in zip(K.TF, K.Codebook) if isinstance(m, str) and m != "Missing" and str(t).upper() not in wb_cb})
short = lambda t: re.sub(r"^HOX", "HX", re.sub(r"^ZNF", "ZN", t))
def pick(t):
    if t in wb_cb:
        k = [k for k in CB if k[0].upper() == wb_cb[t].upper() or k[1].upper().startswith(wb_cb[t].upper() + ".")]
        if k: return "workbook Codebook (codebook_top1.meme)", f"{k[0][0]} {k[0][1]}", CB[k[0]]
    k = sorted([k for k in H12 if k[0].startswith(f"{short(t)}.H12CORE.0.")])
    if k: return "HOCOMOCO v12 CORE", k[0][0], H12[k[0]]
    k = sorted([k for k in JA if k[1].upper() == t])
    if k: return "JASPAR 2024 CORE", f"{k[0][0]} {k[0][1]}", JA[k[0]]
    return None, None, None
def cons(P):
    out = []
    for p in P:
        o = np.argsort(-p); b = "ACGT"
        if p[o[0]] >= .5: out.append(b[o[0]]); continue
        s = None
        for n in (2, 3):
            if p[o[:n]].sum() >= .75: s = CODE[frozenset(b[i] for i in o[:n])]; break
        out.append(s or "n")
    return "".join(out)
def site(t, cs):
    W = pd.read_csv(f"{C}/pair_motif_composite/profile/pools/{t}.tsv", sep="\t").sort_values("rank")
    out = subprocess.run(["samtools", "faidx", FA] + [f"{c}:{s + 1}-{e}" for c, s, e in zip(W.c, W.s, W.e)], capture_output=True, text=True, check=True).stdout.split(">")[1:]
    seqs = ["".join(x.split("\n")[1:]).upper() for x in out]; k = len(cs)
    for pas in (1, 2):
        pat = "".join(IUP[ch.upper()] if (pas == 1 or ch.isupper()) else "[ACGT]" for ch in cs)
        for (c, s, e, rk), sq in zip(zip(W.c, W.s, W.e, W["rank"]), seqs):
            hits = [(m.start(), "+") for m in re.finditer(f"(?={pat})", sq)] + [(len(sq) - m.start() - k, "-") for m in re.finditer(f"(?={pat})", rc(sq))]
            if not hits: continue
            i, st = min(hits, key=lambda h: abs(h[0] + k / 2 - len(sq) / 2)); a = s + i + k // 2 - 12
            dna = subprocess.run(["samtools", "faidx", FA, f"{c}:{a + 1}-{a + 24}"], capture_output=True, text=True, check=True).stdout.split("\n", 1)[1].replace("\n", "").upper()
            return dict(match_pass=pas, summit_rank=int(rk), summit_window=f"{c}:{s}-{e}", motif_site=f"{c}:{s + i}-{s + i + k} ({st})", dna=rc(dna) if st == "-" else dna)
    return None
def msa(g):
    for d in L:
        f = f"{d}/{g.lower()}/{g.lower()}_data.json"
        if os.path.exists(f): return f
    raise SystemExit("no MSA " + g)
I = pd.read_csv(f"{G}/inputs.tsv", sep="\t")
if "motif_source" not in I: I.insert(3, "motif_source", "workbook HOCOMOCO v11")
I = I[~(I.pair.isin([p for p, _ in PAIRS]) & I.status.str.startswith("UNINFORMATIVE (no HOCOMOCO"))]; new = []
for p, t in PAIRS:
    src, mdl, P = pick(t); r = dict(pair=p, tf=t, motif_model=mdl or "none", motif_source=src or "none", consensus=cons(P) if P is not None else "none", amendment="A1")
    if P is None: r["status"] = "UNINFORMATIVE (no motif in workbook Codebook, HOCOMOCO v12 CORE or JASPAR 2024 CORE)"; new.append(r); continue
    s = site(t, r["consensus"])
    if not s: r["status"] = "UNINFORMATIVE (no summit window with a consensus match)"; new.append(r); continue
    ch = []
    for cid, g in zip("AB", p.split("_")): q = json.load(open(msa(g)))["sequences"][0]["protein"]; q["id"] = cid; ch.append({"protein": q})
    ch += [{"dna": {"id": "C", "sequence": s["dna"]}}, {"dna": {"id": "D", "sequence": rc(s["dna"])}}]
    json.dump({"name": f"{p}_dna", "modelSeeds": [1, 2, 3, 4, 5], "dialect": "alphafold3", "version": 1, "sequences": ch}, open(f"{G}/run_in/{p}.json", "w"))
    r.update(status="built", match_pass=s["match_pass"], summit_rank=s["summit_rank"], summit_window=s["summit_window"], motif_site=s["motif_site"], dna_strand_C=s["dna"], dna_strand_D=rc(s["dna"])); new.append(r)
N = pd.DataFrame(new); pd.concat([I, N], ignore_index=True).to_csv(f"{G}/inputs.tsv", sep="\t", index=False)
N[N.status == "built"].pair.to_csv(f"{G}/pairs_amend1.txt", index=False, header=False)
print(N.drop(columns=["dna_strand_D"], errors="ignore").to_string(index=False))
