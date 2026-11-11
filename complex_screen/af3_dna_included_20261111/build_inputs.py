# DNA-included AF3 test (DECLARATION.md, committed before this script): builds one AF3 JSON per pair = TF protein (chain A/B in pair order) + partner +
# a 24-bp duplex (chains C, D) carrying the flagged TF's top motif. Motif = the TF's rank-0 HOCOMOCO v11 model in TF1000cells_motifs.xlsx
# (sheet HOCOMOCOv11_full_annotation_HUM; the workbook's AME runs used HOCOMOCO v11); TFs absent from that sheet have no motif -> pair UNINFORMATIVE,
# skipped. Flanks: the TF's raw HITAG MACS2 summit windows (pair_motif_composite/profile/pools/<TF>.tsv, summit +/- 100 bp, ranked by -log10 p), taken in
# rank order; the first window with a consensus match (IUPAC, both strands; pass 1 all positions, pass 2 upper-case positions only) gives a 24-bp hg38
# sequence centred on the match (closest match to the summit), written so the motif reads on strand C. Proteins reuse the cached per-protein MSAs and
# templates (as make_one.py). Seeds 1-5. Writes inputs.tsv and run_in/<PAIR>.json. usage: build_inputs.py
import json, os, re, subprocess
import pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; C = f"{X}/complex_screen"; G = f"{C}/af3_dna_included_20261111"; S5 = f"{C}/af3_corr060_20261104"
L = [f"{S5}/msa_out", f"{C}/af3_novel_20261103/msa_out", f"{C}/af3_local/msa_out", f"{C}/af3_dna/msa_out"]; FA = f"{X}/downstream/cache/hg38.fa"
IUP = dict(A="A", C="C", G="G", T="T", R="[AG]", Y="[CT]", S="[CG]", W="[AT]", K="[GT]", M="[AC]", B="[CGT]", D="[AGT]", H="[ACT]", V="[ACG]", N="[ACGT]")
rc = lambda s: s[::-1].translate(str.maketrans("ACGTN", "TGCAN"))
F = pd.read_csv(f"{S5}/rescore_5seed/dbd_face.tsv", sep="\t"); fl = F[F.flag == "DBD-face"]
PAIRS = [(p, t) for p, t in zip(fl.pair, fl.tf)] + [("MTF2_RBBP4", "MTF2"), ("DDIT3_MTF2", "DDIT3")]
xl = pd.ExcelFile(f"{X}/motif_analysis/TF1000cells_motifs.xlsx"); H = xl.parse("HOCOMOCOv11_full_annotation_HUM")
def motif(t):
    h = H[H["Transcription factor"].astype(str) == t].sort_values("Model rank"); return (h.Model.iloc[0], h.Consensus.iloc[0]) if len(h) else (None, None)
def msa(g):
    for d in L:
        f = f"{d}/{g.lower()}/{g.lower()}_data.json"
        if os.path.exists(f): return f
    raise SystemExit("no MSA " + g)
def site(t, cons):
    W = pd.read_csv(f"{C}/pair_motif_composite/profile/pools/{t}.tsv", sep="\t").sort_values("rank")
    reg = [f"{c}:{s + 1}-{e}" for c, s, e in zip(W.c, W.s, W.e)]
    out = subprocess.run(["samtools", "faidx", FA] + reg, capture_output=True, text=True, check=True).stdout.split(">")[1:]
    seqs = ["".join(x.split("\n")[1:]).upper() for x in out]
    for pas in (1, 2):
        pat = "".join(IUP[ch.upper()] if (pas == 1 or ch.isupper()) else "[ACGT]" for ch in cons); k = len(cons)
        for (c, s, e, rk), sq in zip(zip(W.c, W.s, W.e, W["rank"]), seqs):
            hits = [(m.start(), "+") for m in re.finditer(f"(?={pat})", sq)] + [(m.start(), "-") for m in re.finditer(f"(?={pat})", rc(sq))]
            if not hits: continue
            hits = [(i if st == "+" else len(sq) - i - k, st) for i, st in hits]; i, st = min(hits, key=lambda h: abs(h[0] + k / 2 - len(sq) / 2))
            a = s + i + k // 2 - 12; dna = subprocess.run(["samtools", "faidx", FA, f"{c}:{a + 1}-{a + 24}"], capture_output=True, text=True, check=True).stdout.split("\n", 1)[1].replace("\n", "").upper()
            if st == "-": dna = rc(dna)
            return dict(pass_=pas, summit_rank=int(rk), window=f"{c}:{s}-{e}", motif_site=f"{c}:{s + i}-{s + i + k} ({st})", dna=dna)
    return None
os.makedirs(f"{G}/run_in", exist_ok=True); rows = []
for p, t in PAIRS:
    mdl, cons = motif(t); r = dict(pair=p, tf=t, motif_model=mdl or "none", consensus=cons or "none")
    if not mdl: r.update(status="UNINFORMATIVE (no HOCOMOCO v11 motif in workbook)"); rows.append(r); continue
    s = site(t, cons)
    if not s: r.update(status="UNINFORMATIVE (no summit window with a consensus match)"); rows.append(r); continue
    ch = []
    for cid, g in zip("AB", p.split("_")): q = json.load(open(msa(g)))["sequences"][0]["protein"]; q["id"] = cid; ch.append({"protein": q})
    ch += [{"dna": {"id": "C", "sequence": s["dna"]}}, {"dna": {"id": "D", "sequence": rc(s["dna"])}}]
    json.dump({"name": f"{p}_dna", "modelSeeds": [1, 2, 3, 4, 5], "dialect": "alphafold3", "version": 1, "sequences": ch}, open(f"{G}/run_in/{p}.json", "w"))
    r.update(status="built", match_pass=s["pass_"], summit_rank=s["summit_rank"], summit_window=s["window"], motif_site=s["motif_site"], dna_strand_C=s["dna"], dna_strand_D=rc(s["dna"])); rows.append(r)
T = pd.DataFrame(rows); T.to_csv(f"{G}/inputs.tsv", sep="\t", index=False)
T[T.status == "built"].pair.to_csv(f"{G}/pairs.txt", index=False, header=False)
print(T.drop(columns=["dna_strand_D"]).to_string(index=False))
