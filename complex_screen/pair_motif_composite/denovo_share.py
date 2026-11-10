# Fallback (iv): de novo-only motif sharing between TFs. All HOMER de novo motifs of all TFs in a per-TF run (TFBASE, default per_tf_raw)
# with fold >= 1.5 and p <= 1e-10 are pooled and clustered greedily (strongest first; a motif joins the first family whose
# representative has HOMER-style column-Pearson similarity >= 0.6, else founds a family). compareMotifs.pl took ~8 min per 50 motifs here. A family is shared by a pair
# when both TFs have a member. Families present in more than MAXTF TFs are treated as generic. novel = no member has a HOMER BestGuess
# score >= 0.6 (runs used -noknown, so novelty comes from Tomtom of each family representative vs db/known.meme: novel = no hit q < 0.05). Pair statistic: shared non-generic families; empirical p vs 200 random panel pairs matched on depth (cell tertiles) and
# correlation-percentile bin (same scheme as novel_pairs.py). Output: <TFBASE>/denovo_families.tsv, denovo_pairs.tsv, denovo_rates.tsv
import os, re, glob, subprocess, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import summarize as S
X = "/work/users/d/y/dyy12/XuLab"; O = os.path.dirname(os.path.abspath(__file__)); G = os.environ.get("TFBASE", f"{O}/per_tf_raw"); MAXTF = 10; rng = np.random.default_rng(20261110)
allm = []
for t in sorted(os.listdir(f"{G}/results")):
    for f in sorted(glob.glob(f"{G}/results/{t}/L*/homerMotifs.all.motifs")):
        for m in S.denovo(f):
            if m["fold"] >= 1.5 and m["log10p"] <= -10:
                h = m["raw"].split("\n"); x = h[0].split("\t"); sc = re.search(r"\(([\d.]+)\)$", x[1]); x[1] = f"{t}|{len(allm)}|{x[1].split('BestGuess:')[-1]}"; h[0] = "\t".join(x)
                allm.append(dict(tf=t, id=len(allm), cons=m["cons"], best=float(sc.group(1)) if sc else 0.0, guess=x[1].split("|", 2)[2], raw="\n".join(h)))
print(len(allm), "de novo motifs from", len({m["tf"] for m in allm}), "TFs", flush=True)
def mat(raw): return np.array([[float(v) for v in l.split()] for l in raw.split("\n")[1:] if l.strip()])
def simf(a, b):
    best = -1.0
    for bb in (b, b[::-1, ::-1]):
        La, Lb = len(a), len(bb); mn = min(La, Lb) // 2 + 1
        for off in range(-(Lb - 1), La):
            lo, hi = min(0, off), max(La, off + Lb)
            if min(La, off + Lb) - max(0, off) < mn: continue
            A = np.full((hi - lo, 4), .25); B = A.copy(); A[-lo:-lo + La] = a; B[off - lo:off - lo + Lb] = bb
            x = A.ravel() - A.mean(); y = B.ravel() - B.mean(); d = np.sqrt((x * x).sum() * (y * y).sum())
            if d > 0: best = max(best, float((x * y).sum() / d))
    return best
order = sorted(range(len(allm)), key=lambda i: -len(allm[i]["cons"])); reps = []; fam = {}
for i in order:
    mi = mat(allm[i]["raw"]); hit = None
    for k, (j, mj) in enumerate(reps):
        if simf(mi, mj) >= 0.6: hit = k; break
    if hit is None: reps.append((i, mi)); hit = len(reps) - 1
    fam.setdefault(hit, set()).add(i)
mm = "MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\nBackground letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n"
for k, (i, mi) in enumerate(reps): mm += f"MOTIF fam{k}\nletter-probability matrix: alength= 4 w= {len(mi)}\n" + "\n".join(" ".join(f"{v:.4f}" for v in r) for r in mi) + "\n\n"
os.makedirs(f"{G}/denovo_cmp", exist_ok=True); open(f"{G}/denovo_cmp/reps.meme", "w").write(mm)
subprocess.run(f"module load meme/5.5.7 >/dev/null 2>&1; tomtom -no-ssc -verbosity 1 -text -min-overlap 5 -dist pearson -thresh 0.05 {G}/denovo_cmp/reps.meme {O}/db/known.meme > {G}/denovo_cmp/tomtom.tsv", shell=True, executable="/bin/bash")
tt = pd.read_csv(f"{G}/denovo_cmp/tomtom.tsv", sep="\t", comment="#"); tt = tt.sort_values("q-value").drop_duplicates("Query_ID").set_index("Query_ID")
F = []
for k, ids in sorted(fam.items()):
    mem = [allm[i] for i in ids]; tf_ = sorted({m["tf"] for m in mem})
    F.append(dict(family=k, n_tf=len(tf_), tfs=",".join(tf_), cons=min(mem, key=lambda m: len(m["cons"]))["cons"], novel=f"fam{k}" not in tt.index, guess=tt["Target_ID"].get(f"fam{k}", "none")))
F = pd.DataFrame(F); F.to_csv(f"{G}/denovo_families.tsv", sep="\t", index=False); print(len(F), "families;", int((F.n_tf <= MAXTF).sum()), "non-generic;", int(F.novel.sum()), "novel", flush=True)
tfs = sorted({m["tf"] for m in allm}); has = {t: set(F[F.tfs.str.split(",").apply(lambda L: t in L) & (F.n_tf <= MAXTF)].family) for t in tfs}
R1 = pd.read_csv(f"{X}/regulatome/r1_tftf/v2/r1v2_pairs_provisional.tsv", sep="\t"); R1 = R1[R1.a.isin(tfs) & R1.b.isin(tfs)].copy()
R1["pct"] = R1[["pearson_domain_pct", "spqn_noquies_pct"]].max(1); R1["pair"] = R1.a + "-" + R1.b; R1["novel"] = R1.string_physical.isna() & ~R1.capselex.astype(str).str.startswith("called")
cells = pd.read_csv(f"{X}/complex_screen/pair_motif_lookup/tf_cells.tsv", sep="\t").set_index("tf").n_cells; tert = np.quantile(cells.values, [1 / 3, 2 / 3])
cb = {t: int(np.searchsorted(tert, cells[t])) if t in cells.index else 3 for t in tfs}
R1["dbin"] = [tuple(sorted((cb[a], cb[b]))) for a, b in zip(R1.a, R1.b)]; R1["pbin"] = pd.cut(R1.pct, [-1, .6, .8, .9, 1.01], labels=False)
R1["n_shared"] = [len(has[a] & has[b]) for a, b in zip(R1.a, R1.b)]
R1["n_shared_novel"] = [len({f for f in has[a] & has[b] if F.set_index("family").novel[f]}) for a, b in zip(R1.a, R1.b)]
R1["shared"] = [";".join(f"{F.set_index('family').cons[f]}({F.set_index('family').n_tf[f]}TF{',novel' if F.set_index('family').novel[f] else ','+F.set_index('family').guess[f].split('/')[0]})" for f in sorted(has[a] & has[b])) for a, b in zip(R1.a, R1.b)]
def emp(r):
    pool = R1[(R1.dbin == r.dbin) & (R1.pbin == r.pbin) & (R1.pair != r.pair)]
    if len(pool) < 20: pool = R1[(R1.dbin == r.dbin) & (R1.pair != r.pair)]
    nul = rng.choice(pool.n_shared.values, 200, replace=True); return (1 + np.sum(nul >= r.n_shared)) / 201
R1["emp_p"] = [emp(r) if r.n_shared else 1.0 for _, r in R1.iterrows()]
R1[["pair", "pct", "novel", "n_shared", "n_shared_novel", "emp_p", "shared"]].sort_values(["emp_p", "pct"], ascending=[True, False]).to_csv(f"{G}/denovo_pairs.tsv", sep="\t", index=False)
nv = R1[R1.novel]; rows = []
for lab, s in [("novel pct>=0.90", nv[nv.pct >= .9]), ("novel 0.80-0.90", nv[(nv.pct >= .8) & (nv.pct < .9)]), ("novel 0.60-0.80", nv[(nv.pct >= .6) & (nv.pct < .8)]), ("all pairs", R1)]:
    rows.append(dict(set=lab, n=len(s), frac_any_shared=round((s.n_shared > 0).mean(), 3), frac_emp_le05=round((s.emp_p <= .05).mean(), 3), frac_novel_word=round((s.n_shared_novel > 0).mean(), 3)))
pd.DataFrame(rows).to_csv(f"{G}/denovo_rates.tsv", sep="\t", index=False); print(pd.DataFrame(rows).to_string())
for p in ["CTCF-STAG2", "MAZ-ZNF143", "NFYA-ZNF644", "EZH2-MTF2", "RBBP7-SRCAP", "DDIT3-RBBP4", "DDIT3-RBBP7"]:
    s = R1[(R1.pair == p) | (R1.b + "-" + R1.a == p)]
    if len(s): print(p, int(s.n_shared.iloc[0]), round(s.emp_p.iloc[0], 3), s.shared.iloc[0][:200])
