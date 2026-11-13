# Composite + gapped motif study, summary. Composite: de novo motifs (HOMER, discovery chr, co-bound vs A-only + B-only matched background) that
#  (i) reach discovery p <= 1e-5 with fold >= 1.5, (ii) validate on held-out chr at p <= 0.05 (HOMER -mknown with the de novo motifs), and
#  (iii) do not match A's or B's own motif family (Tomtom vs the 2,528 DB, q < 0.05: novel = no match; otherwise the best DB match is shown).
# Gapped: SpaMo spacing / orientation preferences (adjusted p <= 0.05) of a secondary DB motif around the primary (own) motif, present in co-bound
#  sites on BOTH halves and absent from the A-only control on both halves. Null = the same rates in 60 random non-candidate pairs.
# Output composite_gapped/{composite_hits.tsv, gapped_hits.tsv, pair_summary.tsv} (+ xlsx on the Mac)
import os, re, glob, math, subprocess
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; G = f"{O}/composite_gapped"
P = pd.read_csv(f"{G}/pairs.tsv", sep="\t"); tasks = [l.strip() for l in open(f"{G}/tasks.tsv") if l.strip()]
alt = {}
for l in open(f"{O}/db/known.meme"):
    if l.startswith("MOTIF "): x = l.split(); alt[x[1]] = x[2] if len(x) > 2 else x[1]
def toks(s): return set(re.split(r"[.@:|/_\- ]+", str(s).upper()))
comp, gap, summ = [], [], []
for pair in tasks:
    R = f"{G}/results/{pair}"; a, b = pair.split("-"); info = P.set_index("pair").loc[pair].to_dict(); info["pair"] = pair
    ms = []
    for f in sorted(glob.glob(f"{R}/dn_L*/homerMotifs.all.motifs")):
        for blk in open(f).read().split(">")[1:]:
            h = blk.split("\n")[0].split("\t"); st = h[5]
            t = float(re.search(r"T:[\d.]+\(([\d.]+)%", st).group(1)); bb = float(re.search(r"B:[\d.]+\(([\d.]+)%", st).group(1))
            ms.append(dict(cons=h[0], name=h[1], log10p=float(h[3]) / math.log(10), fold=(t + .01) / (bb + .01), raw=">" + blk.strip()))
    vf = f"{R}/denovo_val/knownResults.txt"; val = {}
    if os.path.exists(vf):
        k = pd.read_csv(vf, sep="\t"); k.columns = ["name", "cons", "p", "logp", "q", "nt", "pt", "nb", "pb"]
        for r in k.itertuples(): val[r.name.split("/")[0]] = (r.logp / math.log(10), float(str(r.pt).rstrip("%")) / max(float(str(r.pb).rstrip("%")), .01))
    keep = [m for m in ms if m["log10p"] <= -5 and m["fold"] >= 1.5]
    for m in keep:
        v = val.get(m["name"].split("/")[0]) or val.get(m["name"]); m["val_log10p"], m["val_fold"] = (v if v else (np.nan, np.nan))
    keep = [m for m in keep if np.isfinite(m["val_log10p"]) and m["val_log10p"] <= math.log10(.05) and m["val_fold"] > 1]
    if keep:
        mf = f"{R}/val_ok.meme"; mm = "MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\nBackground letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n"
        for i, m in enumerate(keep):
            M = np.array([[float(x) for x in l.split()] for l in m["raw"].split("\n")[1:] if l.strip()]); mm += f"MOTIF d{i}\nletter-probability matrix: alength= 4 w= {len(M)}\n" + "\n".join(" ".join(f"{x:.4f}" for x in r) for r in M) + "\n\n"
        open(mf, "w").write(mm)
        tt = subprocess.run(f"module load meme/5.5.7 >/dev/null 2>&1; tomtom -no-ssc -verbosity 1 -text -min-overlap 5 -dist pearson -thresh 0.05 {mf} {O}/db/known.meme", shell=True, capture_output=True, text=True, executable="/bin/bash").stdout
        T = pd.DataFrame([l.split("\t") for l in tt.splitlines() if l and not l.startswith(("Query", "#"))])
        best = {}
        if len(T): T.columns = list(range(T.shape[1])); T[5] = T[5].astype(float); best = T.sort_values(5).drop_duplicates(0).set_index(0)[1].to_dict()
        for i, m in enumerate(keep):
            bm = best.get(f"d{i}"); own = bm is not None and bool((toks(bm) | toks(alt.get(bm, ""))) & {a, b})
            comp.append(dict(pair=pair, set=info["set"], af3=info["af3"], consensus=m["cons"], disc_log10p=round(m["log10p"], 1), disc_fold=round(m["fold"], 2), val_log10p=round(m["val_log10p"], 2), val_fold=round(m["val_fold"], 2),
                             db_match=alt.get(bm, bm) if bm else "none (novel)", matches_A_or_B_own=own, composite_candidate=not own))
    for f in glob.glob(f"{R}/spamo_*_cob.tsv"):
        t, pt = os.path.basename(f).split("_")[1:3]
        try: d = pd.read_csv(f, sep="\t", comment="#")
        except Exception: continue
        if not len(d) or "adj_p-value" not in d.columns and "adj_pvalue" not in d.columns: continue
        pc = "adj_p-value" if "adj_p-value" in d.columns else "adj_pvalue"; d = d[d[pc] <= 0.05]
        ctl = f.replace("_cob.tsv", "_aonly.tsv"); dc = pd.DataFrame()
        try: dc = pd.read_csv(ctl, sep="\t", comment="#"); dc = dc[dc[pc] <= 0.05]
        except Exception: pass
        for r in d.itertuples():
            sec = getattr(r, "secondary_id", None) or getattr(r, "secondary_motif", None); gp = getattr(r, "gap", None); ori = getattr(r, "orientation", None)
            inctl = len(dc) and ((dc.get("secondary_id", pd.Series()) == sec) & (dc.get("gap", pd.Series()) == gp)).any()
            gap.append(dict(pair=pair, set=info["set"], af3=info["af3"], primary_tf=t, part=pt, secondary=alt.get(sec, sec), gap=gp, orientation=ori, adj_p=getattr(r, pc.replace("-", "_"), None), in_aonly_control=bool(inctl)))
    summ.append(info)
C = pd.DataFrame(comp); Gp = pd.DataFrame(gap); S = pd.DataFrame(summ)
if len(Gp):
    k = Gp[~Gp.in_aonly_control].groupby(["pair", "primary_tf", "secondary", "gap", "orientation"]).part.nunique().reset_index(); gboth = k[k.part == 2]
else: gboth = pd.DataFrame(columns=["pair"])
S["n_composite"] = S.pair.map(C[C.composite_candidate].groupby("pair").size() if len(C) else {}).fillna(0).astype(int) if len(C) else 0
S["n_composite_novel"] = S.pair.map(C[C.composite_candidate & (C.db_match == "none (novel)")].groupby("pair").size() if len(C) else {}).fillna(0).astype(int) if len(C) else 0
S["n_gapped_both_halves"] = S.pair.map(gboth.groupby("pair").size() if len(gboth) else {}).fillna(0).astype(int)
C.to_csv(f"{G}/composite_hits.tsv", sep="\t", index=False); Gp.to_csv(f"{G}/gapped_hits.tsv", sep="\t", index=False); gboth.to_csv(f"{G}/gapped_both_halves.tsv", sep="\t", index=False); S.to_csv(f"{G}/pair_summary.tsv", sep="\t", index=False)
print("pairs tested:", len(S), S.set.value_counts().to_dict())
for st, s in S.groupby("set"): print(f"  {st:14s} n={len(s):3d}  with composite {int((s.n_composite > 0).sum())} ({100 * (s.n_composite > 0).mean():.0f}%)  novel {int((s.n_composite_novel > 0).sum())}  with gapped (both halves, not in A-only) {int((s.n_gapped_both_halves > 0).sum())} ({100 * (s.n_gapped_both_halves > 0).mean():.0f}%)")
pd.set_option("display.width", 250); print(S.sort_values(["n_composite", "n_gapped_both_halves"], ascending=False).head(25)[["pair", "set", "af3", "disc_cob", "val_cob", "n_composite", "n_composite_novel", "n_gapped_both_halves"]].to_string(index=False))
if len(C): print(C[C.composite_candidate].sort_values("val_log10p").head(25).to_string(index=False))
if len(gboth): print(gboth.head(25).to_string(index=False))
