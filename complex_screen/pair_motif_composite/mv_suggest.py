# Suggestive shared motifs for the frozen candidates (exploratory, NOT significance-tested; Yuyang 2026-11-11: worth reporting as hypotheses for
# experimental follow-up, like CAP-SELEX suggestions). Uses motif_valid panel-standardized scores (depth trend removed, shrunk, vs the 79 TFs).
# A motif is a suggestive shared motif for pair A-B when: panel z >= 2 in both TFs on discovery (odd chr) AND panel z > 0 in both on held-out
# (even + X); not generic (fisher z >= 3 in > 10 of 79 TFs); not on the Codebook artifact list (37 motifs incl. endogenous NFI / SP-KLF / ETS / NR);
# CpG-dense consensus (>= 2 CpG) flagged. Pair score = best min(zA, zB) on discovery; context = its percentile among all 3,081 panel pairs.
# Output motif_valid/suggestive_shared_motifs.tsv (pair x motif rows), motif_valid/suggestive_pairs.tsv (one row per candidate)
import re
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; G = f"{O}/motif_valid"
Zd = pd.read_csv(f"{G}/scores_panel_disc.tsv", sep="\t", index_col=0); Zv = pd.read_csv(f"{G}/scores_panel_val.tsv", sep="\t", index_col=0); Fd = pd.read_csv(f"{G}/scores_fisher_disc.tsv", sep="\t", index_col=0)
alt, cons, cur, rows = {}, {}, None, []
for l in open(f"{O}/db/known.meme"):
    if l.startswith("MOTIF "):
        if cur: cons[cur] = "".join("ACGT"[int(np.argmax(r))] for r in rows)
        x = l.split(); cur = x[1]; alt[cur] = x[2] if len(x) > 2 else x[1]; rows = []
    elif cur and len(l.split()) == 4 and l.strip()[0] in "0123456789.":
        rows.append([float(v) for v in l.split()])
if cur: cons[cur] = "".join("ACGT"[int(np.argmax(r))] for r in rows)
cbl = set()
for l in open(f"{X}/motif_analysis/cache/motifdb/codebook_artifacts.meme"):
    if l.startswith("MOTIF "): cbl.add(l.split()[1])
# Codebook artifact motifs are not in known.meme by ID; flag DB motifs that Tomtom-match them would need a scan: approximate by consensus k-mer overlap
cb_cons = set()
cur, rows = None, []
for l in list(open(f"{X}/motif_analysis/cache/motifdb/codebook_artifacts.meme")) + ["MOTIF end"]:
    if l.startswith("MOTIF "):
        if cur and rows: cb_cons.add("".join("ACGT"[int(np.argmax(r))] for r in rows))
        cur = l.split()[1]; rows = []
    elif cur and len(l.split()) == 4 and l.strip()[0] in "0123456789.": rows.append([float(v) for v in l.split()])
def rc(s): return s[::-1].translate(str.maketrans("ACGT", "TGCA"))
def cb_like(c): return any(len(k) >= 6 and (k[:6] in c or k[:6] in rc(c)) for k in cb_cons)
generic = (Fd >= 3).sum(1) > 10; tfs = list(Zd.columns)
FZ = pd.read_csv(f"{X}/complex_screen/candidates_frozen_v1/candidates_frozen_v1.tsv", sep="\t"); FZ = FZ[FZ.candidate.astype(str) == "True"]
allp = [(a, b) for i, a in enumerate(tfs) for b in tfs[i + 1:]]; ok = ~generic
best_all = np.array([np.nanmax(np.fmin(Zd[a].values, Zd[b].values)[ok.values]) for a, b in allp])
R, PR = [], []
for r in FZ.itertuples():
    a, b = r.tf_a, r.tf_b
    if a not in tfs or b not in tfs: continue
    md, mv = np.fmin(Zd[a], Zd[b]), np.fmin(Zv[a], Zv[b]); sel = (md >= 2) & (mv > 0) & ok
    af3 = r.af3_5seed_tier if r.af3_5seed_tier != "not_run" else r.af3_1seed_tier; pb = float(np.nanmax(md[ok]))
    for m in md[sel].sort_values(ascending=False).index[:10]:
        c = cons[m]; R.append(dict(pair=r.pair, best_pct=r.best_pct, af3=af3, motif=m, motif_name=alt[m], consensus=c, z_disc_A=round(Zd.at[m, a], 2), z_disc_B=round(Zd.at[m, b], 2), z_val_A=round(Zv.at[m, a], 2), z_val_B=round(Zv.at[m, b], 2),
                                   n_tf_fisher_z3=int((Fd.loc[m] >= 3).sum()), cpg_dense=c.count("CG") >= 2, codebook_like=cb_like(c)))
    s = [x for x in R if x["pair"] == r.pair]; clean = [x for x in s if not x["cpg_dense"] and not x["codebook_like"]]
    PR.append(dict(pair=r.pair, best_pct=r.best_pct, af3=af3, n_suggestive=len(s), n_clean=len(clean), best_min_z_disc=round(pb, 2), panel_pct=round(float(np.mean(best_all < pb)), 3),
                   top=("; ".join(f"{x['motif_name']} ({x['consensus']}) disc {min(x['z_disc_A'], x['z_disc_B'])}/val {min(x['z_val_A'], x['z_val_B'])}" for x in clean[:3])) or ""))
T = pd.DataFrame(R); T.to_csv(f"{G}/suggestive_shared_motifs.tsv", sep="\t", index=False)
Pp = pd.DataFrame(PR).sort_values(["n_clean", "best_min_z_disc"], ascending=False); Pp.to_csv(f"{G}/suggestive_pairs.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 160)
print(len(Pp), "candidates;", int((Pp.n_suggestive > 0).sum()), "with >= 1 suggestive shared motif;", int((Pp.n_clean > 0).sum()), "after CpG / Codebook-like flags")
print("AF3 strong/weak with clean suggestive motif:"); print(Pp[Pp.af3.isin(["strong", "weak"]) & (Pp.n_clean > 0)].head(25).to_string(index=False))
print("top overall:"); print(Pp[Pp.n_clean > 0].head(25).to_string(index=False))
