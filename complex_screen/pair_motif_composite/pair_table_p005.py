# Pair table, shared enriched motifs at p <= 0.05 (user rule 2026-11-12). A motif is shared by A-B when it reaches p <= 0.05 in BOTH TFs on BOTH
# chromosome halves (odd = discovery, even + X = held-out). Sources: HOMER known (472 motifs) and the 2,528-motif DB (FIMO presence, one-sided p from
# the enrichment z), both on lab peaks (top 1,000 summits in open chromatin, HOT removed, matched background). Flags per motif: generic (passes the
# same rule in > 10 of 79 TFs), Tn5 background (p <= 0.05 in >= 5 of 10 pooled-Tn5 control libraries, either half), Codebook artifact/endogenous.
# Pairs: all 3,081 panel pairs with candidate / known / AF3 columns. Output motif_tables/pair_shared_motifs_p005.tsv (+ xlsx on the Mac)
import glob, re
import numpy as np, pandas as pd
from scipy.stats import norm
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; T = f"{O}/motif_tables"; L = np.log10(0.05)
tfs = list(pd.read_csv(f"{X}/complex_screen/pair_motif_lookup/tf_cells.tsv", sep="\t").tf)
H = {}
for f in glob.glob(f"{O}/recall/homer/raw_lab/*/*/knownResults.txt"):
    t, pt = f.split("/")[-3:-1]; k = pd.read_csv(f, sep="\t"); _s = pd.Series((k.iloc[:, 3] / np.log(10)).values, index=k.iloc[:, 0].str.split("/").str[0]); H[(t, pt)] = _s[~_s.index.duplicated()]  # HOMER library repeats some names
hp = {t: set(H[(t, "disc")][H[(t, "disc")] <= L].index) & set(H[(t, "val")][H[(t, "val")] <= L].index) for t in tfs if (t, "disc") in H and (t, "val") in H}
Fd = pd.read_csv(f"{O}/motif_valid/scores_fisher_disc.tsv", sep="\t", index_col=0); Fv = pd.read_csv(f"{O}/motif_valid/scores_fisher_val.tsv", sep="\t", index_col=0); zc = norm.isf(0.05)
dp = {t: set(Fd.index[(Fd[t] >= zc) & (Fv[t] >= zc)]) for t in Fd.columns}
alt = {}
for l in open(f"{O}/db/known.meme"):
    if l.startswith("MOTIF "): x = l.split(); alt[x[1]] = x[2] if len(x) > 2 else x[1]
ncnt = {}
for f in glob.glob(f"{T}/nc/pooled_*_*.tsv"):
    k = pd.read_csv(f, sep="\t"); s = pd.Series((k.iloc[:, 3] / np.log(10)).values, index=k.iloc[:, 0].str.split("/").str[0])
    for m in s.index[s <= L]: ncnt[m] = ncnt.get(m, 0) + 1
tn5 = {m for m, c in ncnt.items() if c >= 5}
hgen = pd.Series([m for t in hp for m in hp[t]]).value_counts(); dgen = pd.Series([m for t in dp for m in dp[t]]).value_counts()
cbl = set(l.split()[2].upper() if len(l.split()) > 2 else "" for l in open(f"{X}/motif_analysis/cache/motifdb/codebook_artifacts.meme") if l.startswith("MOTIF "))
A = pd.read_csv(f"{X}/complex_screen/candidates_frozen_v1/candidates_all3081.tsv", sep="\t"); A["af3"] = [b if b != "not_run" else a for a, b in zip(A.af3_1seed_tier, A.af3_5seed_tier)]
def fmt(ms, gen, src, a, b):
    out = []
    key = (lambda m: float(H[(a, "disc")].get(m, 0)) + float(H[(b, "disc")].get(m, 0))) if src == "H" else (lambda m: -float(Fd.at[m, a] + Fd.at[m, b]))
    for m in sorted(ms, key=key):
        tags = []
        if gen.get(m, 0) > 10: tags.append(f"generic {gen[m]} TFs")
        if src == "H" and m in tn5: tags.append("Tn5 background")
        nm = m if src == "H" else f"{alt.get(m, m)}"
        out.append(nm + (f" [{', '.join(tags)}]" if tags else ""))
    return out
rows = []
for r in A.itertuples():
    a, b = r.tf_a, r.tf_b; sh = hp.get(a, set()) & hp.get(b, set()); sd = dp.get(a, set()) & dp.get(b, set())
    hs = fmt(sh, hgen, "H", a, b); ds = fmt(sd, dgen, "D", a, b)
    clean_h = [x for x in hs if "[" not in x]; clean_d = [x for x in ds if "[" not in x]
    rows.append(dict(pair=r.pair, tf_a=a, tf_b=b, correlation_pct=r.best_pct, known=r.known, known_reason=r.known_reason if isinstance(r.known_reason, str) else "", candidate=r.candidate, af3=r.af3,
                     n_shared_homer=len(hs), n_shared_homer_specific=len(clean_h), n_shared_db=len(ds), n_shared_db_specific=len(clean_d),
                     shared_homer_specific="; ".join(clean_h), shared_db_specific="; ".join(clean_d[:15]), shared_homer_all="; ".join(hs), shared_db_all="; ".join(ds[:30])))
P = pd.DataFrame(rows); P["any_specific_shared"] = (P.n_shared_homer_specific + P.n_shared_db_specific) > 0
P = P.sort_values(["candidate", "any_specific_shared", "n_shared_homer_specific", "n_shared_db_specific", "correlation_pct"], ascending=[False, False, False, False, False]); P.to_csv(f"{T}/pair_shared_motifs_p005.tsv", sep="\t", index=False)
c = P[P.candidate.astype(str) == "True"]; k = P[(P.known.astype(str) == "True") & ~P.known_reason.isin(["NEUROD4", "STAG2_class"])]
for lab, s in [("frozen candidates", c), ("AF3 strong/weak candidates", c[c.af3.isin(["strong", "weak"])]), ("known pairs", k), ("all 3,081 pairs", P)]:
    print(f"{lab}: n={len(s)}; any shared (incl. generic/Tn5) {int(((s.n_shared_homer + s.n_shared_db) > 0).sum())}; specific shared {int(s.any_specific_shared.sum())} ({100 * s.any_specific_shared.mean():.1f}%)")
print("Tn5-background motifs:", sorted(tn5)[:20]); print("TFs with >=1 HOMER motif at p<=0.05 both halves:", sum(1 for t in hp if hp[t]), "| DB:", sum(1 for t in dp if dp[t]))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
print(c[c.af3.isin(["strong", "weak"]) & c.any_specific_shared][["pair", "correlation_pct", "af3", "shared_homer_specific", "shared_db_specific"]].head(30).to_string(index=False))
print(k[["pair", "correlation_pct", "n_shared_homer", "n_shared_homer_specific", "n_shared_db_specific", "shared_homer_specific", "shared_db_specific"]].head(40).to_string(index=False))
