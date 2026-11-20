# Pair shared-motif table for candidates_frozen_v2 (cutoff .75, 1,198 candidates; 2026-11-20).
# Same rule and per-TF inputs as pair_table_p005.py (2026-11-12): a motif is shared by A-B when p <= 0.05 in BOTH TFs on BOTH chromosome halves
# (HOMER known 472 + 2,528-motif DB). Flags: generic = passes in > 10 of 79 TFs; Tn5 background = HOMER motif at p <= 0.05 in >= 5 of 10 pooled-Tn5
# control libraries; specific = neither. Codebook artifacts: motifs that Tomtom-match one of the 37 Codebook artifact PWMs
# (funnel_20261120/tomtom_*.tsv, job 9971617: -dist pearson -min-overlap 5 -thresh 0.05) are dropped from the *_noCB columns (main).
# The per-TF enrichment does not depend on the candidate set (per-TF HOMER/FIMO results, generic count over all TFs, Tn5 controls), so this only
# recomputes the pair table for the new list. Motif lists are complete (no truncation). Asserts the v1 counts are reproduced for all 3,081 pairs.
# Writes motif_tables/pair_shared_motifs_v2_all3081.tsv, pair_shared_motifs_v2_candidates1198.tsv, pair_shared_motifs_v2_AF3strong.tsv
import glob
import numpy as np, pandas as pd
from scipy.stats import norm
X = "/work/users/d/y/dyy12/XuLab"; C = f"{X}/complex_screen"; O = f"{C}/pair_motif_composite"; T = f"{O}/motif_tables"; L = np.log10(0.05); FU = f"{C}/funnel_20261120"
tfs = list(pd.read_csv(f"{C}/pair_motif_lookup/tf_cells.tsv", sep="\t").tf)
H = {}
for f in glob.glob(f"{O}/recall/homer/raw_lab/*/*/knownResults.txt"):
    t, pt = f.split("/")[-3:-1]; k = pd.read_csv(f, sep="\t"); _s = pd.Series((k.iloc[:, 3] / np.log(10)).values, index=k.iloc[:, 0].str.split("/").str[0]); H[(t, pt)] = _s[~_s.index.duplicated()]
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
artH = set(pd.read_csv(f"{FU}/tomtom_homer_known.tsv", sep="\t").iloc[:, 0].str.replace(r"__\d+$", "", regex=True))
artD = set(pd.read_csv(f"{FU}/tomtom_known.tsv", sep="\t").iloc[:, 0])
A = pd.read_csv(f"{C}/candidates_frozen_v2/candidates_all3081.tsv", sep="\t")
def lab(ms, gen, src, a, b):
    out = []
    key = (lambda m: float(H[(a, "disc")].get(m, 0)) + float(H[(b, "disc")].get(m, 0))) if src == "H" else (lambda m: -float(Fd.at[m, a] + Fd.at[m, b]))
    for m in sorted(ms, key=key):
        tags = []
        if gen.get(m, 0) > 10: tags.append(f"generic {gen[m]} TFs")
        if src == "H" and m in tn5: tags.append("Tn5 background")
        art = m in (artH if src == "H" else artD)
        out.append((m if src == "H" else alt.get(m, m), tags, art))
    return out
j = lambda xs: "; ".join(xs)
rows = []
for r in A.itertuples():
    a, b = r.tf_a, r.tf_b
    hs = lab(hp.get(a, set()) & hp.get(b, set()), hgen, "H", a, b); ds = lab(dp.get(a, set()) & dp.get(b, set()), dgen, "D", a, b)
    hsp = [n for n, t, x in hs if not t]; dsp = [n for n, t, x in ds if not t]; hnc = [n for n, t, x in hs if not t and not x]; dnc = [n for n, t, x in ds if not t and not x]
    fl = lambda L_: [n + (f" [{', '.join(t + (['Codebook artifact match'] if x else []))}]" if (t or x) else "") for n, t, x in L_]
    rows.append(dict(pair=r.pair, tf_a=a, tf_b=b, correlation_pct=r.best_pct, best_matrix=r.best_matrix, candidate_v2=r.candidate, known=r.known,
                     known_reason=r.known_reason if isinstance(r.known_reason, str) else "", af3_5seed=r.af3_5seed_final, af3_1seed=r.af3_1seed_tier,
                     motif_specific_noCB=len(hnc) + len(dnc) > 0, top_specific_noCB=(hnc + dnc)[0] if hnc + dnc else "",
                     n_shared_homer_specific_noCB=len(hnc), shared_homer_specific_noCB=j(hnc), n_shared_db_specific_noCB=len(dnc), shared_db_specific_noCB=j(dnc),
                     motif_specific_inclCB=len(hsp) + len(dsp) > 0, n_shared_homer_specific=len(hsp), n_shared_db_specific=len(dsp),
                     n_shared_homer=len(hs), shared_homer_all=j(fl(hs)), n_shared_db=len(ds), shared_db_all=j(fl(ds))))
P = pd.DataFrame(rows)
V1 = pd.read_csv(f"{T}/pair_shared_motifs_p005.tsv", sep="\t").set_index("pair").loc[P.pair]
for c_ in ("n_shared_homer", "n_shared_homer_specific", "n_shared_db", "n_shared_db_specific"): assert (V1[c_].values == P[c_].values).all(), c_
F1 = pd.read_csv(f"{FU}/pair_funnel_20261120_triple.tsv", sep="\t")      # v1 funnel MAIN motif flag reproduced on its 55 strong/weak rows
assert P.set_index("pair").loc[F1.pair.str.replace("ZUP1", "ZUFSP"), "motif_specific_noCB"].all()
P = P.sort_values(["candidate_v2", "motif_specific_noCB", "n_shared_homer_specific_noCB", "n_shared_db_specific_noCB", "correlation_pct"], ascending=[False, False, False, False, False])
P.to_csv(f"{T}/pair_shared_motifs_v2_all3081.tsv", sep="\t", index=False)
c = P[P.candidate_v2.astype(str) == "True"]; assert len(c) == 1198
c.to_csv(f"{T}/pair_shared_motifs_v2_candidates1198.tsv", sep="\t", index=False); c[c.af3_5seed == "strong"].to_csv(f"{T}/pair_shared_motifs_v2_AF3strong.tsv", sep="\t", index=False)
print("max cell length:", int(P.astype(str).apply(lambda s: s.str.len()).max().max()))
S = []
for nm, s in (("v2 candidates (cutoff .75)", c), ("v2 candidates, AF3 5-seed strong", c[c.af3_5seed == "strong"]), ("v2 candidates, AF3 5-seed strong or weak", c[c.af3_5seed.isin(["strong", "weak"])]),
              ("all 3,081 panel pairs", P)):
    S.append(dict(set=nm, n=len(s), any_shared=int(((s.n_shared_homer + s.n_shared_db) > 0).sum()), specific_shared=int(s.motif_specific_inclCB.sum()),
                  specific_shared_noCodebook=int(s.motif_specific_noCB.sum()), pct_specific_noCodebook=round(100 * s.motif_specific_noCB.mean(), 1)))
S = pd.DataFrame(S); S.to_csv(f"{T}/pair_shared_motifs_v2_summary.tsv", sep="\t", index=False); print(S.to_string(index=False))
