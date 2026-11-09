# For T1/T2/T3 pairs of triple_support.tsv (+ calibration pairs): is the DNA-binding partner's own known motif enriched in the OTHER partner's
# existing motif results, and how specifically? Existing HOMER-known / AME outputs only (read through build_pair_motif_lookup.py's readers).
# Per direction D -> L and per set: best own motif of D in L's table (p), rank of L among TFs in that set (1 = smallest p; absent = p 1),
# whether L passes, and how many TFs pass that motif. specific = L passes AND rank <= 10 AND n_pass <= 10.
# Writes partner_motif_rank.tsv and partner_motif_rank_summary.tsv. usage: partner_motif_rank.py
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.dirname(HERE); LK = f"{C}/pair_motif_lookup"
assert not os.path.exists(f"{HERE}/partner_motif_rank.tsv"), "output exists"
src = open(f"{LK}/build_pair_motif_lookup.py").read(); src = src[:src.index("# ---------------- de novo motifs")]
sys.argv = [f"{LK}/build_pair_motif_lookup.py", HERE]; g = {"__file__": f"{LK}/build_pair_motif_lookup.py"}; exec(src, g)
KNOWN, canon, tokens, cells = g["KNOWN"], g["canon"], g["tokens"], g["cells"]
ALIAS = dict(g["ALIAS"]); ALIAS.update({"DDIT3": {"DDIT3", "CHOP"}, "EPAS1": {"EPAS1", "HIF2A", "HIF2"}, "ZUFSP": {"ZUFSP", "ZUP1"}, "MLX": {"MLX"}, "E4F1": {"E4F1"}})
is_own = lambda tf, m: bool(ALIAS.get(tf, {tf}) & tokens(m))
T = pd.read_csv(f"{HERE}/triple_support.tsv", sep="\t", keep_default_na=False)
dna = {}
for x in ("a", "b"):
    for p, d in zip(T.pair, T[f"{x}_dna_motif"]): dna[p.split("-")[0 if x == "a" else 1]] = d == "True" or d is True
P = [(r.pair, r.tier) for r in T[T.tier.isin(["T1", "T2", "T3"])].itertuples()] + [("CTCF-STAG2", "calib_pos"), ("EZH2-MTF2", "calib_pos"), ("MAZ-ZNF143", "calib_pos"), ("RBBP7-SRCAP", "calib_neg")]
SETS = ["A_raw_homer", "A_raw_ame", "B_imp_peaks_homer", "B_imp_peaks_homer_ocr", "B_imp_peaks_ame", "B_imp_peaks_ame_ocr", "C_imp_bins_homer", "C_imp_bins_ame"]
# per set: motif -> {tf: (p, pass)}
TAB = {}
for s in SETS:
    d = {}
    for tf, df in KNOWN[s].items():
        for m, p, ok in zip(df.motif, df.p, df["pass"]):
            if not isinstance(m, str): continue
            q = d.setdefault(m, {}); 
            if tf not in q or p < q[tf][0]: q[tf] = (float(p), bool(ok))
    TAB[s] = (d, sorted(KNOWN[s]))
rows = []
for pair, tier in P:
    a, b = [canon(t) for t in pair.split("-")]
    for D, L in ((a, b), (b, a)):
        if not dna.get(D, D in ("CTCF", "MAZ", "ZNF143")) : continue
        for s in SETS:
            d, tfs = TAB[s]; own = [m for m in d if is_own(D, m)]
            base = dict(pair=pair, tier=tier, motif_tf=D, target_tf=L, set=s, raw=s.startswith("A_raw"), n_tfs_in_set=len(tfs), target_in_set=L in tfs)
            if not own or L not in tfs: rows.append(dict(base, motif="none" if not own else ".", p=np.nan, rank=np.nan, target_pass=False, n_pass=np.nan, specific=False)); continue
            best = None
            for m in own:
                pv = {t: d[m].get(t, (1.0, False)) for t in tfs}; pL = pv[L][0]; rk = 1 + sum(v[0] < pL for v in pv.values()); npass = sum(v[1] for v in pv.values())
                c = dict(base, motif=m, p=pL, rank=rk, target_pass=pv[L][1], n_pass=npass, specific=bool(pv[L][1] and rk <= 10 and npass <= 10))
                if best is None or (c["specific"], c["target_pass"], -c["rank"], -c["p"]) > (best["specific"], best["target_pass"], -best["rank"], -best["p"]): best = c
            rows.append(best)
R = pd.DataFrame(rows); R.to_csv(f"{HERE}/partner_motif_rank.tsv", sep="\t", index=False, float_format="%.3g")
S = []
for (pair, tier), G in R.groupby(["pair", "tier"], sort=False):
    tested = G[G.motif.ne("none") & G.target_in_set]
    S.append(dict(pair=pair, tier=tier, directions=" ; ".join(sorted({f"{x}->{y}" for x, y in zip(G.motif_tf, G.target_tf)})) or ".", n_sets_tested=len(tested),
                  n_target_pass=int(tested.target_pass.sum()), best_rank=tested["rank"].min() if len(tested) else np.nan, min_n_pass=tested.n_pass.min() if len(tested) else np.nan,
                  specific_raw=bool(G[G.raw].specific.any()), specific_imputed=bool(G[~G.raw].specific.any()),
                  specific_sets=",".join(G[G.specific].set) or ".", note="no own known motif in HOMER/AME libraries" if len(tested) == 0 else "."))
for pair, tier in P:
    if pair not in {x["pair"] for x in S}: S.append(dict(pair=pair, tier=tier, directions=".", n_sets_tested=0, n_target_pass=0, best_rank=np.nan, min_n_pass=np.nan, specific_raw=False, specific_imputed=False, specific_sets=".", note="neither partner has a DNA motif"))
S = pd.DataFrame(S); S.to_csv(f"{HERE}/partner_motif_rank_summary.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 60)
print(S.to_string(index=False)); print(S.assign(any_spec=S.specific_raw | S.specific_imputed).groupby("tier").agg(n=("pair", "size"), tested=("n_sets_tested", lambda x: (x > 0).sum()), specific_raw=("specific_raw", "sum"), specific_imp=("specific_imputed", "sum"), specific_any=("any_spec", "sum")).to_string())
print(R[R.tier.str.startswith("calib")].to_string(index=False))
