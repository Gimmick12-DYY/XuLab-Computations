# Descriptive motif columns for the deck pair table. One row per pair in interface_audit_20261110/triple_support_flagged_v2.tsv, joined on the sorted pair
# name to pair_motif_composite/profile/novel_pairs_motif_profile.tsv (commit a987c98). Does not affect ranking. Writes motif_descriptive.tsv; prints missing pairs.
import pandas as pd
C = "/work/users/d/y/dyy12/XuLab/complex_screen"; O = f"{C}/triple_support_20261110"; M = ("peak500", "peak2000", "seqN20", "seqN3k")
key = lambda p: "-".join(sorted(p.split("-")))
T = pd.read_csv(f"{C}/interface_audit_20261110/triple_support_flagged_v2.tsv", sep="\t", keep_default_na=False)
P = pd.read_csv(f"{C}/pair_motif_composite/profile/novel_pairs_motif_profile.tsv", sep="\t", keep_default_na=False); P["k"] = P.pair.map(key); assert P.k.is_unique
P = P.set_index("k")
def top3(s):
    if not s or s == "NA": return "NA"
    return "; ".join(x.split(" [")[0].split("|")[-1].strip() for x in s.split("]; ")[:3])
rows, miss = [], []
for r in T.itertuples():
    k = key(r.pair); hit = k in P.index; miss += [] if hit else [r.pair]
    x = dict(pair=r.pair, af3_tier=r.af3_tier, n_mat_ge080=r.n_mat_ge080, boltz_trimmed=r.boltz_trimmed, boltz_fulllength=r.boltz_fulllength)
    for m in M:
        x[f"top3_motifs_{m}"] = top3(P.loc[k, f"top10_{m}"]) if hit else "NA"; x[f"matched_pct_{m}"] = P.loc[k, f"conc_{m}_matched_pct"] if hit else "NA"
    x["motif_status"] = "descriptive only; not detectable at this depth"; rows.append(x)
R = pd.DataFrame(rows); R.to_csv(f"{O}/motif_descriptive.tsv", sep="\t", index=False)
print(len(R), "rows; missing", len(miss), miss)
