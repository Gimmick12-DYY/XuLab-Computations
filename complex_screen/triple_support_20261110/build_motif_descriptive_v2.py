# Descriptive motif columns for the deck pair table, v2 (v1 motif_descriptive.tsv kept). One row per pair in interface_audit_20261110/triple_support_flagged_v2.tsv;
# motif values from pair_motif_composite/profile/novel_pairs_motif_profile.tsv, falling back to all_pairs_motif_profile.tsv (both commit a987c98), joined on the sorted
# pair name; source_table records which. matched_pct_* = conc_<method>_matched_pct (see README). Does not affect ranking. Writes motif_descriptive_v2.tsv; prints word counts.
import collections, pandas as pd
C = "/work/users/d/y/dyy12/XuLab/complex_screen"; O = f"{C}/triple_support_20261110"; G = f"{C}/pair_motif_composite/profile"; M = ("peak500", "peak2000", "seqN20", "seqN3k")
key = lambda p: "-".join(sorted(p.split("-")))
T = pd.read_csv(f"{C}/interface_audit_20261110/triple_support_flagged_v2.tsv", sep="\t", keep_default_na=False)
def load(f):
    P = pd.read_csv(f"{G}/{f}", sep="\t", keep_default_na=False); P["k"] = P.pair.map(key); assert P.k.is_unique; return P.set_index("k")
N, A = load("novel_pairs_motif_profile.tsv"), load("all_pairs_motif_profile.tsv")
def top3(s):
    if not s or s == "NA": return []
    return [x.split(" [")[0].split("|")[-1].strip() for x in s.split("]; ")[:3]]
rows = []; W = collections.Counter(); WM = {m: collections.Counter() for m in M}; Z = []; miss = []
for r in T.itertuples():
    k = key(r.pair); src = "novel_pairs_motif_profile.tsv" if k in N.index else ("all_pairs_motif_profile.tsv" if k in A.index else "NA"); P = N if k in N.index else A
    if src == "NA": miss.append(r.pair)
    x = dict(pair=r.pair, af3_tier=r.af3_tier, n_mat_ge080=r.n_mat_ge080, boltz_trimmed=r.boltz_trimmed, boltz_fulllength=r.boltz_fulllength, source_table=src); z = False
    for m in M:
        t = top3(P.loc[k, f"top10_{m}"]) if src != "NA" else []
        x[f"top3_motifs_{m}"] = "; ".join(t) if t else "NA"; x[f"matched_pct_{m}"] = P.loc[k, f"conc_{m}_matched_pct"] if src != "NA" else "NA"
        W.update(t); WM[m].update(t); z |= "ZNF460" in t
    x["motif_status"] = "descriptive only; not detected in our data"; rows.append(x); Z += [r.pair] if z else []
R = pd.DataFrame(rows); R.to_csv(f"{O}/motif_descriptive_v2.tsv", sep="\t", index=False)
print(len(R), "rows; unresolved", miss, "| source", R.source_table.value_counts().to_dict())
print("ZNF460 in any top 3:", len(Z), "of", len(R)); print("total top-3 slots", sum(W.values()), "| top 5 overall:", W.most_common(5))
for m in M: print(m, WM[m].most_common(5))
