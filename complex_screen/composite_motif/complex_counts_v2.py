# Count table v2 for Jie (draft files kept). (a) two recovery columns over each complex's direct or deck-core pairs: primary = mean pair percentile >= 0.90 on the per-domain
# enrichment matrix (compartment_rpkm_correlation.py --transform enrichment; percentile among all off-diagonal TF pairs); stricter = >= 1 R1 co-binding edge.
# Pair selection is curated by hand and listed in the table. (b) verified citations. (c) candidates pending AF3, with flags.
import numpy as np, pandas as pd
C = "/work/users/d/y/dyy12/XuLab/complex_screen"; O = f"{C}/complex_counts"; X = "/work/users/d/y/dyy12/XuLab"
key = lambda a, b: "__".join(sorted((a, b)))
P = pd.read_csv(f"{X}/regulatome/r1_tftf/v2/r1v2_pairs_provisional.tsv", sep="\t"); P["k"] = [key(a, b) for a, b in zip(P.a, P.b)]; P = P.set_index("k")
R = pd.read_csv(f"{X}/tf_complex/results_compartment_enrichment_domain_sm2/tf_similarity_genome.tsv", sep="\t", index_col=0); R.columns = R.columns.str.strip().str.upper(); R.index = R.index.str.strip().str.upper()
v = np.sort(R.values[~np.eye(len(R), dtype=bool)]); pct = lambda a, b: np.searchsorted(v, R.loc[a, b], side="right") / len(v)
K = [("PRC2", ["EZH2__MTF2"], ["EZH2__RBBP4", "EZH2__RBBP7", "MTF2__RBBP4", "MTF2__RBBP7", "RBBP4__RBBP7"], "deck pair; bridged by SUZ12, which is off-panel; RBBP4/7 contact SUZ12"),
     ("NuRD", ["MBD3__RBBP4", "MBD3__RBBP7"], ["RBBP4__RBBP7"], "deck core; bridged by MTA, off-panel; RBBP4/7 are alternate paralogs"),
     ("NuRD + ZNF512B", ["RBBP4__ZNF512B"], ["MBD3__ZNF512B", "RBBP7__ZNF512B"], "direct, NAR 2024 doi 10.1093/nar/gkae926"),
     ("NURF-type (SMARCA1-RBBP4)", ["RBBP4__SMARCA1"], ["RBBP7__SMARCA1"], "co-subunits; contact not confirmed"),
     ("PRC1.6 (PCGF6-HP1)", ["HP1__PCGF6"], ["PCGF6__RBBP4", "PCGF6__RBBP7", "EZH2__PCGF6"], "deck pair; co-subunits"),
     ("ChAHP (ADNP-HP1)", ["ADNP__HP1"], ["ADNP__ZNF644"], "direct, Ostapcuk 2018 doi 10.1038/s41586-018-0153-8"),
     ("CHAMP1-HP1", ["CHAMP1__HP1"], [], "split from ChAHP: CHAMP1 is not a ChAHP subunit; contact not confirmed"),
     ("G9a/GLP scaffold (WIZ-ZNF644)", ["WIZ__ZNF644"], ["HP1__WIZ", "HP1__ZNF644"], "deck pair; both bind G9a/GLP, off-panel"),
     ("DNMT1-HP1 / DNMT1-EZH2", ["DNMT1__HP1", "DNMT1__EZH2"], ["DNMT1__RBBP4", "DNMT1__RBBP7"], "reported interactions; from memory, verify"),
     ("MEF2 dimer", ["MEF2A__MEF2D"], [], "dimer"),
     ("RELA-ZBTB7A", ["RELA__ZBTB7A"], [], "STRING pair; contact not confirmed"),
     ("Cohesin-CTCF", ["CTCF__STAG2"], [], "deck pair; direct; our technique may harm CTCF function"),
     ("MAZ-ZNF143", ["MAZ__ZNF143"], [], "deck pair; not a confirmed direct contact"),
     ("HES1-RBPJ", [], ["HES1__RBPJ"], "no direct pair kept: STRING only; not scored"),
     ("HOX paralogs", [], ["HOXA5__HOXB6", "HOXA5__HOXC4", "HOXB6__HOXC4"], "no direct pair kept: paralogs, not a complex; not scored")]
rows = []
for n, keep, drop, note in K:
    keep = [key(*p.split("__")) for p in keep]; drop = [key(*p.split("__")) for p in drop]; m = np.mean([pct(*p.split("__")) for p in keep]) if keep else np.nan
    rows.append(dict(complex=n, pairs_used=",".join(keep) or ".", pairs_dropped=",".join(drop) or ".", mean_pair_pct_enrichment=round(m, 3) if keep else np.nan, recovered_deck_rule=bool(m >= 0.90) if keep else "not scored",
                     per_pair_pct=",".join(f"{pct(*p.split('__')):.2f}" for p in keep) or ".", n_edges_used=int(P.view1.reindex(keep).sum()) if keep else 0, recovered_ge1_edge=bool(P.view1.reindex(keep).sum() >= 1) if keep else "not scored",
                     n_edges_all_listed=int(P.view1.reindex(keep + drop).sum()), boltz_trimmed=",".join(P.boltz_tier.reindex(keep).astype(str)) or ".", note=note))
A = pd.DataFrame(rows); A.to_csv(f"{O}/a_known_complexes_v2.tsv", sep="\t", index=False)
pd.DataFrame([dict(pair="HP1__RBBP4", string_physical=P.string_physical["HP1__RBBP4"], status="unassigned known pair", view1_edge=bool(P.view1["HP1__RBBP4"]), percentile_enrichment=round(pct("HP1", "RBBP4"), 3))]).to_csv(f"{O}/a_unassigned_known_pairs_v2.tsv", sep="\t", index=False)
called = P[P.capselex.astype(str).str.startswith("called")]; rb = []
for k, r in called.iterrows(): rb.append(dict(pair=k, source="CAP-SELEX " + r.capselex, verified=True, citation="CAP-SELEX table (complex_screen/cap_selex)", k_edge=bool(r.view1), percentile_enrichment=round(pct(*k.split("__")), 3), edge_type=r.edge_type, boltz_trimmed=r.boltz_tier, curator_note=""))
for k, c in [("RBBP4__ZNF512B", "Nucleic Acids Res 2024, doi 10.1093/nar/gkae926; bioRxiv 2023.07.31.551264"), ("ADNP__HP1", "Ostapcuk et al., Nature 2018, doi 10.1038/s41586-018-0153-8")]:
    r = P.loc[k]; rb.append(dict(pair=k, source="literature", verified=True, citation=c, k_edge=bool(r.view1), percentile_enrichment=round(pct(*k.split("__")), 3), edge_type=r.edge_type, boltz_trimmed=r.boltz_tier, curator_note=""))
B = pd.DataFrame(rb).rename(columns={"k_edge": "cobinding_edge"}); B.to_csv(f"{O}/b_recently_revealed_v2.tsv", sep="\t", index=False)
Cc = pd.read_csv(f"{O}/c_novel.tsv", sep="\t", keep_default_na=False); Cc.insert(0, "category", "candidates (pending AF3)")
Cc["hub_flag"] = ["hub:" + ",".join(x for x in ("DDIT3", "ZUFSP") if x in p) if any(x in p for x in ("DDIT3", "ZUFSP")) else "." for p in Cc.pair]
Cc.to_csv(f"{O}/c_candidates_pending_AF3_v2.tsv", sep="\t", index=False)
sc = A[A.recovered_deck_rule != "not scored"]
S = pd.DataFrame([dict(category="(a) known complexes scored", n=len(sc), detail=f"deck rule recovered {int((sc.recovered_deck_rule == True).sum())}; >= 1 edge {int((sc.recovered_ge1_edge == True).sum())}; {len(A) - len(sc)} listed, not scored; 1 unassigned pair"),
                  dict(category="(b) recently revealed", n=len(B), detail=f"{len(called)} CAP-SELEX, 2 literature, verified; {int(B.cobinding_edge.sum())} with co-binding edge"),
                  dict(category="(c) candidates (pending AF3): complex", n=int((Cc.class_with_rulings == "complex").sum()), detail="DR1-THAP6"),
                  dict(category="(c) candidates (pending AF3): complex_weak", n=int((Cc.class_with_rulings == "complex_weak").sum()), detail=f"{int((Cc.flag == 'seed_unstable').sum())} seed_unstable; {int(Cc.hub_flag.str.startswith('hub').sum())} with DDIT3 or ZUFSP across (c)"),
                  dict(category="(c) demoted to co-localized", n=int((Cc.class_with_rulings == "co-localized").sum()), detail="trimmed_strong_only")])
S.to_csv(f"{O}/summary_v2.tsv", sep="\t", index=False); pd.set_option("display.width", 240); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 60)
print(S.to_string(index=False)); print(A.drop(columns=["pairs_dropped", "note"]).to_string(index=False)); print(B.drop(columns=["citation", "curator_note"]).to_string(index=False))
print(pd.read_csv(f"{O}/a_unassigned_known_pairs_v2.tsv", sep="\t").to_string(index=False))
