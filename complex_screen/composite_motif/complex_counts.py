# Draft count table for Jie from the provisional R1 v2 table. (a) known complexes, STRING physical >= 700 pairs plus the deck's known pairs, collapsed by hand into complexes;
# recovered = at least half of the complex's pairs are co-binding (view 1) edges. (b) recently revealed: CAP-SELEX called panel pairs plus literature candidates to curate.
# (c) novel: not STRING, not CAP-SELEX, co-binding support plus a Boltz structure call, with the retest rulings applied. AF3 columns read af3_scores.tsv, else pending.
import os
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; C = f"{X}/complex_screen"; O = f"{C}/complex_counts"; os.makedirs(O, exist_ok=True)
P = pd.read_csv(f"{X}/regulatome/r1_tftf/v2/r1v2_pairs_provisional.tsv", sep="\t")
key = lambda a, b: "__".join(sorted((a, b))); P["k"] = [key(a, b) for a, b in zip(P.a, P.b)]; P = P.set_index("k")
FL = pd.concat([pd.read_csv(f"{C}/fulllength_check/fulllength_scores.tsv", sep="\t"), pd.read_csv(f"{C}/fulllength_check/retest6/fulllength_scores.tsv", sep="\t")])
HPK = lambda s: s.replace("CBX1", "HP1"); FL["k"] = [key(*HPK(p).split("__")) for p in FL.pair]; FL = FL.set_index("k")
AF = None
if os.path.exists(f"{C}/af3_local/af3_scores.tsv"):
    AF = pd.read_csv(f"{C}/af3_local/af3_scores.tsv", sep="\t"); AF["k"] = [key(*HPK(p).replace("ZUP1", "ZUFSP").split("__")) for p in AF.pair]; AF = AF.set_index("k")
def af3(k):
    if AF is None or k not in AF.index: return ("pending", "pending", "pending")
    return (AF.tier[k], f"{AF.iptm_top[k]:.2f}", f"{AF.ipsae_max[k]:.2f}")
KNOWN = {"PRC2": ["EZH2__MTF2", "EZH2__RBBP4", "EZH2__RBBP7", "MTF2__RBBP4", "MTF2__RBBP7", "RBBP4__RBBP7"], "NuRD": ["MBD3__RBBP4", "MBD3__RBBP7", "RBBP4__RBBP7"],
         "NuRD + ZNF512B": ["MBD3__ZNF512B", "RBBP4__ZNF512B", "RBBP7__ZNF512B"], "NURF-type (SMARCA1-RBBP)": ["RBBP4__SMARCA1", "RBBP7__SMARCA1"],
         "PRC1.6 (PCGF6)": ["HP1__PCGF6", "PCGF6__RBBP4", "PCGF6__RBBP7", "EZH2__PCGF6"], "ChAHP / HP1-ADNP-CHAMP1": ["ADNP__HP1", "CHAMP1__HP1", "ADNP__ZNF644"],
         "G9a/GLP scaffold (WIZ-ZNF644-HP1)": ["WIZ__ZNF644", "HP1__WIZ", "HP1__ZNF644"], "DNMT1 complexes": ["DNMT1__EZH2", "DNMT1__HP1", "DNMT1__RBBP4", "DNMT1__RBBP7"],
         "MEF2 dimer": ["MEF2A__MEF2D"], "Notch (HES1-RBPJ)": ["HES1__RBPJ"], "RELA-ZBTB7A": ["RELA__ZBTB7A"], "HOX paralogs": ["HOXA5__HOXB6", "HOXA5__HOXC4", "HOXB6__HOXC4"],
         "Cohesin-CTCF (deck, not STRING >= 700)": ["CTCF__STAG2"], "MAZ-ZNF143 (deck, not STRING >= 700)": ["MAZ__ZNF143"]}
rows = []; seen = set()
for n, ps in KNOWN.items():
    ps = [key(*p.split("__")) for p in ps]; assert all(p in P.index for p in ps), n; seen |= set(ps); D = P.loc[ps]
    rows.append(dict(complex=n, n_pairs=len(ps), pairs=",".join(ps), n_string_700=int(D.string_physical.notna().sum()), n_cobinding_edges=int(D.view1.sum()), mean_view1_score=round(D.view1_score.mean(), 3),
                     cobinding_recovered=bool(D.view1.mean() >= 0.5), boltz_trimmed=",".join(D.boltz_tier.astype(str))))
A = pd.DataFrame(rows); A.to_csv(f"{O}/a_known_complexes.tsv", sep="\t", index=False)
sp = set(P.index[P.string_physical.notna()]); miss = sorted(sp - seen)
# (b)
called = P[P.capselex.astype(str).str.startswith("called")]; rb = []
for k, r in called.iterrows(): rb.append(dict(pair=k, source="CAP-SELEX called: " + r.capselex, cobinding_edge=bool(r.view1), view1_score=round(r.view1_score, 3), edge_type=r.edge_type, boltz_trimmed=r.boltz_tier, curator_note=""))
for k, s in [("MBD3__ZNF512B", "literature candidate (ZNF512B-NuRD); from memory, verify"), ("ADNP__HP1", "literature candidate (ChAHP, 2018); from memory, verify")]:
    r = P.loc[key(*k.split("__"))]; rb.append(dict(pair=k, source=s, cobinding_edge=bool(r.view1), view1_score=round(r.view1_score, 3), edge_type=r.edge_type, boltz_trimmed=r.boltz_tier, curator_note=""))
B = pd.DataFrame(rb); B.to_csv(f"{O}/b_recently_revealed.tsv", sep="\t", index=False)
# (c)
RULE = {key("DR1", "THAP6"): ("complex", "."), key("DDIT3", "XPA"): ("complex_weak", "seed_unstable"), key("DR1", "HOXC9"): ("complex_weak", "seed_unstable"),
        key("HP1", "ZNF507"): ("co-localized", "trimmed_strong_only"), key("DR1", "ZNF746"): ("co-localized", "trimmed_strong_only"), key("ZNF143", "ZNF512"): ("co-localized", "trimmed_strong_only")}
base = P[P.view1 & P.string_physical.isna() & ~P.capselex.astype(str).str.startswith("called") & P.edge_type.isin(["complex", "complex_weak"])]; rc = []
for k, r in base.iterrows():
    cl, fl = RULE.get(k, (r.edge_type, "."))
    full = FL.loc[k] if k in FL.index else None; t = af3(k)
    rc.append(dict(pair=k, class_provisional=r.edge_type, class_with_rulings=cl, flag=fl, view1_score=round(r.view1_score, 3), library_pair=r.gpair, hub_partner=r.hub_partner, boltz_trimmed=r.boltz_tier, boltz_trim_ipsae=round(r.boltz_ipsae, 2),
                   boltz_fulllength="not run" if full is None else full.full_tier, boltz_verdict="." if full is None else full.verdict, af3_tier=t[0], af3_iptm=t[1], af3_ipsae=t[2]))
Cc = pd.DataFrame(rc).sort_values(["class_with_rulings", "view1_score"], ascending=[True, False]); Cc.to_csv(f"{O}/c_novel.tsv", sep="\t", index=False)
S = pd.DataFrame([dict(category="(a) known complexes", n=len(A), detail=f"{int(A.cobinding_recovered.sum())} recovered by co-binding (>= half of pairs are edges)"),
                  dict(category="(b) recently revealed", n=len(B), detail=f"{len(called)} CAP-SELEX called; {len(B) - len(called)} literature candidates to curate; {int(B.cobinding_edge.sum())} with co-binding edge"),
                  dict(category="(c) novel, with rulings: complex", n=int((Cc.class_with_rulings == "complex").sum()), detail="AF3 pending"),
                  dict(category="(c) novel, with rulings: complex_weak", n=int((Cc.class_with_rulings == "complex_weak").sum()), detail="AF3 pending; most not retested at full length"),
                  dict(category="(c) novel, demoted to co-localized", n=int((Cc.class_with_rulings == "co-localized").sum()), detail="trimmed_strong_only")])
S.to_csv(f"{O}/summary_draft.tsv", sep="\t", index=False); pd.set_option("display.width", 240); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 70)
print(S.to_string(index=False)); print(A.drop(columns="pairs").to_string(index=False)); print("STRING >= 700 pairs not assigned to a complex:", miss); print(B.to_string(index=False)); print(Cc.drop(columns=["library_pair", "af3_iptm", "af3_ipsae"]).to_string(index=False))
