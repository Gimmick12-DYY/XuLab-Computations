# R1 pre-freeze tables: reads the provisional r1_build_v2.py pairs table and applies the design rulings; writes to v2_final/. --freeze writes frozen copies and their sha256; nothing in v2/ changes.
# The class uses the full-length Boltz tier where one exists; trimmed and full-length tiers are both recorded, with AF3 alongside. Final rulings (2026-11-02): DR1-THAP6 complex_weak; DDIT3-XPA complex_weak, seed_unstable; DR1-HOXC9 co-localized;
# CBX1(HP1)-ZNF507, DR1-ZNF746, ZNF143-ZNF512 co-localized, trimmed_strong_only; "complex (weak, structure only)" renamed physical-only (weak). AF3-strong MLX-SREBF2, HOXC4-RBBP4, MTF2-RBBP4 are physical-only (structure) at w 0.5; the other
# physical-only (structure) pairs are w 0, trimmed_only_untested. Every NEUROD4 edge is w 0, not_expressed. AF3 elsewhere is recorded as support only. Boltz none on a STRING pair = no information.
# w: complex, DNA-guided 1.0; complex_weak, AF3-strong physical-only (structure), physical-only (DNA) 0.5 provisional; others 0.
import hashlib, os, sys
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; R = f"{X}/regulatome/r1_tftf"; O = f"{R}/v2_final"; C = f"{X}/complex_screen"; os.makedirs(O, exist_ok=True); FREEZE = "--freeze" in sys.argv
key = lambda a, b: "__".join(sorted((a, b))); PAN = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; pn = lambda g: PAN.get(g, g)
P = pd.read_csv(f"{R}/v2/r1v2_pairs_provisional.tsv", sep="\t", keep_default_na=False, na_values=[""]); P["k"] = [key(a, b) for a, b in zip(P.a, P.b)]
S = pd.read_csv(f"{C}/screen/scores.tsv", sep="\t"); S = S[(S.status == "ok") & S.set.isin(["het", "hp1_paralog"])]; tr = {}
for r in S.itertuples():
    k = key(pn(r.chain_a), pn(r.chain_b)); t = "strong" if (r.ipsae_max >= 0.3 and r.iptm_top >= 0.4) else ("weak" if r.ipsae_max >= 0.1 else "none"); o = {"none": 0, "weak": 1, "strong": 2}
    if k not in tr or o[t] > o[tr[k]]: tr[k] = t
P["boltz_trimmed_tier"] = P.k.map(tr)
FL = pd.concat([pd.read_csv(f"{C}/fulllength_check/fulllength_scores.tsv", sep="\t"), pd.read_csv(f"{C}/fulllength_check/retest6/fulllength_scores.tsv", sep="\t")]); FL["k"] = [key(*[pn(g) for g in p.split("__")]) for p in FL.pair]; FL = FL.set_index("k")
P["boltz_full_tier"] = P.k.map(FL.full_tier).fillna("."); P["boltz_full_verdict"] = P.k.map(FL.verdict).fillna(".")
AF = pd.read_csv(f"{C}/af3_local/af3_scores.tsv", sep="\t"); AF["k"] = [key(*[pn(g) for g in p.split("__")]) for p in AF.pair]; AF = AF.set_index("k")
for c, s in (("af3_tier", "tier"), ("af3_iptm", "iptm_top"), ("af3_ipsae", "ipsae_max"), ("af3_pae_min", "chain_pair_pae_min"), ("af3_seeds_strong", "seeds_strong")): P[c] = P.k.map(AF[s])
P["af3_tier"] = P.af3_tier.fillna("not_run")
def cls(r):
    t = r.edge_type
    if t.startswith("STAG2") or t == "DNA-guided" or t == "physical-only (DNA)": return t
    if pd.notna(r.string_physical) and r.view1: return "complex"
    if r.boltz_full_tier != ".":
        f = r.boltz_full_tier
        if r.view1: return {"strong": "complex", "weak": "complex_weak"}.get(f, "co-localized")
        return {"strong": "physical-only (structure)", "weak": "physical-only (weak)"}.get(f, ".")
    return "physical-only (weak)" if t == "complex (weak, structure only)" else t
P["class"] = P.apply(cls, axis=1); P["flags"] = ""
def flag(m, s): P.loc[m, "flags"] = P.loc[m, "flags"] + s + ";"
K = lambda a, b: P.k == key(a, b)
P.loc[K("DR1", "THAP6"), "class"] = "complex_weak"; flag(K("DR1", "THAP6"), "boltz_full_strong_3of3")
P.loc[K("DDIT3", "XPA"), "class"] = "complex_weak"; flag(K("DDIT3", "XPA"), "seed_unstable")
P.loc[K("DR1", "HOXC9"), "class"] = "co-localized"; flag(K("DR1", "HOXC9"), "seed_unstable_af3_none")
for a, b in (("HP1", "ZNF507"), ("DR1", "ZNF746"), ("ZNF143", "ZNF512")): P.loc[K(a, b), "class"] = "co-localized"; flag(K(a, b), "trimmed_strong_only")
AFS = [("MLX", "SREBF2"), ("HOXC4", "RBBP4"), ("MTF2", "RBBP4")]
for a, b in AFS: P.loc[K(a, b), "class"] = "physical-only (structure)"
W = {"complex": 1.0, "DNA-guided": 1.0, "complex_weak": 0.5, "physical-only (DNA)": 0.5}; P["w"] = P["class"].map(W).fillna(0.0)
for a, b in AFS: P.loc[K(a, b), "w"] = 0.5
flag(K("MLX", "SREBF2"), "triple_strong"); flag(K("MLX", "SREBF2"), "bHLH_dimer_caveat"); flag(K("MTF2", "RBBP4"), "STRING_0.996")
flag(P["class"].eq("physical-only (structure)") & (P.w == 0), "trimmed_only_untested")
flag(P["class"].eq("physical-only (DNA)"), "capselex_no_cobinding"); flag(K("RBBP4", "RBBP7"), "af3_none_paralogs")
neu = (P.a == "NEUROD4") | (P.b == "NEUROD4"); P.loc[neu & (P["class"] != "."), "w"] = 0.0; flag(neu & (P["class"] != "."), "not_expressed")
flag(P.string_physical.notna() & P.boltz_trimmed_tier.eq("none"), "boltz_none_no_information")
flag(P.af3_tier.eq("strong"), "af3_strong"); flag(P.af3_tier.isin(["none"]) & P["class"].isin(["complex", "complex_weak", "physical-only (structure)"]), "af3_none")
P["prc2_gene_edge_flag"] = [("not_PRC2_enriched" if {a, b} & {"EZH2", "MTF2", "PCGF6"} else ".") for a, b in zip(P.a, P.b)]; P["flags"] = P["flags"].replace("", ".").str.rstrip(";")
cols = ["a", "b", "class", "w", "flags", "prc2_gene_edge_flag", "view1", "view1_score", "gpair", "hub_partner", "boltz_trimmed_tier", "boltz_iptm", "boltz_ipsae", "boltz_full_tier", "boltz_full_verdict", "af3_tier", "af3_iptm", "af3_ipsae", "af3_pae_min", "af3_seeds_strong", "capselex", "string_physical", "edge_type"]
Pc = P[cols].rename(columns={"edge_type": "edge_type_provisional"})
tag = "frozen" if FREEZE else "prefreeze"
Pc.to_csv(f"{O}/r1_pairs_{tag}.tsv", sep="\t", index=False, float_format="%.4f"); Pc[Pc["class"] != "."].to_csv(f"{O}/r1_edges_{tag}.tsv", sep="\t", index=False, float_format="%.4f")
Sm = Pc.groupby("class").agg(n=("a", "size"), n_w_positive=("w", lambda s: int((s > 0).sum())), w_max=("w", "max"), view1=("view1", "sum"), af3_run=("af3_tier", lambda s: int((s != "not_run").sum())), af3_strong=("af3_tier", lambda s: int((s == "strong").sum())), af3_weak=("af3_tier", lambda s: int((s == "weak").sum()))).reset_index(); Sm.to_csv(f"{O}/r1_class_counts_{tag}.tsv", sep="\t", index=False)
T = AF.reset_index(drop=True)[["pair", "iptm_top", "chain_pair_pae_min", "ipsae_max", "tier", "seeds_strong", "boltz_trim_iptm", "boltz_trim_ipsae", "boltz_trim_tier", "boltz_full_iptm", "boltz_full_ipsae", "boltz_full_tier", "boltz_verdict"]]; T.to_csv(f"{O}/af3_vs_boltz.tsv", sep="\t", index=False, float_format="%.3f")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 100); pd.set_option("display.max_colwidth", 50)
print(Sm.to_string(index=False)); print(Pc[Pc["class"].isin(["complex", "DNA-guided", "physical-only (DNA)"]) | Pc["flags"].str.contains("seed_unstable|trimmed_strong_only|triple_strong|STRING_0.996|boltz_full_strong") | (Pc.a.eq("HOXC4") & Pc.b.eq("RBBP4"))][["a", "b", "class", "w", "flags", "gpair", "view1_score", "boltz_trimmed_tier", "boltz_full_tier", "af3_tier", "af3_ipsae", "string_physical"]].round(3).to_string(index=False))
print(Pc[Pc["class"].eq("physical-only (structure)") & Pc.af3_tier.ne("not_run")][["a", "b", "boltz_full_tier", "af3_tier", "af3_ipsae"]].round(3).to_string(index=False)); print("physical-only (structure) untested:", int(Pc["flags"].str.contains("trimmed_only_untested").sum())); print("provisional to final class changes:"); print(pd.crosstab(Pc.edge_type_provisional, Pc["class"]).to_string())
print("AF3 strong:", Pc[Pc.af3_tier.eq("strong")][["a", "b", "class", "af3_ipsae"]].round(2).to_string(index=False))
if FREEZE:
    with open(f"{O}/r1_frozen.sha256", "w") as f:
        for t in ("pairs", "edges"): p = f"{O}/r1_{t}_frozen.tsv"; f.write(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {p}\n")
