# Slide 1 table data: highest percentile across the five element sets per pair, novel candidates with a predicted interface. Writes data/summary_table.tsv and data/summary_novel_all.tsv.
import numpy as np, pandas as pd
def rd(f):
    m = pd.read_csv(f, sep="\t", index_col=0); m.index = m.index.str.strip().str.upper(); m.columns = m.columns.str.strip().str.upper(); return m
S = {"all": rd("data/corr_sets/pct_all.tsv"), "no quiescent": rd("data/corr_sets/pct_noquies.tsv"), "TSS": rd("data/corr_sets/pct_tss.tsv"), "enhancer": rd("data/corr_sets_extra/pct_enhancer_biv.tsv"), "repressive": rd("data/corr_sets_extra/pct_repressive.tsv")}
ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; pn = lambda t: ALIAS.get(t, t); key = lambda a, b: tuple(sorted((a, b)))
fl = lambda v: f"{np.floor(v * 100 + 1e-9) / 100:.2f}"
def best(a, b): k = max(S, key=lambda s: S[s].loc[a, b]); return float(S[k].loc[a, b]), k
tfs = list(S["all"].index); P = pd.read_csv("data/screen_ro/r1_pairs_frozen.tsv", sep="\t", keep_default_na=False, na_values=[""])
strg = {key(r.a, r.b) for r in P[P.string_physical.notna()].itertuples()}; cap = {key(r.a, r.b) for r in P[P.capselex.str.startswith("called")].itertuples()}; stag = {key(r.a, r.b) for r in P[P["class"].str.startswith("STAG2")].itertuples()}
hub = {pn(t) for t in pd.read_csv("data/screen_ro/hub_partner.tsv", sep="\t").query("hub_partner").tf}
A = pd.read_csv("data/screen_ro/af3_vs_boltz.tsv", sep="\t", keep_default_na=False); F = pd.read_csv("data/screen_ro/fulllength_scores.tsv", sep="\t", keep_default_na=False); B = pd.read_csv("data/screen_ro/scores.tsv", sep="\t", keep_default_na=False); B = B[B.set.isin(["het", "hp1_paralog"])]
RANK = {"strong": 3, "weak": 2, "none": 1, ".": 1}; S_ = {}
def put(p, rank_src, tier, text):
    k = key(*[pn(x) for x in p.split("__")])
    if k[0] == k[1]: return
    c = S_.setdefault(k, {})
    if rank_src not in c or RANK[tier] > RANK[c[rank_src][0]]: c[rank_src] = (tier, text)
for r in A.itertuples(): put(r.pair, "af3", r.tier, f"AF3 {r.tier} {r.seeds_strong}/5" if r.tier == "strong" else f"AF3 {r.tier}")
for r in F.itertuples(): put(r.pair, "full", r.full_tier, f"Boltz full-length {r.full_tier} (ipTM {r.full_iptm:.2f}); AF3 not run")
for r in B.itertuples(): put(r.pair_id, "screen", r.hit, f"Boltz {r.hit} (ipTM {float(r.iptm_top):.2f}); AF3 not run" if r.hit != "." else "Boltz none")
def struct(k):
    c = S_.get(k, {})
    for src in ("af3", "full", "screen"):
        if src in c: return src, c[src][0], c[src][1]
    return None, None, None
print("MLX-SREBF2", {k: round(S[k].loc["MLX", "SREBF2"], 4) for k in S}); rows = []; KN = set()
CUR = {"MAZ–ZNF143": ("Known", [("MAZ", "ZNF143")]), "EZH2–MTF2": ("Known", [("EZH2", "MTF2")]), "WIZ–ZNF644": ("Known", [("WIZ", "ZNF644")]), "MEF2A–MEF2D": ("Known", [("MEF2A", "MEF2D")]), "HP1–PCGF6": ("Known", [("HP1", "PCGF6")]), "RBBP4–ZNF512B": ("Known", [("RBBP4", "ZNF512B")]), "MBD3–RBBP4/7": ("Known", [("MBD3", "RBBP4"), ("MBD3", "RBBP7")]), "CHAMP1–HP1": ("Known", [("CHAMP1", "HP1")]), "ADNP–HP1": ("Known", [("ADNP", "HP1")]), "DNMT1–HP1/EZH2": ("Known", [("DNMT1", "HP1"), ("DNMT1", "EZH2")]), "RBBP4–SMARCA1": ("Known", [("RBBP4", "SMARCA1")]), "CTCF–STAG2": ("Known", [("CTCF", "STAG2")]), "RELA–ZBTB7A": ("Known", [("RELA", "ZBTB7A")]), "FOXK1–MLX": ("CAP-SELEX candidate", [("FOXK1", "MLX")]), "HOXD10–SREBF2": ("CAP-SELEX candidate", [("HOXD10", "SREBF2")]), "MLX–MSX2": ("CAP-SELEX candidate", [("MLX", "MSX2")]), "MLX–SREBF2": ("Novel", [("MLX", "SREBF2")]), "HOXC4–RBBP4": ("Novel", [("HOXC4", "RBBP4")])}
for nm, (cat, ps) in CUR.items():
    vs = [(best(*p), p) for p in ps]; (v, s_), p_ = max(vs, key=lambda x: x[0][0]); rows.append(dict(pair=nm, category=cat, value=v, set=s_, best_of="; ".join(f"{a}-{b} {fl(best(a,b)[0])}" for a, b in ps))); KN |= {key(*p) for p in ps}
print(pd.DataFrame(rows).assign(value=lambda d: d.value.round(4)).to_string(index=False)); C = pd.DataFrame(rows); C.to_csv("data/summary_current_values.tsv", sep="\t", index=False)
N = []
for i, a in enumerate(tfs):
    for b in tfs[i + 1:]:
        k = key(a, b)
        if k in KN or k in strg or k in cap or k in stag or "NEUROD4" in k: continue
        v, s_ = best(a, b)
        if v < .60: continue
        src, t, tx = struct(k)
        if t not in ("strong", "weak"): continue
        N.append(dict(a=k[0], b=k[1], value=round(v, 4), set=s_, src=src, tier=t, structure=tx, promiscuous=",".join(x for x in k if x in hub), all_pct=round(S["all"].loc[a, b], 3)))
N = pd.DataFrame(N).sort_values("value", ascending=False); N.to_csv("data/summary_novel_all.tsv", sep="\t", index=False); print(len(N), N.tier.value_counts().to_dict(), N.src.value_counts().to_dict(), "strg", len(strg), "cap", cap, "excluded-known in corr"); print(N.head(45).to_string(index=False))
