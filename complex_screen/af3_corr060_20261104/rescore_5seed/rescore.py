# Rescore the 5-seed AF3 stage (seeds5/all.txt pairs; scores_s5/<pair>.tsv from score_one.py) and check novelty. DESCRIPTIVE.
# Tier at 5 seeds: strong = >= 3 of 5 seeds strong (seed strong: ipSAE >= 0.3 and ipTM >= 0.4, score_one.py); weak = not strong and max ipSAE >= 0.1;
# none otherwise; "missing" if no 5-seed score. Novelty (R1 v3 annotations, panel names: CBX1/3/5 -> HP1, ZUP1 -> ZUFSP): known if STRING physical >= 0.70,
# CAP-SELEX called, or R1 class "complex"; otherwise novel (no literature beyond these sources is encoded). 1-seed tier shown alongside.
# Writes rescore_5seed/tiers_5seed.tsv. usage: rescore.py
import os
import pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/af3_corr060_20261104"; OUT = f"{O}/rescore_5seed"
ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; pn = lambda t: ALIAS.get(t, t); key = lambda a, b: tuple(sorted((pn(a), pn(b))))
R = pd.read_csv(f"{X}/regulatome/r1_tftf/v2_final_v3/r1_pairs_frozen_v3.tsv", sep="\t", keep_default_na=False); RK = {key(a, b): r for (a, b), (_, r) in zip(zip(R.a, R.b), R.iterrows())}
def rd(p):
    if not os.path.exists(p) or os.path.getsize(p) == 0: return None
    x = open(p).read().rstrip("\n").split("\t"); return dict(iptm=float(x[1]), ipsae_max=float(x[2]), tier_any=x[4], seeds_strong=int(x[5]), n_seeds=int(x[6]))
pairs = [l.strip() for l in open(f"{O}/seeds5/all.txt") if l.strip()]
late = f"{O}/seeds5/pairs_big_late.txt"
if os.path.exists(late): pairs += [l.strip() for l in open(late) if l.strip() and l.strip() not in pairs]
rows = []
for p in pairs:
    a, b = p.split("_", 1); s5 = rd(f"{O}/scores_s5/{p}.tsv"); s1 = rd(f"{O}/scores_s1/{p}.tsv"); r = RK.get(key(a, b))
    t5 = "missing" if s5 is None else ("strong" if s5["seeds_strong"] >= 3 else ("weak" if s5["ipsae_max"] >= .1 else "none"))
    sp_ = r["string_physical"] if r is not None else ""; cs = r["capselex"] if r is not None else ""; cl = r["class"] if r is not None else ""
    known = (sp_ != "" and float(sp_) >= .7) or str(cs).startswith("called") or cl == "complex"
    rows.append(dict(DESCRIPTIVE="DESCRIPTIVE", pair=p, panel_pair="-".join(key(a, b)), tier_5seed=t5, seeds_strong=s5["seeds_strong"] if s5 else None, n_seeds=s5["n_seeds"] if s5 else None,
                     ipsae_max_5seed=s5["ipsae_max"] if s5 else None, iptm_top_5seed=s5["iptm"] if s5 else None, tier_1seed=s1["tier_any"] if s1 else ".", string_physical=sp_ or ".", capselex=cs or ".", r1_class=cl or ".", novelty="known" if known else "novel"))
os.makedirs(OUT, exist_ok=True); T = pd.DataFrame(rows); T.to_csv(f"{OUT}/tiers_5seed.tsv", sep="\t", index=False)
print(pd.crosstab(T.tier_5seed, T.novelty, margins=True).to_string())
