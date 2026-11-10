# Score the PRC2 runs like the Boltz screen, per chain pair: ipTM of the top model, ipSAE best over seeds x samples, tier per seed. Writes the top model's PAE (rounded) for the slide inset.
import glob, json, os
import numpy as np, pandas as pd
C = "/work/users/d/y/dyy12/XuLab/complex_screen"; O = f"{C}/af3_local/out"; S = f"{C}/jie_structures"
def ipsae(pae, ra, rb, cut=10.0):
    best = 0.0
    for x, y in ((ra, rb), (rb, ra)):
        sub = pae[np.ix_(x, y)]; ok = sub < cut; n = ok.sum(1); d0 = 1.24 * np.cbrt(np.maximum(n, 27) - 15) - 1.8; s = np.where(ok, 1 / (1 + (sub / d0[:, None]) ** 2), 0).sum(1) / np.maximum(n, 1); best = max(best, float(s.max()))
    return best
tier = lambda ip, ps: "strong" if (ps >= 0.3 and ip >= 0.4) else ("weak" if ps >= 0.1 else "none")
rows = []
for job, names in (("EZH2_MTF2", "AB"), ("EZH2_SUZ12_MTF2", "ABC"), ("MLX_SREBF2", "AB")):
    d = f"{O}/{job.lower()}"; top = json.load(open(f"{d}/{job.lower()}_summary_confidences.json")); c0 = json.load(open(f"{d}/{job.lower()}_confidences.json")); ch = np.array(c0["token_chain_ids"]); np.savez_compressed(f"{S}/{job}.pae.npz", pae=np.round(np.array(c0["pae"]), 1).astype(np.float16), chains=ch)
    print(job, "ipTM", top["iptm"], "ranking", top["ranking_score"], "chain_pair_iptm", top["chain_pair_iptm"])
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j <= i: continue
            by = {}; idx = {x: np.flatnonzero(ch == x) for x in names}
            for f in sorted(glob.glob(f"{d}/seed-*_sample-*/confidences.json")):
                sd = os.path.basename(os.path.dirname(f)).split("_")[0]; s = json.load(open(f.replace("confidences.json", "summary_confidences.json"))); x = ipsae(np.array(json.load(open(f))["pae"]), idx[a], idx[b]); q = by.setdefault(sd, dict(i=x, p=s["chain_pair_iptm"][i][j], r=s["ranking_score"])); q["i"] = max(q["i"], x)
                if s["ranking_score"] > q["r"]: q["r"], q["p"] = s["ranking_score"], s["chain_pair_iptm"][i][j]
            rows.append(dict(job=job, pair=f"{a}-{b}", iptm_top=round(top["chain_pair_iptm"][i][j], 3), pae_min=min(top["chain_pair_pae_min"][i][j], top["chain_pair_pae_min"][j][i]), ipsae_max=round(max(v["i"] for v in by.values()), 3), seeds_strong=sum(tier(v["p"], v["i"]) == "strong" for v in by.values()), per_seed="; ".join(f"{k.split('-')[1]}:{v['p']:.2f}/{v['i']:.2f} {tier(v['p'], v['i'])}" for k, v in sorted(by.items()))))
T = pd.DataFrame(rows); T.to_csv(f"{S}/af3_prc2_scores.tsv", sep="\t", index=False); pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 100); print(T.to_string(index=False))
