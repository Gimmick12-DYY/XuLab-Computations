# Score AF3 pairs like the Boltz screen: ipTM of the top-ranked model, ipSAE (same function, PAE cutoff 10 A) best over all seeds x samples, same tier rule;
# per-seed best ipSAE and top ipTM, chain-pair PAE min. Compares with Boltz trimmed and full-length tiers. Writes af3_scores.tsv
import glob, json, os
import numpy as np, pandas as pd
C = "/work/users/d/y/dyy12/XuLab/complex_screen"; O = f"{C}/af3_novel_20261103"
S = pd.read_csv(f"{C}/screen/scores.tsv", sep="\t").set_index("pair_id")
FL = pd.concat([pd.read_csv(f"{C}/fulllength_check/fulllength_scores.tsv", sep="\t"), pd.read_csv(f"{C}/fulllength_check/retest6/fulllength_scores.tsv", sep="\t")]).set_index("pair")
def ipsae(pae, la, lb, cut=10.0):
    best = 0.0
    for (r0, r1), (c0, c1) in [((0, la), (la, la + lb)), ((la, la + lb), (0, la))]:
        sub = pae[r0:r1, c0:c1]; ok = sub < cut; n = ok.sum(1); d0 = 1.24 * np.cbrt(np.maximum(n, 27) - 15) - 1.8
        s = np.where(ok, 1 / (1 + (sub / d0[:, None]) ** 2), 0).sum(1) / np.maximum(n, 1); best = max(best, float(s.max()))
    return best
def tier(ip, ps): return "strong" if (ps >= 0.3 and ip >= 0.4) else ("weak" if ps >= 0.1 else "none")
rows = []
for n in [l.strip() for l in open(f"{O}/pairs.txt") if l.strip()]:
    d = f"{O}/out/{n.lower()}"
    if not os.path.exists(f"{d}/{n.lower()}_summary_confidences.json"): continue
    a, b = n.split("_"); pid = f"{a}__{b}"; top = json.load(open(f"{d}/{n.lower()}_summary_confidences.json"))
    r = dict(pair=pid, iptm_top=top["iptm"], chain_pair_pae_min=min(top["chain_pair_pae_min"][0][1], top["chain_pair_pae_min"][1][0]), ranking_top=top["ranking_score"], frac_disordered=top["fraction_disordered"], has_clash=top["has_clash"])
    by = {}; ips = []
    for f in sorted(glob.glob(f"{d}/seed-*_sample-*/confidences.json")):
        sd = os.path.basename(os.path.dirname(f)).split("_")[0]; c = json.load(open(f)); ch = np.array(c["token_chain_ids"]); la = int((ch == "A").sum()); lb = int((ch == "B").sum())
        s = json.load(open(f.replace("confidences.json", "summary_confidences.json"))); x = ipsae(np.array(c["pae"]), la, lb); ips.append(x)
        q = by.setdefault(sd, dict(ips=0.0, ip=-1.0, rk=-9.0)); q["ips"] = max(q["ips"], x)
        if s["ranking_score"] > q["rk"]: q["rk"], q["ip"] = s["ranking_score"], s["iptm"]
    r.update(n_models=len(ips), ipsae_max=max(ips), ipsae_median=float(np.median(ips)), tier=tier(top["iptm"], max(ips)))
    r["per_seed_iptm_ipsae_tier"] = "; ".join(f"{k.split('-')[1]}:{v['ip']:.2f}/{v['ips']:.2f} {tier(v['ip'], v['ips'])}" for k, v in sorted(by.items()))
    r["seeds_strong"] = sum(tier(v["ip"], v["ips"]) == "strong" for v in by.values())
    if pid in S.index: r.update(boltz_trim_iptm=S.loc[pid, "iptm_top"], boltz_trim_ipsae=S.loc[pid, "ipsae_max"], boltz_trim_tier=S.loc[pid, "hit"])
    if pid in FL.index: r.update(boltz_full_iptm=FL.loc[pid, "full_iptm"], boltz_full_ipsae=FL.loc[pid, "full_ipsae"], boltz_full_tier=FL.loc[pid, "full_tier"], boltz_verdict=FL.loc[pid, "verdict"])
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(f"{O}/af3_scores.tsv", sep="\t", index=False, float_format="%.3f")
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 90); print(T.round(2).to_string(index=False))
