# Scores one finished pair like af3_novel_20261103/af3_score.py (ipTM of the top model; ipSAE best over all seeds x samples, PAE cutoff 10 A; strong = ipSAE >= 0.3 and ipTM >= 0.4, weak = ipSAE >= 0.1). usage: score_one.py A_B outdir scores_dir
import glob, json, os, sys
import numpy as np
n, d, sc = sys.argv[1:4]; l = n.lower()
def ipsae(pae, la, lb, cut=10.0):
    best = 0.0
    for (r0, r1), (c0, c1) in [((0, la), (la, la + lb)), ((la, la + lb), (0, la))]:
        sub = pae[r0:r1, c0:c1]; ok = sub < cut; k = ok.sum(1); d0 = 1.24 * np.cbrt(np.maximum(k, 27) - 15) - 1.8
        s = np.where(ok, 1 / (1 + (sub / d0[:, None]) ** 2), 0).sum(1) / np.maximum(k, 1); best = max(best, float(s.max()))
    return best
def tier(ip, ps): return "strong" if (ps >= 0.3 and ip >= 0.4) else ("weak" if ps >= 0.1 else "none")
top = json.load(open(f"{d}/{l}_summary_confidences.json")); by = {}; ips = []
for f in sorted(glob.glob(f"{d}/seed-*_sample-*/confidences.json")):
    sd = os.path.basename(os.path.dirname(f)).split("_")[0]; c = json.load(open(f)); ch = np.array(c["token_chain_ids"]); x = ipsae(np.array(c["pae"]), int((ch == "A").sum()), int((ch == "B").sum())); ips.append(x); s = json.load(open(f.replace("confidences.json", "summary_confidences.json")))
    q = by.setdefault(sd, dict(ips=0.0, ip=-1.0, rk=-9.0)); q["ips"] = max(q["ips"], x)
    if s["ranking_score"] > q["rk"]: q["rk"], q["ip"] = s["ranking_score"], s["iptm"]
a, b = n.split("_"); row = [f"{a}__{b}", top["iptm"], round(max(ips), 3), round(float(np.median(ips)), 3), tier(top["iptm"], max(ips)), sum(tier(v["ip"], v["ips"]) == "strong" for v in by.values()), len(by), len(ips), top["fraction_disordered"], top["has_clash"], min(top["chain_pair_pae_min"][0][1], top["chain_pair_pae_min"][1][0]), "; ".join(f"{k.split('-')[1]}:{v['ip']:.2f}/{v['ips']:.2f} {tier(v['ip'], v['ips'])}" for k, v in sorted(by.items()))]
open(f"{sc}/{n}.tsv", "w").write("\t".join(map(str, row)) + "\n")
