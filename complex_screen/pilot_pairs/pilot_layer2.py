# Pilot pairs, layer 2: Boltz-2 scores per pair from (i) the pilot run (screen chains; pilot_pairs/boltz), (ii) the screen
# (screen/scores.tsv) and (iii) the calibration constructs (calib/calib_scores.tsv). Picks the model to render: pilot run >
# calibration construct (CTCF-STAG2: pilot construct p01 has the CES-pocket segment) > screen structure.
import glob, json, os
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; C = f"{X}/complex_screen"; P = f"{C}/pilot_pairs"
PAIRS = [("MAZ", "ZNF143"), ("EZH2", "MTF2"), ("MLX", "SREBF2"), ("CTCF", "STAG2"), ("RBBP7", "SRCAP")]
CALIB = {("CTCF", "STAG2"): "p01_CTCF_STAG2", ("MLX", "SREBF2"): "p18_MLX_SREBF2"}
os.makedirs(f"{P}/layer2", exist_ok=True)

def ipsae(pae, la, lb, cut=10.0):
    best = 0.0
    for (r0, r1), (c0, c1) in [((0, la), (la, la + lb)), ((la, la + lb), (0, la))]:
        sub = pae[r0:r1, c0:c1]; ok = sub < cut; n = ok.sum(1)
        d0 = 1.24 * np.cbrt(np.maximum(n, 27) - 15) - 1.8
        s = np.where(ok, 1 / (1 + (sub / d0[:, None]) ** 2), 0).sum(1) / np.maximum(n, 1)
        best = max(best, float(s.max()))
    return best

def tier(iptm, ips): return "strong" if (ips >= 0.3 and iptm >= 0.4) else ("weak" if ips >= 0.1 else "none")
S = pd.read_csv(f"{C}/screen/scores.tsv", sep="\t").set_index("pair_id")
K = pd.read_csv(f"{C}/calib/calib_scores.tsv", sep="\t").set_index("pair_id")
rows, models = [], {}
for a, b in PAIRS:
    pid = f"{a}__{b}"; tag = f"{a}_{b}"; rec = dict(pair=tag)
    d = f"{P}/boltz/out/boltz_results_inputs/predictions/{pid}"
    conf = sorted(glob.glob(f"{d}/confidence_{pid}_model_*.json"))
    if conf:
        seqs = [l.split(":", 1)[1].strip() for l in open(f"{P}/boltz/inputs/{pid}.yaml") if "sequence:" in l]
        ip, ps, cs = [], [], []
        for f in conf:
            k = f.split("_model_")[1][:-5]; c = json.load(open(f))
            ip.append(c["pair_chains_iptm"]["0"]["1"]); cs.append(c["confidence_score"])
            ps.append(ipsae(np.load(f"{d}/pae_{pid}_model_{k}.npz")["pae"], len(seqs[0]), len(seqs[1])))
        t = int(np.argmax(cs))
        rec.update(pilot_iptm_top=ip[t], pilot_ipsae_top=ps[t], pilot_ipsae_max=max(ps), pilot_tier=tier(ip[t], max(ps)),
                   pilot_tokens=len(seqs[0]) + len(seqs[1]))
        models[tag] = dict(cif=f"{d}/{pid}_model_{t}.cif", source="pilot run (screen chains)", a=a, b=b)
    if pid in S.index and S.loc[pid, "status"] == "ok":
        r = S.loc[pid]; rec.update(screen_iptm_top=r.iptm_top, screen_ipsae_max=r.ipsae_max, screen_hit=r.hit)
    if (a, b) in CALIB and CALIB[(a, b)] in K.index:
        r = K.loc[CALIB[(a, b)]]; rec.update(calib_construct=CALIB[(a, b)], calib_iptm_top=r.iptm_top, calib_ipsae_max=r.ipsae_max,
                                              calib_tier=tier(r.iptm_top, r.ipsae_max))
        cif = sorted(glob.glob(f"{C}/calib/out/boltz_results_{CALIB[(a, b)]}/predictions/{CALIB[(a, b)]}/{CALIB[(a, b)]}_model_0.cif"))
        if cif and ((a, b) == ("CTCF", "STAG2") or tag not in models):
            models[tag] = dict(cif=cif[0], source=f"calibration construct {CALIB[(a, b)]}", a=a, b=b)
    if tag not in models and os.path.exists(f"{C}/screen/structures/{pid}.cif"):
        models[tag] = dict(cif=f"{C}/screen/structures/{pid}.cif", source="screen", a=a, b=b)
    rec["rendered_model"] = models.get(tag, {}).get("source", "none")
    rows.append(rec)
R = pd.DataFrame(rows); R.to_csv(f"{P}/layer2/scores.tsv", sep="\t", index=False, float_format="%.3f")
json.dump(models, open(f"{P}/layer2/models.json", "w"), indent=1)
pd.set_option("display.width", 250); print(R.round(3).to_string(index=False))
