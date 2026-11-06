# Score Boltz-2 calibration runs: ipTM / ipSAE per pair (5 samples), AUROC positives vs negatives, length/disorder bias
import json, glob, os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

root, pairs_tsv = sys.argv[1:3]
pairs = pd.read_csv(pairs_tsv, sep="\t")
tim = pd.read_csv(f"{root}/calib/timing.tsv", sep="\t", header=None, names=["pair_id", "rc", "sec", "host", "gpu"])

def ipsae(pae, la, lb, cut=10.0):
    """ipSAE (Dunbrack 2025): max over directions of max_i mean_j 1/(1+(PAE/d0)^2) for PAE<cut, d0 from n valid j"""
    best = 0.0
    for (r0, r1), (c0, c1) in [((0, la), (la, la + lb)), ((la, la + lb), (0, la))]:
        sub = pae[r0:r1, c0:c1]
        ok = sub < cut
        n = ok.sum(1)
        d0 = 1.24 * np.cbrt(np.maximum(n, 27) - 15) - 1.8
        s = np.where(ok, 1 / (1 + (sub / d0[:, None]) ** 2), 0).sum(1) / np.maximum(n, 1)
        best = max(best, float(s.max()))
    return best

rows = []
for _, p in pairs.iterrows():
    d = f"{root}/calib/out/boltz_results_{p.pair_id}/predictions/{p.pair_id}"
    files = sorted(glob.glob(f"{d}/confidence_{p.pair_id}_model_*.json"))
    if not files: rows.append(dict(pair_id=p.pair_id)); continue
    # chain lengths from the input yaml order
    seqs = [l.split(":", 1)[1].strip() for l in open(f"{root}/calib/inputs/{p.pair_id}.yaml") if "sequence:" in l]
    L = [len(s) for s in seqs]
    # query chain A vs partner(s): for 3-chain inputs take the best A-B / A-C interface
    partners = list(range(1, len(L)))
    ip, ps, cs, pl = [], [], [], []
    for f in files:
        c = json.load(open(f)); k = f.split("_model_")[1].split(".")[0]
        ip.append(max(c["pair_chains_iptm"]["0"][str(j)] for j in partners))
        cs.append(c["confidence_score"]); pl.append(c["complex_plddt"])
        pae = np.load(f"{d}/pae_{p.pair_id}_model_{k}.npz")["pae"]
        off = np.cumsum([0] + L)
        best = 0
        for j in partners:
            idx = np.r_[off[0]:off[1], off[j]:off[j + 1]]
            best = max(best, ipsae(pae[np.ix_(idx, idx)], L[0], L[j]))
        ps.append(best)
    top = int(np.argmax(cs))
    rows.append(dict(pair_id=p.pair_id, iptm_top=ip[top], iptm_max=max(ip), iptm_mean=np.mean(ip), iptm_sd=np.std(ip),
                     ipsae_top=ps[top], ipsae_max=max(ps), plddt_top=pl[top], n_samples=len(files)))

r = pairs[["pair_id", "cls", "gene_a", "gene_b", "n_tokens"]].merge(pd.DataFrame(rows), on="pair_id").merge(tim[["pair_id", "sec", "gpu"]], on="pair_id", how="left")

# disorder of each construct: fraction of residues with AFDB pLDDT < 50 (token-weighted over chains)
def frac_dis(g, a, b):
    g = {"ZUFSP": "ZUP1"}.get(g, g); f = f"{root}/seqs/afdb/{g}.json"
    if not os.path.exists(f): return np.nan, 0
    pl = np.array(json.load(open(f))["plddt"])[a - 1:b]; return (pl < 50).sum(), len(pl)
dis = []
for _, p in pairs.iterrows():
    n = t = 0
    for g, a, b in [(p.gene_a, p.start_a, p.end_a), (p.gene_b, p.start_b, p.end_b)]:
        x, y = frac_dis(g, a, b); n += x; t += y
    dis.append(n / t if t else np.nan)
r["frac_disordered"] = dis
r.to_csv(f"{root}/calib/calib_scores.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
print(r.drop(columns=["gpu"]).round(3).to_string())

lab = r[r.cls.isin(["positive", "negative"])].dropna(subset=["iptm_top"])
y = (lab.cls == "positive").astype(int)
print(f"\nlabelled: {y.sum()} positive, {(1 - y).sum()} negative")
for m in ["iptm_top", "iptm_max", "iptm_mean", "ipsae_top", "ipsae_max"]:
    print(f"AUROC {m:10s} {roc_auc_score(y, lab[m]):.3f}")
neg = lab[lab.cls == "negative"]
print("max negative iptm_top:", round(neg.iptm_top.max(), 3), " max negative ipsae_top:", round(neg.ipsae_top.max(), 3))
for thr in [0.4, 0.5, 0.6, 0.7, 0.8]:
    tp = (lab[lab.cls == "positive"].iptm_top >= thr).sum(); fp = (neg.iptm_top >= thr).sum()
    print(f"ipTM >= {thr}: TP {tp}/{y.sum()}  FP {fp}/{len(neg)}")
ok = r.dropna(subset=["iptm_top"])
for m in ["iptm_top", "ipsae_top"]:
    print(f"bias {m}: spearman vs n_tokens all={spearmanr(ok.n_tokens, ok[m])[0]:.2f}, negatives={spearmanr(neg.n_tokens, neg[m])[0]:.2f}; "
          f"vs frac_disordered all={spearmanr(ok.frac_disordered, ok[m], nan_policy='omit')[0]:.2f}, "
          f"negatives={spearmanr(neg.frac_disordered, neg[m], nan_policy='omit')[0]:.2f}")
t = r.dropna(subset=["sec"])
fit = np.polyfit(t.n_tokens, t.sec, 2)
print("runtime fit sec = %.3g*n^2 + %.3g*n + %.3g" % tuple(fit), "| GPUs:", t.gpu.value_counts().to_dict())
json.dump(list(fit), open(f"{root}/calib/runtime_fit.json", "w"))
