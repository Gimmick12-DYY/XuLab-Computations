# Score one finished screen batch: ipTM / ipSAE over the 5 samples, keep the top CIF only if ipSAE_max >= 0.1,
# delete PAE matrices and other samples. Writes screen/batch_scores/<batch>.tsv
import json, glob, os, sys, shutil
import numpy as np

root, batch = sys.argv[1:3]
pred = f"{root}/screen/out/{batch}/boltz_results_{batch}/predictions"
keep_dir = f"{root}/screen/structures"
os.makedirs(keep_dir, exist_ok=True); os.makedirs(f"{root}/screen/batch_scores", exist_ok=True)
ids = [l.strip() for l in open(f"{root}/screen/batches/{batch}.txt") if l.strip()]

def ipsae(pae, la, lb, cut=10.0):
    best = 0.0
    for (r0, r1), (c0, c1) in [((0, la), (la, la + lb)), ((la, la + lb), (0, la))]:
        sub = pae[r0:r1, c0:c1]; ok = sub < cut; n = ok.sum(1)
        d0 = 1.24 * np.cbrt(np.maximum(n, 27) - 15) - 1.8
        s = np.where(ok, 1 / (1 + (sub / d0[:, None]) ** 2), 0).sum(1) / np.maximum(n, 1)
        best = max(best, float(s.max()))
    return best

cols = ["pair_id", "status", "n_samples", "iptm_top", "iptm_max", "iptm_mean", "ipsae_top", "ipsae_max", "ipsae_mean",
        "plddt_top", "conf_top", "cif"]
out = []
for pid in ids:
    y = os.path.realpath(f"{root}/screen/inputs/{batch}/{pid}.yaml")
    seqs = [l.split(":", 1)[1].strip() for l in open(y) if "sequence:" in l]
    L = [len(seqs[0])] * 2 if len(seqs) == 1 else [len(s) for s in seqs]
    d = f"{pred}/{pid}"
    files = sorted(glob.glob(f"{d}/confidence_{pid}_model_*.json"))
    if not files:
        out.append([pid, "missing"] + ["NA"] * (len(cols) - 2)); continue
    ip, ps, cs, pl, ks = [], [], [], [], []
    for f in files:
        k = f.split("_model_")[1].split(".")[0]; c = json.load(open(f))
        ip.append(c["pair_chains_iptm"]["0"]["1"]); cs.append(c["confidence_score"]); pl.append(c["complex_plddt"]); ks.append(k)
        pz = f"{d}/pae_{pid}_model_{k}.npz"
        ps.append(ipsae(np.load(pz)["pae"], L[0], L[1]) if os.path.exists(pz) else np.nan)
    t = int(np.argmax(cs))
    cif = "."
    if np.nanmax(ps) >= 0.1:
        src = f"{d}/{pid}_model_{ks[t]}.cif"
        if os.path.exists(src):
            cif = f"{keep_dir}/{pid}.cif"; shutil.copy(src, cif)
            json.dump(json.load(open(files[t])), open(f"{keep_dir}/{pid}.confidence.json", "w"))
    out.append([pid, "ok", len(files), ip[t], max(ip), np.mean(ip), ps[t], np.nanmax(ps), np.nanmean(ps), pl[t], cs[t], cif])
    shutil.rmtree(d)  # PAE matrices, other samples

with open(f"{root}/screen/batch_scores/{batch}.tsv", "w") as f:
    f.write("\t".join(cols) + "\n")
    for r in out:
        f.write("\t".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + "\n")
print(batch, sum(r[1] == "ok" for r in out), "ok /", len(out))
