# Split the job list into Boltz batches (<= 90 pairs, <= ~3 h predicted GPU time each); b000 = 5 largest heterodimers (memory check)
import json, os, sys
import numpy as np, pandas as pd
root = sys.argv[1]
f = np.poly1d(json.load(open(f"{root}/calib/runtime_fit.json")))
j = pd.read_csv(f"{root}/jobs/joblist.tsv", sep="\t")
j["pred_sec"] = np.maximum(f(j.n_tokens) - 50, 20)  # ~50 s per call is model loading, paid once per batch
order = {"het": 0, "homo": 1, "hp1_paralog": 2}
j = j.sort_values(["set", "n_tokens"], key=lambda s: s.map(order) if s.name == "set" else -s, ascending=[True, True])
first = j[j.set == "het"].head(5)
rest = j.drop(first.index)
batches, cur, t = [list(first.pair_id)], [], 0
for _, r in rest.iterrows():
    if cur and (len(cur) >= 90 or t + r.pred_sec > 10800 or r.set != prev):
        batches.append(cur); cur, t = [], 0
    cur.append(r.pair_id); t += r.pred_sec; prev = r.set
batches.append(cur)
os.makedirs(f"{root}/screen/batches", exist_ok=True)
sub = {"het": "het", "homo": "homo", "hp1_paralog": "hp1_paralogs"}
st = dict(zip(j.pair_id, j.set)); ps = dict(zip(j.pair_id, j.pred_sec))
for i, b in enumerate(batches):
    name = f"b{i:03d}"; d = f"{root}/screen/inputs/{name}"; os.makedirs(d, exist_ok=True)
    open(f"{root}/screen/batches/{name}.txt", "w").write("\n".join(b) + "\n")
    for p in b:
        dst = f"{d}/{p}.yaml"
        if not os.path.exists(dst): os.symlink(f"{root}/jobs/inputs/{sub[st[p]]}/{p}.yaml", dst)
    print(name, st[b[0]], len(b), f"{sum(ps[p] for p in b) / 3600:.2f} h")
