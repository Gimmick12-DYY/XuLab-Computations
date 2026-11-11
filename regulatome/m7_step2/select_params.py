# m7_step2: choose hyperparameters from tuning val.json files (train/validation only) and write frozen/params.json. usage: select_params.py <kind> <expr|kd>
import glob, json, os, sys
LINK = os.environ.get("M7_LINK", "powerlaw")
import pandas as pd
O = "/work/users/d/y/dyy12/XuLab/regulatome/m7_step2"; kind, tg = sys.argv[1:3]; base = f"{O}/fits/{kind}_{LINK}/tune/{tg}"
rows = []
for f in glob.glob(f"{base}/**/val.json", recursive=True):
    d = json.load(open(f)); j = d["job"]; m = "val_r2" if tg == "expr" else "val_spearman_mean"; rows.append(dict(model=j["model"], mode=j["mode"], cond=j["cond"], rung=j["rung"], hp=int(j["hp"]), metric=d[m], params=json.dumps(d["params"])))
T = pd.DataFrame(rows); print(len(T), "tuning results"); os.makedirs(f"{O}/frozen", exist_ok=True); F = f"{O}/frozen/params_{LINK}.json"; P = json.load(open(F)) if os.path.exists(F) else {}
keys = ["model", "rung"] if tg == "expr" else ["model", "mode", "cond"]; tab = []
for k, G in T.groupby(keys):
    b = G.sort_values("metric", ascending=False).iloc[0]; key = f"expr|{b.model}|rung{b.rung}" if tg == "expr" else f"kd|{b.model}|{b['mode']}|{b.cond}"; P[key] = json.loads(b.params); tab.append(dict(key=key, n=len(G), best=round(b.metric, 5), worst=round(G.metric.min(), 5), params=b.params))
json.dump(P, open(F, "w"), indent=1, sort_keys=True); S = pd.DataFrame(tab); S.to_csv(f"{O}/frozen/tuning_{tg}_{LINK}.tsv", sep="\t", index=False); print(S.to_string(index=False))
