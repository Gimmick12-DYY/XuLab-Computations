# m7_step2 Amendment 7 (EXPLORATORY, post-hoc): widened-universe expression ladder. Cannot change the section 7 expression verdict.
# usage: m7_wide.py tune <link>            (raw binding; train/validation only)  -> frozen/params_wide_<link>.json, wide_exploratory/tuning_wide_<link>.tsv
#        m7_wide.py score <raw|imp> <link> (after the freeze commit; FROZEN_WIDE.txt must exist) -> wide_exploratory/res_<kind>_<link>/{ladder,dropone}.tsv
import json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
O = "/work/users/d/y/dyy12/XuLab/regulatome/m7_step2"; W = f"{O}/wide_exploratory"; LAB = "EXPLORATORY (Amendment 7, post-hoc)"
NB = 2000; BSEED = 20261116; NT = 79; NP_ = 38; NTH = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
ENET = [dict(alpha=a, l1_ratio=l) for a in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2) for l in (.1, .5, .9)]
LGB = [dict(num_leaves=n, min_data_in_leaf=m, feature_fraction=f) for n in (7, 15, 31) for m in (10, 30) for f in (.7, 1.0)]
G = pd.read_csv(f"{O}/splits/gene_split_wide.tsv", sep="\t"); TR = (G.set == "train").values; VA = (G.set == "val").values; TE = (G.set == "test").values
Y = {"level": G.y_level.values.astype(float), "detect": G.y_detect.values.astype(int)}
def load(kind, link):
    p = f"{O}/features/{kind}_{link}_wide"; base = np.load(f"{p}/base.npz")["base"]; F = {"real": dict(np.load(f"{p}/real.npz"))}
    for d in range(5): F[f"pseudo{d}"] = dict(np.load(f"{p}/pseudo{d}.npz"))
    return base, F
def design(base, F, cond, rung):
    b = base[:, :5 if rung == 1 else 7]
    if rung <= 2: return b
    D = F[cond]; M = [b, D["prom"].T, D["dist"].T]
    if rung == 4: M.append(D["pair"].T)
    return np.hstack(M).astype(np.float32)
def fit(model, target, p, Xtr, ytr, Xva, yva):
    if model == "enet":
        if target == "level":
            from sklearn.linear_model import ElasticNet; return ElasticNet(alpha=p["alpha"], l1_ratio=p["l1_ratio"], max_iter=5000, tol=1e-4).fit(Xtr, ytr)
        from sklearn.linear_model import LogisticRegression; return LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=p["l1_ratio"], C=1.0 / (p["alpha"] * len(ytr)), max_iter=3000, tol=1e-4).fit(Xtr, ytr)
    import lightgbm as lgb
    ds = lgb.Dataset(Xtr, label=ytr, free_raw_data=False, params=dict(feature_pre_filter=False)); v = lgb.Dataset(Xva, label=yva, reference=ds, free_raw_data=False, params=dict(feature_pre_filter=False))
    obj = dict(objective="l2") if target == "level" else dict(objective="binary", metric="binary_logloss")
    return lgb.train(dict(**obj, learning_rate=.05, num_leaves=p["num_leaves"], min_data_in_leaf=p["min_data_in_leaf"], feature_fraction=p["feature_fraction"], bagging_fraction=.8, bagging_freq=1, lambda_l2=1.0, num_threads=NTH, seed=0, verbosity=-1), ds, 2000, valid_sets=[v], callbacks=[lgb.early_stopping(50, verbose=False)])
def predict(m, model, target, X):
    if model == "enet" and target == "detect": return m.predict_proba(X)[:, 1]
    return m.predict(X)
def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
def metric(target, y, p): return {"r2": r2(y, p)} if target == "level" else {"auroc": roc_auc_score(y, p), "ap": average_precision_score(y, p)}

if sys.argv[1] == "tune":
    link = sys.argv[2]; base, F = load("raw", link); rows = []; P = {}
    for target in ("level", "detect"):
        for model, grid in (("enet", ENET), ("lgb", LGB)):
            for rung in (1, 2, 3, 4):
                X = design(base, F, "real", rung); best = None
                for h, p in enumerate(grid):
                    m = fit(model, target, p, X[TR], Y[target][TR], X[VA], Y[target][VA]); pv = predict(m, model, target, X[VA])
                    s = r2(Y[target][VA], pv) if target == "level" else roc_auc_score(Y[target][VA], pv); rows.append(dict(label=LAB, link=link, target=target, model=model, rung=rung, hp=h, params=json.dumps(p), val_metric=s))
                    if best is None or s > best[0]: best = (s, p)
                P[f"{target}|{model}|rung{rung}"] = best[1]; print(link, target, model, rung, round(best[0], 4), best[1], flush=True)
    json.dump(P, open(f"{O}/frozen/params_wide_{link}.json", "w"), indent=1, sort_keys=True); pd.DataFrame(rows).to_csv(f"{W}/tuning_wide_{link}.tsv", sep="\t", index=False, float_format="%.5f")

elif sys.argv[1] == "score":
    assert os.path.exists(f"{O}/FROZEN_WIDE.txt")
    kind, link = sys.argv[2], sys.argv[3]; P = json.load(open(f"{O}/frozen/params_wide_{link}.json")); base, F = load(kind, link); res = f"{W}/res_{kind}_{link}"; os.makedirs(res, exist_ok=True)
    n = int(TE.sum()); idx = np.random.default_rng(BSEED).integers(0, n, (NB, n)); L = []
    def boot(target, y, p, key):
        out = np.empty(NB)
        for b in range(NB):
            i = idx[b]; yi = y[i]
            if target == "level": out[b] = r2(yi, p[i])
            else: out[b] = roc_auc_score(yi, p[i]) if key == "auroc" else average_precision_score(yi, p[i])
        return out
    ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    for target in ("level", "detect"):
        keys = ["r2"] if target == "level" else ["auroc", "ap"]; yt = Y[target][TE]
        for model in ("enet", "lgb"):
            prev = {}
            for rung in (1, 2, 3, 4):
                p = P[f"{target}|{model}|rung{rung}"]; conds = ["real"] + ([f"pseudo{d}" for d in range(5)] if rung >= 3 else []); preds = {}
                for c in conds:
                    X = design(base, F, c, rung); m = fit(model, target, p, X[TR], Y[target][TR], X[VA], Y[target][VA]); preds[c] = predict(m, model, target, X[TE])
                for k in keys:
                    v = metric(target, yt, preds["real"])[k]; bo = boot(target, yt, preds["real"], k); lo, hi = ci(bo)
                    row = dict(label=LAB, kind=kind, link=link, target=target, model=model, rung=rung, metric=k, value=v, lo=lo, hi=hi)
                    if k in prev: row["gain_vs_prev"] = v - prev[k][0]; row["gain_lo"], row["gain_hi"] = ci(bo - prev[k][1])
                    if rung >= 3:
                        pv = [metric(target, yt, preds[f"pseudo{d}"])[k] for d in range(5)]; pb = np.mean([boot(target, yt, preds[f"pseudo{d}"], k) for d in range(5)], axis=0)
                        row["pseudo_value"] = float(np.mean(pv)); row["real_minus_pseudo"] = v - float(np.mean(pv)); row["rmp_lo"], row["rmp_hi"] = ci(bo - pb)
                    L.append(row); prev[k] = (v, bo)
                print(kind, link, target, model, rung, flush=True)
    pd.DataFrame(L).to_csv(f"{res}/ladder.tsv", sep="\t", index=False, float_format="%.5f")
    # drop-one-TF, rung 3, level target
    tfs = [l.strip() for l in open("/work/users/d/y/dyy12/XuLab/regulatome/m1_binding/v2/tfs.txt") if l.strip()]; FT = pd.read_csv("/work/users/d/y/dyy12/XuLab/regulatome/m6_improve/repro_v1/per_tf_floor_test.tsv", sep="\t").set_index("tf")
    D = []; yt = Y["level"][TE]
    for model in ("enet", "lgb"):
        p = P[f"level|{model}|rung3"]; full = {}
        for c in ["real"] + [f"pseudo{d}" for d in range(5)]:
            X = design(base, F, c, 3); m = fit(model, "level", p, X[TR], Y["level"][TR], X[VA], Y["level"][VA]); full[c] = (X, r2(yt, predict(m, model, "level", X[TE])))
        for j, t in enumerate(tfs):
            keep = np.ones(full["real"][0].shape[1], bool); keep[[7 + j, 7 + NT + j]] = False; ch = {}
            for c in full:
                X = full[c][0][:, keep]; m = fit(model, "level", p, X[TR], Y["level"][TR], X[VA], Y["level"][VA]); ch[c] = full[c][1] - r2(yt, predict(m, model, "level", X[TE]))
            D.append(dict(label=LAB, kind=kind, link=link, model=model, tf=t, cells=int(FT.cells.get(t, -1)), tertile=FT.tertile.get(t, ""), claim_tier=bool(FT.d.get(t, 0) >= .04), drop_one_change_r2=ch["real"], pseudo_drop_one_change_r2=float(np.mean([ch[f"pseudo{d}"] for d in range(5)])), full_r2=full["real"][1]))
            if j % 10 == 0: print("dropone", model, j, flush=True)
    pd.DataFrame(D).to_csv(f"{res}/dropone.tsv", sep="\t", index=False, float_format="%.6f")
