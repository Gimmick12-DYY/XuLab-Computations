# m32: per-test-TF table from the saved test predictions. Descriptive.
import numpy as np, pandas as pd
from scipy.stats import rankdata
RG = "/work/users/d/y/dyy12/XuLab/regulatome"; OUT = f"{RG}/m32_ml_tf_split/out"
F3 = set(pd.read_csv(f"{RG}/m27_kd_quality_subset/out/kd_quality_per_tf.tsv", sep="\t").query("F3 == True").tf)
TF = pd.read_csv(f"{RG}/m32_ml_tf_split/tf_folds.tsv", sep="\t").set_index("tf")
ROWS = {}; POOL = []
for model in ("lgb", "enet"):
    for fs in ("full", "baseline", "hitag_only"):
        d = np.load(f"{OUT}/{fs}_{model}_test_pred.npz", allow_pickle=True); TT = list(d["tfs"]); p, y, tf = d["pred"], d["y"], d["tf"]
        sps, hits = [], 0
        for i, t in enumerate(TT):
            m = (tf == i) & np.isfinite(y) & np.isfinite(p); pi, yi = p[m], y[m]
            s = float(np.corrcoef(rankdata(pi), rankdata(yi))[0, 1]) if np.std(pi) > 0 else 0.0; resp = np.abs(yi) >= 2; top = np.argsort(-np.abs(pi))[:200]; h = int(resp[top].sum())
            r = ROWS.setdefault(t, dict(label="descriptive, unseen-TF test", TF=t, test_fold=int(TF.tf_fold[t]), reproducible_kd_F3=t in F3, n_genes=int(m.sum()), n_responding_genes=int(resp.sum()), top200_expected=round(200 * resp.mean(), 1)))
            r[f"spearman_{fs}_{model}"] = s; r[f"top200_hits_{fs}_{model}"] = h; sps.append(s); hits += h
        POOL.append(dict(label="descriptive, unseen-TF test", model=model, features=fs, n_test_tfs=len(sps), mean_spearman=float(np.mean(sps)), median_spearman=float(np.median(sps)), n_tfs_spearman_gt_0=int((np.array(sps) > 0).sum()), total_top200_hits=hits))
T = pd.DataFrame(ROWS.values()); T["total_top200_expected"] = T.top200_expected
T.sort_values("spearman_full_lgb", ascending=False).to_csv(f"{OUT}/per_test_tf.tsv", sep="\t", index=False, float_format="%.4f")
P = pd.DataFrame(POOL); P.loc[:, "total_top200_expected"] = T.top200_expected.sum(); P.to_csv(f"{OUT}/pooled_test.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 250); print(P.drop(columns="label").round(4).to_string(index=False))
print(T.sort_values("spearman_full_lgb", ascending=False)[["TF", "reproducible_kd_F3", "n_genes", "n_responding_genes", "spearman_full_lgb", "spearman_baseline_lgb", "spearman_hitag_only_lgb", "top200_hits_full_lgb", "top200_hits_baseline_lgb", "top200_hits_hitag_only_lgb", "top200_expected"]].round(3).to_string(index=False))
