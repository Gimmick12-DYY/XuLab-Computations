# m32 (SPEC.md): m18 ML v2 model evaluated on unseen TFs (TF folds, all genes). usage: m32_fit.py <full|baseline|hitag_only> <lgb|enet>
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
RG = "/work/users/d/y/dyy12/XuLab/regulatome"; M18 = f"{RG}/m18_ml_v2"; F = f"{M18}/features"; D = f"{RG}/m32_ml_tf_split"; OUT = f"{D}/out"; SEED = 20261112
FSET, MODEL = sys.argv[1], sys.argv[2]; NAME = f"{FSET}_{MODEL}"
BLOCKS = {"full": ["A", "E", "R", "H", "C", "B", "BC", "D", "F", "G"], "baseline": ["A", "E", "R", "H", "C"], "hitag_only": ["B", "BC", "D", "F", "G"]}[FSET]
IX = np.load(f"{F}/index.npz", allow_pickle=True); TT = list(IX["tfs"].astype(str)); ug = IX["genes"].astype(str); NG = len(ug)
Rz = np.load(f"{RG}/m9_abc_diagnosis/work/responses_arm2.npz", allow_pickle=True); assert list(Rz["tfs"].astype(str)) == TT and (Rz["genes"].astype(str) == ug).all()
Z = Rz["r1"].astype(float).copy(); OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}; gix = {g: k for k, g in enumerate(ug)}
for i, t in enumerate(TT):
    for o in OWN.get(t, [t]):
        if o in gix: Z[i, gix[o]] = np.nan
TFOLD = pd.read_csv(f"{D}/tf_folds.tsv", sep="\t").set_index("tf").tf_fold.reindex(TT).values.astype(int)
ld = lambda n: np.load(f"{F}/{n}.npz", allow_pickle=True)
gA, gF, gD = ld("gene_A"), ld("gene_F"), ld("gene_D_real"); pr = ld("pair_real"); pn = list(pr["names"]); prX = pr["X"].reshape(len(TT), NG, len(pn))
RNA = pd.read_csv(f"{F}/rna_ntc_halves.tsv.gz", sep="\t").set_index("gene"); ALIAS = {"HP1": "CBX5", "ZUFSP": "ZUP1"}
rna_m = RNA["mean_log1p_cp10k_h0"].reindex(ug).fillna(0).values; rna_d = RNA["detect_h0"].reindex(ug).fillna(0).values
def block_cols(b, i):
    if b == "A": return gA["X"], list(gA["names"])
    if b == "R": own = float(RNA["mean_log1p_cp10k_h0"].get(ALIAS.get(TT[i], TT[i]), 0.0)); return np.column_stack([rna_m, rna_d, np.full(NG, own)]), ["R_mean_log1p_cp10k", "R_detect", "R_tf_own_expr"]
    if b == "H": return np.zeros((NG, 0)), []
    if b == "D": return gD["X"], list(gD["names"])
    if b == "F": ks = [k for k, n in enumerate(pn) if n.startswith("F_")]; return np.column_stack([gF["X"], prX[i][:, ks]]), list(gF["names"]) + [pn[k] for k in ks]
    if b == "E": ks = [k for k, n in enumerate(pn) if n in ("E_cells", "E_depth_tertile")]; return prX[i][:, ks], [pn[k] for k in ks]  # E_kd_pct dropped: needs the test TF's knockdown
    pref = {"C": "C_", "B": "B_", "BC": "BC_", "G": "G_"}[b]; ks = [k for k, n in enumerate(pn) if n.startswith(pref) and not (b == "B" and n.startswith("BC_"))]; return prX[i][:, ks], [pn[k] for k in ks]
XS, YS, TF_ = [], [], []
for i in range(len(TT)):
    M, N = [], []
    for b in BLOCKS: m, n = block_cols(b, i); M.append(m); N += n
    XS.append(np.column_stack(M).astype(np.float32)); YS.append(Z[i]); TF_.append(np.full(NG, i))
X = np.vstack(XS); Y = np.concatenate(YS); TFr = np.concatenate(TF_); Gr = np.tile(np.arange(NG), len(TT)); FOLD = TFOLD[TFr]; ok = np.isfinite(Y); del XS
print(NAME, "rows", len(Y), "features", X.shape[1], flush=True)
mu = pd.Series(Y[ok]).groupby(TFr[ok]).mean(); Yc = Y - pd.Series(TFr).map(mu).fillna(0).values  # training TFs centred by their own mean (test TFs are scored by Spearman, unaffected)
def sp_rows(pv, rows):
    out = {}; idx = np.flatnonzero(rows)
    for i in np.unique(TFr[idx]):
        m = TFr[idx] == i
        if m.sum() > 10 and np.std(pv[m]) > 0: out[i] = float(np.corrcoef(rankdata(pv[m]), rankdata(Y[idx][m]))[0, 1])
    return out
import lightgbm as lgb
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
GRID_L = [dict(num_leaves=a, min_data_in_leaf=b) for a in (15, 63) for b in (50, 200)]; GRID_E = [dict(alpha=a, l1_ratio=r) for a in (1e-3, 1e-2, 1e-1) for r in (.1, .5, .9)]
NT = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
def lgbp(g): return dict(objective="l2", learning_rate=.05, feature_fraction=.8, bagging_fraction=.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED, num_threads=NT, **g)
OOF = np.full(len(Y), np.nan); INFO = []
for k in range(5):
    te = (FOLD == k); va = (FOLD == (k + 1) % 5) & ok; tr = ~te & ~(FOLD == (k + 1) % 5) & ok; best = None
    for g in (GRID_L if MODEL == "lgb" else GRID_E):
        if MODEL == "lgb":
            m = lgb.train(lgbp(g), lgb.Dataset(X[tr], Yc[tr]), 2000, valid_sets=[lgb.Dataset(X[va], Yc[va])], callbacks=[lgb.early_stopping(50, verbose=False)]); pv = m.predict(X[va], num_iteration=m.best_iteration); nit = max(m.best_iteration, 10)
        else:
            sc = StandardScaler().fit(X[tr]); m = ElasticNet(max_iter=3000, tol=1e-3, **g).fit(np.nan_to_num(sc.transform(X[tr])), Yc[tr]); pv = m.predict(np.nan_to_num(sc.transform(X[va]))); nit = None
        s = sp_rows(pv, va); s = np.mean(list(s.values())) if s else -9
        if best is None or s > best[0]: best = (s, g, nit)
    trva = tr | va
    if MODEL == "lgb": m = lgb.train(lgbp(best[1]), lgb.Dataset(X[trva], Yc[trva]), best[2]); OOF[te] = m.predict(X[te])
    else: sc = StandardScaler().fit(X[trva]); m = ElasticNet(max_iter=3000, tol=1e-3, **best[1]).fit(np.nan_to_num(sc.transform(X[trva])), Yc[trva]); OOF[te] = m.predict(np.nan_to_num(sc.transform(X[te])))
    INFO.append(dict(test_fold=k, test_tfs=",".join(TT[i] for i in np.unique(TFr[te])), grid=json.dumps(best[1]), n_iter=best[2], inner_val_mean_spearman=best[0])); print("fold", k, best, flush=True)
np.savez_compressed(f"{OUT}/{NAME}_test_pred.npz", pred=OOF.astype(np.float32), y=Y.astype(np.float32), tf=TFr, gene=Gr, tfs=np.array(TT), genes=ug)
pd.DataFrame(INFO).to_csv(f"{OUT}/{NAME}_folds.tsv", sep="\t", index=False); print("done")
