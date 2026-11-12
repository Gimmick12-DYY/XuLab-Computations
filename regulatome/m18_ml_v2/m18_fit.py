# m18_ml_v2 fitting (SPEC + Amendments 1-3). usage: m18_fit.py <config>   (configs listed in CONFIGS; m18_fit.py list prints them)
# Every preprocessing step is fitted on training folds only:
#   target centring  -> centre(): per-TF mean over training genes, applied to all genes            [line tagged LEAK-GUARD-1]
#   feature scaling  -> enet only, StandardScaler fitted on (inner-)training rows                   [LEAK-GUARD-2]
#   tuning/early stop -> inner validation = fold (k+1) % 5 inside the training folds               [LEAK-GUARD-3]
#   percentile features are genome-wide and target-free (m18_features.py)
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
RG = "/work/users/d/y/dyy12/XuLab/regulatome"; O = f"{RG}/m18_ml_v2"; F = f"{O}/features"; OUT = f"{O}/out/runs"; os.makedirs(OUT, exist_ok=True)
SEED = 20261112; PRIMARY = ["CTCF", "DNMT1", "E4F1", "MEF2A", "NRF1", "PA2G4", "RBBP7", "RBPJ", "SAFB", "SETDB1"]
LADDER = [("r1_A", ["A", "E"]), ("r2_AR", ["A", "E", "R"]), ("r3_ARH", ["A", "E", "R", "H"]), ("r4_ARHC", ["A", "E", "R", "H", "C"]), ("r5_B", ["A", "E", "R", "H", "C", "B", "BC"]),
          ("r6_D", ["A", "E", "R", "H", "C", "B", "BC", "D"]), ("r7_F", ["A", "E", "R", "H", "C", "B", "BC", "D", "F"]), ("r8_G_full", ["A", "E", "R", "H", "C", "B", "BC", "D", "F", "G"])]
CONFIGS = {}
for tset in ("primary", "all"):
    for model in ("lgb", "enet"):
        for rung, blocks in LADDER:
            CONFIGS[f"{tset}_{model}_{rung}"] = dict(tset=tset, model=model, blocks=blocks, variant="real", half=0)
for v in ("swap", "pseudo0", "pseudo1", "pseudo2"):
    for model in ("lgb", "enet"): CONFIGS[f"primary_{model}_full_{v}"] = dict(tset="primary", model=model, blocks=LADDER[-1][1], variant=v, half=0)
for model in ("lgb", "enet"):
    CONFIGS[f"primary_{model}_full_permtarget"] = dict(tset="primary", model=model, blocks=LADDER[-1][1], variant="real", half=0, permute=True)
    CONFIGS[f"primary_{model}_r2_AR_half1"] = dict(tset="primary", model=model, blocks=LADDER[1][1], variant="real", half=1)
    CONFIGS[f"primary_{model}_full_half1"] = dict(tset="primary", model=model, blocks=LADDER[-1][1], variant="real", half=1)
    for keepE in (True, False):
        CONFIGS[f"primary_{model}_full_byTF_{'keepE' if keepE else 'dropE'}"] = dict(tset="primary", model=model, blocks=LADDER[-1][1] if keepE else [b for b in LADDER[-1][1] if b != "E"], variant="real", half=0, bytf=True, drop_tf_identity=not keepE)
if sys.argv[1] == "list": print("\n".join(CONFIGS)); sys.exit()
cfg = CONFIGS[sys.argv[1]]; NAME = sys.argv[1]; rng = np.random.default_rng(SEED)
# ---- data
IX = np.load(f"{F}/index.npz", allow_pickle=True); TT = list(IX["tfs"].astype(str)); ug = IX["genes"].astype(str); NG = len(ug)
Rz = np.load(f"{RG}/m9_abc_diagnosis/work/responses_arm2.npz", allow_pickle=True); assert list(Rz["tfs"].astype(str)) == TT and (Rz["genes"].astype(str) == ug).all()
Z = Rz["r1"].astype(float).copy(); OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}; gix = {g: k for k, g in enumerate(ug)}
for i, t in enumerate(TT):
    for o in OWN.get(t, [t]):
        if o in gix: Z[i, gix[o]] = np.nan
TS = [t for t in (PRIMARY if cfg["tset"] == "primary" else TT) if t in TT]; tix = [TT.index(t) for t in TS]
fold = pd.read_csv(f"{O}/folds.tsv", sep="\t").set_index("gene").fold.reindex(ug).values.astype(int)
ld = lambda n: np.load(f"{F}/{n}.npz", allow_pickle=True)
gA, gF, gD = ld("gene_A"), ld("gene_F"), ld(f"gene_D_{cfg['variant']}"); pr = ld(f"pair_{cfg['variant']}"); pr_real = ld("pair_real")
RNA = pd.read_csv(f"{F}/rna_ntc_halves.tsv.gz", sep="\t").set_index("gene"); h = cfg["half"]; ALIAS = {"HP1": "CBX5", "ZUFSP": "ZUP1"}
rna_m = RNA[f"mean_log1p_cp10k_h{h}"].reindex(ug).fillna(0).values; rna_d = RNA[f"detect_h{h}"].reindex(ug).fillna(0).values
pn = list(pr["names"]); prX = pr["X"].reshape(len(TT), NG, len(pn)); prR = pr_real["X"].reshape(len(TT), NG, len(pn))
def block_cols(b, i):  # features of block b for TF row i -> (matrix NG x k, names)
    if b == "A": return gA["X"], list(gA["names"])
    if b == "R":
        own = float(RNA[f"mean_log1p_cp10k_h{h}"].get(ALIAS.get(TT[i], TT[i]), 0.0))
        return np.column_stack([rna_m, rna_d, np.full(NG, own)]), ["R_mean_log1p_cp10k", "R_detect", "R_tf_own_expr"]
    if b == "H": return np.zeros((NG, 0)), []
    if b == "D": return gD["X"], list(gD["names"])
    if b == "F":
        ks = [k for k, n in enumerate(pn) if n.startswith("F_")]; return np.column_stack([gF["X"], prX[i][:, ks]]), list(gF["names"]) + [pn[k] for k in ks]
    pref = {"C": "C_", "B": "B_", "BC": "BC_", "G": "G_", "E": "E_"}[b]; src = prR if b == "C" else prX  # motif block never changes with the binding variant
    ks = [k for k, n in enumerate(pn) if n.startswith(pref) and not (b == "B" and n.startswith("BC_"))]; return src[i][:, ks], [pn[k] for k in ks]
def design(i):
    M, N, B = [], [], []
    for b in cfg["blocks"]:
        m, n = block_cols(b, i)
        if cfg.get("drop_tf_identity") and b == "R": m, n = m[:, :2], n[:2]
        M.append(m); N += n; B += [b] * len(n)
    return np.column_stack(M).astype(np.float32), N, B
XS = []; YS = []; TF_ = []; G_ = []
for i in tix:
    Xi, names, blocks = design(i); XS.append(Xi); YS.append(Z[i]); TF_.append(np.full(NG, i)); G_.append(np.arange(NG))
X = np.vstack(XS); Y = np.concatenate(YS); TFr = np.concatenate(TF_); Gr = np.concatenate(G_); FOLD = fold[Gr]; ok = np.isfinite(Y)
print(NAME, "rows", len(Y), "features", X.shape[1], flush=True)
def centre(y, tr):  # LEAK-GUARD-1: per-TF mean over training rows only
    mu = pd.Series(y[tr]).groupby(TFr[tr]).mean(); return y - pd.Series(TFr).map(mu).fillna(0).values
def per_tf_sp(pred, rows):
    out = {}
    for i in np.unique(TFr[rows]):
        m = rows & (TFr == i) & ok
        if m.sum() > 10 and np.std(pred[m]) > 0: out[i] = float(np.corrcoef(rankdata(pred[m]), rankdata(Y[m]))[0, 1])
        elif m.sum() > 10: out[i] = 0.0
    return out
import lightgbm as lgb
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
GRID_L = [dict(num_leaves=a, min_data_in_leaf=b) for a in (15, 63) for b in (50, 200)]; GRID_E = [dict(alpha=a, l1_ratio=r) for a in (1e-3, 1e-2, 1e-1) for r in (.1, .5, .9)]
def fit_predict(tr, va, te, y):  # LEAK-GUARD-3: grid chosen on inner validation rows only, then refit on tr+va
    best = None
    for g in (GRID_L if cfg["model"] == "lgb" else GRID_E):
        if cfg["model"] == "lgb":
            p = dict(objective="l2", learning_rate=.05, feature_fraction=.8, bagging_fraction=.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED, num_threads=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)), **g)
            m = lgb.train(p, lgb.Dataset(X[tr], y[tr]), 2000, valid_sets=[lgb.Dataset(X[va], y[va])], callbacks=[lgb.early_stopping(50, verbose=False)]); pv = m.predict(X[va], num_iteration=m.best_iteration); nit = max(m.best_iteration, 10)
        else:
            sc = StandardScaler().fit(X[tr]); m = ElasticNet(max_iter=3000, tol=1e-3, **g).fit(np.nan_to_num(sc.transform(X[tr])), y[tr]); pv = m.predict(np.nan_to_num(sc.transform(X[va]))); nit = None  # LEAK-GUARD-2
        sv = per_tf_sp_rows(pv, va); sc_ = np.mean(list(sv.values())) if sv else -9
        if best is None or sc_ > best[0]: best = (sc_, g, nit)
    trva = tr | va
    if cfg["model"] == "lgb":
        p = dict(objective="l2", learning_rate=.05, feature_fraction=.8, bagging_fraction=.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED, num_threads=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)), **best[1])
        m = lgb.train(p, lgb.Dataset(X[trva], y[trva]), best[2]); return m.predict(X[te]), m, best
    sc = StandardScaler().fit(X[trva]); m = ElasticNet(max_iter=3000, tol=1e-3, **best[1]).fit(np.nan_to_num(sc.transform(X[trva])), y[trva]); return m.predict(np.nan_to_num(sc.transform(X[te]))), (m, sc), best
def per_tf_sp_rows(pv, va):  # Spearman per TF between predictions pv (for rows va, in order) and Y
    out = {}; idx = np.flatnonzero(va)
    for i in np.unique(TFr[idx]):
        m = TFr[idx] == i
        if m.sum() > 10 and np.std(pv[m]) > 0: out[i] = float(np.corrcoef(rankdata(pv[m]), rankdata(Y[idx][m]))[0, 1])
    return out
OOF = np.full(len(Y), np.nan); CONTRIB = None; INFO = []
splits = []
for k in range(5):
    if cfg.get("bytf"):
        for i in tix: splits.append((k, i))
    else: splits.append((k, None))
for k, hold in splits:
    te_g = FOLD == k; va_g = FOLD == (k + 1) % 5; tr_g = ~te_g & ~va_g
    te = te_g & ok & ((TFr == hold) if hold is not None else True); va = va_g & ok & ((TFr != hold) if hold is not None else True); tr = tr_g & ok & ((TFr != hold) if hold is not None else True)
    y = centre(Y.copy(), tr | va)
    if cfg.get("permute"):  # leak test: target permuted within TF inside the training folds only
        y = y.copy()
        for i in np.unique(TFr[tr | va]): m = np.flatnonzero((tr | va) & (TFr == i)); y[m] = y[rng.permutation(m)]
    pte, mdl, best = fit_predict(tr, va, te, y); OOF[te] = pte; INFO.append(dict(fold=k, held_tf=TT[hold] if hold is not None else "", grid=json.dumps(best[1]), n_iter=best[2], inner_val_sp=best[0]))
    if cfg["model"] == "lgb" and NAME.endswith("r8_G_full") and cfg["tset"] == "primary":
        c = mdl.predict(X[te], pred_contrib=True); CONTRIB = np.zeros((len(Y), c.shape[1])) if CONTRIB is None else CONTRIB; CONTRIB[te] = c
    print("fold", k, "held", hold, "inner", round(best[0], 4), best[1], flush=True)
sp = per_tf_sp(np.nan_to_num(OOF), np.isfinite(OOF)); FT = pd.read_csv(f"{RG}/m7_step2/splits/tf_folds.tsv", sep="\t").set_index("TF").ceiling
R = pd.DataFrame([dict(config=NAME, tf=TT[i], spearman_oof=v, reliability=FT.get(TT[i], np.nan)) for i, v in sp.items()]); R["frac_of_max_r"] = R.spearman_oof / np.sqrt(R.reliability.where(R.reliability > 0))
R.to_csv(f"{OUT}/{NAME}_per_tf.tsv", sep="\t", index=False, float_format="%.5f"); pd.DataFrame(INFO).to_csv(f"{OUT}/{NAME}_folds.tsv", sep="\t", index=False)
np.savez_compressed(f"{OUT}/{NAME}_oof.npz", oof=OOF.astype(np.float32), y=Y.astype(np.float32), tf=TFr, gene=Gr, names=np.array(names), blocks=np.array(blocks))
if CONTRIB is not None:
    bl = np.array(blocks + ["bias"]); S_ = pd.DataFrame({b: np.abs(CONTRIB[:, bl == b]).sum(1) for b in dict.fromkeys(bl) if b != "bias"}); S_["tf"] = [TT[i] for i in TFr]
    S_[np.isfinite(OOF)].groupby("tf").mean().to_csv(f"{OUT}/{NAME}_shap_block_mean_abs.tsv", sep="\t", float_format="%.5f")
    np.savez_compressed(f"{OUT}/{NAME}_contrib.npz", contrib=CONTRIB.astype(np.float32), names=np.array(names + ["bias"]), blocks=bl)
print(R.to_string(index=False)); print("mean", R.spearman_oof.mean())
