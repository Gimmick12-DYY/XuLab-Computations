# m7_step2 fit/tune code (train and validation chromosomes only). Never touches test-chromosome genes or held-out-TF knockdown: test scoring is score.py, run after the freeze.
# usage: m7.py <jobs.tsv> <line index>. A job is one fit: target (kd|expr), arm, centring, model, mode, cond, split, rung, kind, hp (grid index or "frozen"), phase (tune|final).
import json, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import rankdata
X = "/work/users/d/y/dyy12/XuLab"; R = f"{X}/regulatome"; O = f"{R}/m7_step2"; LINK = os.environ.get("M7_LINK", "powerlaw"); FROZEN = f"{O}/frozen/params_{LINK}.json"; JOB_PREFIX = "" if LINK == "powerlaw" else f"{LINK}_"
ARMS = {"arm1": "arm1_high_knockdown", "arm2": "arm2_all_cell"}; ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; panel = lambda t: ALIAS.get(t, t)
OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
GRID = {
    "enet": [dict(alpha=a, l1_ratio=l) for a in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2) for l in (.1, .5, .9)],
    "lgb_kd": [dict(num_leaves=n, min_data_in_leaf=m, feature_fraction=f) for n in (15, 31, 63) for m in (50, 200) for f in (.7, 1.0)],
    "lgb_expr": [dict(num_leaves=n, min_data_in_leaf=m, feature_fraction=f) for n in (7, 15, 31) for m in (10, 30) for f in (.7, 1.0)],
    "mlp": [dict(h=h, drop=d, emb=e) for h in (64, 128, 256) for d in (0.0, 0.2) for e in (8, 16)],
}
tfs = [l.strip() for l in open(f"{R}/m1_binding/v2/tfs.txt") if l.strip()]; TF_IDX = {t: i for i, t in enumerate(tfs)}
GS = pd.read_csv(f"{O}/splits/gene_split.tsv", sep="\t"); GENES = GS.gene.values; NG = len(GENES); GIDX = pd.Series(np.arange(NG), index=GENES)
FOLDS = pd.read_csv(f"{O}/splits/tf_folds.tsv", sep="\t"); PAIRS = pd.read_csv(f"{O}/pair_list.tsv", sep="\t"); NP = len(PAIRS)
TESTABLE = FOLDS.TF.tolist(); PTF = {t: i for i, t in enumerate(TESTABLE)}; PAIR_OF = {t: [i for i, (a, b) in enumerate(zip(PAIRS.a, PAIRS.b)) if t in (a, b)] for t in TESTABLE}
def load_kd(arm):
    d = np.load(f"{R}/m4_validation/kd_response/out_v2/{ARMS[arm]}/kd_response.npz", allow_pickle=True); q = pd.read_csv(f"{R}/m4_validation/kd_response/out_v2/{ARMS[arm]}/guide_qc.tsv", sep="\t")
    u = d["units"].tolist(); ui = {x: k for k, x in enumerate(u)}; uni = d["ntc_detect"] >= .1; g = d["genes"][uni].astype(str); Z = d["z"][:, uni].astype(float)
    kp = q[q.kept == 1]; prim = kp[kp.kind.isin(["all", "construct"])]; tf_ = sorted(prim.tf.unique())
    Pm = np.nan_to_num(np.vstack([np.nanmean(Z[[ui[x] for x in prim.unit[prim.tf == t]]], axis=0) for t in tf_]))  # raw per-TF z, no centring
    pts = sorted({panel(t) for t in tf_}); Rp = np.vstack([Pm[[i for i, t in enumerate(tf_) if panel(t) == p]].mean(0) for p in pts]); gi = pd.Series(np.arange(len(g)), index=g); gi = gi[~gi.index.duplicated(keep="last")]  # duplicate symbols: last occurrence wins, as the dict in m6_improve load()
    Zp = np.full((len(TESTABLE), NG), np.nan); ix = gi.reindex(GENES); ok = ix.notna().values; keep = ix[ok].astype(int).values
    for p, row in zip(pts, Rp):
        if p in PTF: Zp[PTF[p], ok] = row[keep]
    return Zp, np.array([np.isfinite(Zp[i]).any() for i in range(len(TESTABLE))]), d["ntc_mean"][uni].astype(float), g
def expr_target():
    # per-gene mean log1p(CP10k) over NTC cells, Arm 2 >= 10% detection genes, duplicate symbols last occurrence
    d = np.load(f"{R}/m4_validation/kd_response/out_v2/arm2_all_cell/kd_response.npz", allow_pickle=True); u = d["ntc_detect"] >= .1; s = pd.Series(d["ntc_mean"][u].astype(float), index=d["genes"][u].astype(str)); return s[~s.index.duplicated(keep="last")].reindex(GENES).values
class Feat:
    def __init__(s, kind):
        p = f"{O}/features/{kind}_{LINK}"; s.base = np.load(f"{p}/base.npz")["base"]; s.p = p; s.c = {}
    def cond(s, c):
        name = "real" if c in ("real", "base") else c
        if name not in s.c: s.c[name] = {k: v for k, v in np.load(f"{s.p}/{name}.npz").items()}
        return s.c[name]
def set_mask(name, arm=None):
    m = (GS.set == name).values.copy()
    if arm == "arm1": m &= GS.in_arm1.values
    elif arm == "arm2": m &= GS.in_arm2.values
    return m
def tf_lists(split, arm_tfs):
    if split == "gene": return list(arm_tfs), list(arm_tfs)
    k = int(split[4:]); hold = set(FOLDS.TF[FOLDS.fold == k]); return [t for t in arm_tfs if t not in hold], [t for t in arm_tfs if t in hold]
def kd_rows(tfl, genemask, Zp):
    gi = np.flatnonzero(genemask); ti = np.array([PTF[t] for t in tfl]); tt = np.repeat(ti, len(gi)); gg = np.tile(gi, len(ti)); own = np.zeros(len(tt), bool)
    for t in tfl:
        for o in OWN.get(t, [t]):
            if o in GIDX.index: own[(tt == PTF[t]) & (gg == GIDX[o])] = True
    keep = ~own & np.isfinite(Zp[tt, gg]); return tt[keep], gg[keep]
def targets(Zp, trTF, centring, arm_mask_train):
    # Offsets from train-chromosome genes only; per-gene offset from training TFs at train genes only and applied to train genes only (Amendment 1/2). tt: indices into TESTABLE.
    Y = Zp.copy(); tix = np.array([PTF[t] for t in trTF]); tr = np.flatnonzero(arm_mask_train)
    Y = Y - np.nanmean(Y[:, tr], axis=1, keepdims=True)
    gene_off = np.zeros(NG)
    if centring == "train":
        gene_off[tr] = np.nanmean(Y[np.ix_(tix, tr)], axis=0); Y = Y - gene_off[None, :]; Y = Y - np.nanmean(Y[:, tr], axis=1, keepdims=True)
    return Y
def rank_sp(a, b):
    if np.std(a) == 0 or np.std(b) == 0: return 0.0
    r = np.corrcoef(rankdata(a), rankdata(b))[0, 1]; return 0.0 if not np.isfinite(r) else float(r)
def per_tf_sp(tt, y, p):
    return {int(t): rank_sp(p[tt == t], y[tt == t]) for t in np.unique(tt)}
def r2(y, p): return 1 - float(((y - p) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
# ---- KD design
def kd_design(F, cond, tt, gg, bind=True):
    n = len(tt); b = F.base[gg]; D = F.cond(cond)
    if not bind: return b, None, None
    tix79 = np.array([TF_IDX[TESTABLE[t]] for t in tt]); own = np.column_stack([D["prom"][tix79, gg], D["dist"][tix79, gg]]); pair = np.zeros((n, NP), np.float32)
    for t in np.unique(tt):
        m = np.flatnonzero(tt == t)
        for p in PAIR_OF[TESTABLE[t]]: pair[m, p] = D["pair"][p, gg[m]]
    return b, own, pair
def enet_x(b, own, pair, tt, per_tf, expr=None):
    n = len(tt); nb = b.shape[1]; dense = [b]
    if own is not None: dense += [own]
    if pair is not None: dense += [pair]
    if expr is not None: dense += [expr[:, None]]
    D = np.hstack(dense); blocks = [sp.csr_matrix(D)]
    if per_tf:
        q = np.hstack([b, own]) if own is not None else b; k = q.shape[1]; ro = np.repeat(np.arange(n), k); co = (np.repeat(tt, k) * k) + np.tile(np.arange(k), n)
        blocks.append(sp.csr_matrix((q.ravel(), (ro, co)), shape=(n, len(TESTABLE) * k)))
    return sp.hstack(blocks, format="csc")
def lgb_x(b, own, pair, tt, per_tf, expr=None):
    c = [b];
    if own is not None: c.append(own)
    if pair is not None: c.append(pair)
    if expr is not None: c.append(expr[:, None])
    if per_tf: c.append(tt[:, None].astype(np.float32))
    M = np.hstack(c); return M, ([M.shape[1] - 1] if per_tf else [])
def expr_x(F, cond, rung, gg):
    b = F.base[gg][:, :5 if rung == 1 else 7]
    if rung <= 2: return b, None
    D = F.cond(cond); binding = np.hstack([D["prom"][:, gg].T, D["dist"][:, gg].T]);
    if rung == 4: binding = np.hstack([binding, D["pair"][:, gg].T])
    return b, binding
EXP_T = None
# ---- MLP
def mlp_cls():
    import torch, torch.nn as nn
    class Net(nn.Module):
        def __init__(s, n_base, n_extra_kd, n_extra_expr, emb, h, drop, n_tf):
            super().__init__(); s.trunk = nn.Sequential(nn.Linear(n_base, h), nn.ReLU(), nn.Dropout(drop)); s.e = nn.Embedding(n_tf, emb)
            s.kd = nn.Sequential(nn.Linear(h + emb + n_extra_kd, h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h, 1)); s.ex = nn.Sequential(nn.Linear(h + n_extra_expr, h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h, 1)); s.emb_mean = None
        def f_kd(s, b, xk, t, mean_emb=False):
            h = s.trunk(b); em = s.emb_mean.expand(len(b), -1) if mean_emb else s.e(t); return s.kd(torch.cat([h, em, xk], 1)).squeeze(-1)
        def f_ex(s, b, xe):
            h = s.trunk(b); return s.ex(torch.cat([h, xe], 1)).squeeze(-1)
    return Net, torch
def fit_mlp(job, hp, Xtr, Xv, ytr, yv, expr=None, per_tf=True, target="kd", n_tf_idx=None):
    Net, torch = mlp_cls(); torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))); torch.manual_seed(0); rng = np.random.default_rng(0)
    bTr, kTr, eTr, tTr = Xtr; bV, kV, eV, tV = Xv; bs = 8192 if target == "kd" else 256
    nb = bTr.shape[1]; n_k = kTr.shape[1] if target == "kd" or expr is not None else 0; n_e = eTr.shape[1] if eTr is not None else 0
    if target == "expr": n_e = eTr.shape[1]
    net = Net(nb, n_k, n_e, hp["emb"], hp["h"], hp["drop"], len(TESTABLE)); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    T = lambda a: torch.tensor(a, dtype=torch.float32); TL = lambda a: torch.tensor(a, dtype=torch.long)
    bTr_, bV_ = T(bTr), T(bV); ytr_, yv_ = T(ytr), T(yv); kTr_, kV_ = T(kTr) if kTr is not None else None, T(kV) if kV is not None else None; eTr_, eV_ = T(eTr) if eTr is not None else None, T(eV) if eV is not None else None; tTr_, tV_ = TL(tTr) if tTr is not None else None, TL(tV) if tV is not None else None
    # joint: expression rows
    jb = None
    if expr is not None: jb = {k: T(v) if k != "y" else T(v) for k, v in expr.items()}
    best = 1e18; best_state = None; wait = 0; n = len(ytr); mxe = 150 if target == "kd" else 400; pat = 10 if target == "kd" else 25
    for ep in range(mxe):
        net.train(); o = rng.permutation(n)
        for i in range(0, n, bs):
            ix = torch.tensor(o[i:i + bs]); opt.zero_grad()
            if target == "kd":
                loss = ((net.f_kd(bTr_[ix], kTr_[ix], tTr_[ix]) - ytr_[ix]) ** 2).mean()
                if jb is not None:
                    jx = torch.randint(0, len(jb["y"]), (min(256, len(jb["y"])),)); loss = loss + ((net.f_ex(jb["b"][jx], jb["x"][jx]) - jb["y"][jx]) ** 2).mean()
            else: loss = ((net.f_ex(bTr_[ix], eTr_[ix]) - ytr_[ix]) ** 2).mean()
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = net.f_kd(bV_, kV_, tV_) if target == "kd" else net.f_ex(bV_, eV_); mse = float(((pv - yv_) ** 2).mean())
        if mse < best - 1e-6: best, wait = mse, 0; best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else: wait += 1
        if wait >= pat: break
    net.load_state_dict(best_state); net.eval()
    if target == "kd":
        idx = sorted({int(x) for x in tTr}); net.emb_mean = net.e(torch.tensor(idx)).mean(0).detach()
    return net, ep + 1, best
def save_mlp(net, path, meta):
    import torch; torch.save({"state": net.state_dict(), "emb_mean": net.emb_mean, "meta": meta}, path)
def mlp_predict_val(net, X, target):
    import torch
    b, k, e, t = X; T = lambda a: torch.tensor(a, dtype=torch.float32)
    with torch.no_grad():
        if target == "kd": return net.f_kd(T(b), T(k), torch.tensor(t, dtype=torch.long)).numpy()
        return net.f_ex(T(b), T(e)).numpy()
# ---- jobs
def params_for(job):
    if str(job.hp) != "frozen":
        g = GRID["enet"] if job.model == "enet" else (GRID["mlp"] if job.model == "mlp" else GRID["lgb_kd" if job.target == "kd" else "lgb_expr"]); return g[int(job.hp)]
    fp = json.load(open(FROZEN));
    return fp[job_key_params(job)]
def job_key_params(job):
    cc = "real" if (job.target == "expr") or job.cond in ("real",) or str(job.cond).startswith(("pseudo", "mis")) else "base"
    if job.target == "expr": return f"expr|{job.model}|rung{int(job.rung)}"
    return f"kd|{job.model}|{job['mode']}|{cc}"
def out_dir(job):
    r = f"{O}/fits/{job.kind}_{LINK}/{job.phase}/";
    if job.target == "expr": return f"{r}expr/{job.model}_rung{int(job.rung)}_{job.cond}" + (f"_hp{job.hp}" if job.phase == "tune" else "")
    return f"{r}kd/{job.split}/{job.arm}_{job.centring}/{job.model}_{job['mode']}_{job.cond}" + (f"_hp{job.hp}" if job.phase == "tune" else "")
def expr_inputs_for_kd(kind, model, cond):
    if cond == "base": rung, ec = 2, "real"
    elif cond.startswith("pseudo"): rung, ec = 4, cond
    else: rung, ec = 4, "real"
    f = np.load(f"{O}/fits/{kind}_{LINK}/final/expr/{model}_rung{rung}_{ec}/oof.npz"); return f["g"], f["p"]
def fit_expr(job, F):
    p = params_for(job); out = out_dir(job); os.makedirs(out, exist_ok=True); y = expr_target()
    trm = set_mask("train", "arm2") & np.isfinite(y); vm = set_mask("val", "arm2") & np.isfinite(y)
    gtr, gv = np.flatnonzero(trm), np.flatnonzero(vm); b, x = expr_x(F, job.cond, int(job.rung), gtr); bv, xv = expr_x(F, job.cond, int(job.rung), gv); Xtr = np.hstack([b] + ([x] if x is not None else [])); Xv = np.hstack([bv] + ([xv] if xv is not None else []))
    res = dict(job=job.to_dict(), params=p); chrom = GS.chromosome.values[gtr]; uchr = sorted(set(chrom), key=lambda c: (len(c), c)); fold = {c: i % 5 for i, c in enumerate(uchr)}; fo = np.array([fold[c] for c in chrom])
    if job.model in ("enet", "lgb"):
        def fit(a, aa, va):
            if job.model == "enet":
                from sklearn.linear_model import ElasticNet
                return ElasticNet(alpha=p["alpha"], l1_ratio=p["l1_ratio"], max_iter=5000, tol=1e-4).fit(a, aa)
            import lightgbm as lgb
            ds = lgb.Dataset(a, label=aa, free_raw_data=False, params=dict(feature_pre_filter=False)); v = lgb.Dataset(va[0], label=va[1], reference=ds, free_raw_data=False, params=dict(feature_pre_filter=False))
            return lgb.train(dict(objective="l2", learning_rate=.05, num_leaves=p["num_leaves"], min_data_in_leaf=p["min_data_in_leaf"], feature_fraction=p["feature_fraction"], bagging_fraction=.8, bagging_freq=1, lambda_l2=1.0, num_threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")), seed=0, verbosity=-1), ds, 2000, valid_sets=[v], callbacks=[lgb.early_stopping(50, verbose=False)])
        m = fit(Xtr, y[gtr], (Xv, y[gv])); pv = m.predict(Xv); res["val_r2"] = r2(y[gv], pv); res["val_mse"] = float(((y[gv] - pv) ** 2).mean())
        if job.model == "enet": np.savez(f"{out}/model.npz", coef=m.coef_, intercept=m.intercept_); res["nnz"] = int((m.coef_ != 0).sum())
        else: m.save_model(f"{out}/model.txt"); res["best_iter"] = m.best_iteration
        if job.phase == "final":
            oof = np.full(len(gtr), np.nan)
            for k in range(5):
                a = fo != k; mk = fit(Xtr[a], y[gtr][a], (Xv, y[gv])); oof[fo == k] = mk.predict(Xtr[fo == k])
            # genes on train/val chromosomes outside the Arm 2 gene universe (Arm 1-only genes): out-of-sample from the full-train model, for the KD expression feature
            ext = np.flatnonzero((GS.set.isin(["train", "val"]).values) & ~GS.in_arm2.values); bx, xx = expr_x(F, job.cond, int(job.rung), ext); Xx = np.hstack([bx] + ([xx] if xx is not None else [])); px = m.predict(Xx)
            np.savez(f"{out}/oof.npz", g=np.concatenate([gtr, gv, ext]), p=np.concatenate([oof, pv, px]).astype(np.float32))
    else:  # mlp
        m_ = y[gtr].mean(); s_ = y[gtr].std(); ys, yv = (y[gtr] - m_) / s_, (y[gv] - m_) / s_; e_tr = x if x is not None else np.zeros((len(gtr), 0), np.float32); e_v = xv if xv is not None else np.zeros((len(gv), 0), np.float32)
        net, ep, best = fit_mlp(job, p, (b, None, e_tr, None), (bv, None, e_v, None), ys, yv, target="expr"); pv = mlp_predict_val(net, (bv, None, e_v, None), "expr") * s_ + m_; res["val_r2"] = r2(y[gv], pv); res["epochs"] = ep
        save_mlp(net, f"{out}/model.pt", dict(n_base=b.shape[1], n_extra_expr=e_tr.shape[1], hp=p, y_mean=float(m_), y_sd=float(s_)))
    json.dump(res, open(f"{out}/val.json", "w"), indent=1, default=str); print(res, flush=True)
def fit_kd(job, F):
    arm = job.arm; Zp, _has, _, _ = load_kd(arm); arm_tfs = FOLDS.TF[FOLDS.arm1_in].tolist() if arm == "arm1" else FOLDS.TF.tolist(); arm_tfs = [t for t in arm_tfs if np.isfinite(Zp[PTF[t]]).any()]
    trTF, _ = tf_lists(job.split, arm_tfs); by_gene = job.split == "gene"; p = params_for(job); out = out_dir(job); os.makedirs(out, exist_ok=True)
    trm, vm = set_mask("train", arm), set_mask("val", arm); Y = targets(Zp, trTF, job.centring, trm); tt, gg = kd_rows(trTF, trm, Zp); tv, gv = kd_rows(trTF, vm, Zp)
    bind = job.cond != "base"; b, own, pair = kd_design(F, job.cond, tt, gg, bind); bv, ownv, pairv = kd_design(F, job.cond, tv, gv, bind); y, yv = Y[tt, gg], Y[tv, gv]
    res = dict(job=job.to_dict(), params=p, n_train=int(len(y)), n_val=int(len(yv))); per_tf = by_gene; expr, exprv = None, None
    if job.model in ("enet", "lgb") and job['mode'] == "both":
        eg, ep_ = expr_inputs_for_kd(job.kind, job.model, job.cond); e_map = pd.Series(ep_, index=eg); sm = e_map.reindex(GS.index[set_mask("train", "arm2")]); m, s = float(sm.mean()), float(sm.std()); ev = pd.Series(ep_, index=eg)
        expr = ((ev.reindex(gg).values - m) / s).astype(np.float32); exprv = ((ev.reindex(gv).values - m) / s).astype(np.float32); assert np.isfinite(expr).all() and np.isfinite(exprv).all()
    if job.model == "enet":
        from sklearn.linear_model import ElasticNet
        Xtr = enet_x(b, own, pair, tt, per_tf, expr); Xv = enet_x(bv, ownv, pairv, tv, per_tf, exprv); m = ElasticNet(alpha=p["alpha"], l1_ratio=p["l1_ratio"], max_iter=3000, tol=1e-3).fit(Xtr, y); pv = m.predict(Xv)
        np.savez(f"{out}/model.npz", coef=m.coef_, intercept=m.intercept_, per_tf=per_tf); res["nnz"] = int((m.coef_ != 0).sum()); res["n_iter"] = int(m.n_iter_)
    elif job.model == "lgb":
        import lightgbm as lgb
        Xtr, cat = lgb_x(b, own, pair, tt, per_tf, expr); Xv, _ = lgb_x(bv, ownv, pairv, tv, per_tf, exprv); ds = lgb.Dataset(Xtr, label=y, categorical_feature=cat, free_raw_data=False, params=dict(feature_pre_filter=False)); v = lgb.Dataset(Xv, label=yv, reference=ds, free_raw_data=False, params=dict(feature_pre_filter=False))
        m = lgb.train(dict(objective="l2", learning_rate=.05, num_leaves=p["num_leaves"], min_data_in_leaf=p["min_data_in_leaf"], feature_fraction=p["feature_fraction"], bagging_fraction=.8, bagging_freq=1, lambda_l2=1.0, num_threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")), seed=0, verbosity=-1), ds, 2000, valid_sets=[v], callbacks=[lgb.early_stopping(50, verbose=False)]); pv = m.predict(Xv); m.save_model(f"{out}/model.txt"); res["best_iter"] = m.best_iteration; res["per_tf"] = per_tf
    else:  # mlp: kd only or joint with expression on the same trunk
        k = np.hstack([own, pair]) if bind else np.zeros((len(y), 0), np.float32); kv = np.hstack([ownv, pairv]) if bind else np.zeros((len(yv), 0), np.float32); joint = None; tTr, tVv = tt, tv
        if job['mode'] == "both":
            rung = 4 if bind else 2; gtr = np.flatnonzero(set_mask("train", "arm2")); eb, ex = expr_x(F, job.cond if bind else "real", rung, gtr); s = expr_target()[gtr]
            ok = np.isfinite(s); sm, ss = float(s[ok].mean()), float(s[ok].std()); ex_ = ex[ok] if ex is not None else np.zeros((ok.sum(), 0), np.float32); joint = dict(b=F.base[gtr[ok]], x=ex_, y=(s[ok] - sm) / ss); res["expr_y_mean"], res["expr_y_sd"] = sm, ss
            e_x = np.zeros((1, ex_.shape[1]), np.float32); Xtr = (b, k, e_x.repeat(1, 0), tTr); Xv = (bv, kv, e_x, tVv)
        else: Xtr = (b, k, None, tTr); Xv = (bv, kv, None, tVv)
        net, ep, best = fit_mlp(job, p, Xtr, Xv, y, yv, expr=joint, per_tf=per_tf, target="kd"); pv = mlp_predict_val(net, Xv, "kd"); res["epochs"] = ep; res["val_mse"] = best
        save_mlp(net, f"{out}/model.pt", dict(n_base=b.shape[1], n_extra_kd=k.shape[1], n_extra_expr=(joint["x"].shape[1] if joint is not None else 0), hp=p, by_gene=by_gene))
    # validation tuning metric: mean per-TF Spearman over validation-chromosome genes of the training TFs
    sp_val = per_tf_sp(tv, yv, pv); res["val_spearman_mean"] = float(np.mean(list(sp_val.values()))); res["val_mse"] = float(((yv - pv) ** 2).mean())
    json.dump(res, open(f"{out}/val.json", "w"), indent=1, default=str); print({k: v for k, v in res.items() if k != "job"}, flush=True)
if __name__ == "__main__":
    jobs = pd.read_csv(sys.argv[1], sep="\t", dtype=str, keep_default_na=False); job = jobs.iloc[int(sys.argv[2])]; t0 = time.time(); print(job.to_dict(), flush=True); F = Feat(job.kind)
    if job.target == "expr": fit_expr(job, F)
    else: fit_kd(job, F)
    print("done", round(time.time() - t0), "s", flush=True)
