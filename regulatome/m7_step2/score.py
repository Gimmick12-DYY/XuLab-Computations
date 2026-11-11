# m7_step2 test scoring. Run ONCE, after the freeze commit (scripts, frozen/params.json, Amendment 3). Reads test-chromosome genes and held-out-TF knockdown.
# usage: score.py <raw|imp>  -> results/<kind>/per_tf_long.tsv, expr_long.tsv. summarise.py builds the final tables from those.
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m7
from m7 import O, LINK, GS, GENES, NG, TESTABLE, PTF, FOLDS, TF_IDX, per_tf_sp, rank_sp, r2, kd_design, kd_rows, enet_x, lgb_x, expr_x, tf_lists, set_mask, load_kd, Feat, out_dir, mlp_cls, expr_inputs_for_kd
import lightgbm as lgb
kind = sys.argv[1]; res = f"{O}/results/{kind}_{LINK}"; os.makedirs(res, exist_ok=True); F = Feat(kind)
assert os.path.exists(f"{O}/FROZEN.txt")
def mlp_load(path):
    import torch; Net, torch = mlp_cls(); ck = torch.load(path, weights_only=False); m = ck["meta"]; hp = m["hp"]
    net = Net(m["n_base"], m.get("n_extra_kd", 0), m.get("n_extra_expr", 0), hp["emb"], hp["h"], hp["drop"], len(TESTABLE)); net.load_state_dict(ck["state"]); net.emb_mean = ck["emb_mean"]; net.eval(); return net, m, ck
def expr_model_predict(job_dir, model, rung, cond, gg):
    b, x = expr_x(F, cond, rung, gg); X_ = np.hstack([b] + ([x] if x is not None else []))
    if model == "enet": m = np.load(f"{job_dir}/model.npz"); return X_ @ m["coef"] + float(m["intercept"])
    if model == "lgb": return lgb.Booster(model_file=f"{job_dir}/model.txt").predict(X_)
    import torch; net, m, _ = mlp_load(f"{job_dir}/model.pt"); T = lambda a: torch.tensor(a, dtype=torch.float32); e = x if x is not None else np.zeros((len(b), 0), np.float32)
    with torch.no_grad(): return (net.f_ex(T(b), T(e)).numpy()) * m["y_sd"] + m["y_mean"]
def kd_predict(job, Zp, tt, gg):
    d = out_dir(job); bind = job.cond != "base"; b, own, pair = kd_design(F, job.cond, tt, gg, bind); per_tf = job.split == "gene"; expr = None
    if job.model in ("enet", "lgb") and job['mode'] == "both":
        eg, ep_ = expr_inputs_for_kd(job.kind, job.model, job.cond); sm = pd.Series(ep_, index=eg).reindex(GS.index[set_mask("train", "arm2")]); me, se = float(sm.mean()), float(sm.std())
        rung, ec = (2, "real") if job.cond == "base" else ((4, job.cond) if job.cond.startswith("pseudo") else (4, "real")); uni = np.unique(gg); pr = expr_model_predict(f"{O}/fits/{job.kind}_{LINK}/final/expr/{job.model}_rung{rung}_{ec}", job.model, rung, ec, uni); expr = ((pd.Series(pr, index=uni).reindex(gg).values - me) / se).astype(np.float32)
    if job.model == "enet": m = np.load(f"{d}/model.npz"); return enet_x(b, own, pair, tt, per_tf, expr) @ m["coef"] + float(m["intercept"])
    if job.model == "lgb": return lgb.Booster(model_file=f"{d}/model.txt").predict(lgb_x(b, own, pair, tt, per_tf, expr)[0])
    import torch; net, m, _ = mlp_load(f"{d}/model.pt"); T = lambda a: torch.tensor(a, dtype=torch.float32); k = np.hstack([own, pair]) if bind else np.zeros((len(b), 0), np.float32)
    with torch.no_grad():
        if per_tf: return net.f_kd(T(b), T(k), torch.tensor(tt, dtype=torch.long)).numpy()
        return net.f_kd(T(b), T(k), torch.tensor(tt, dtype=torch.long), mean_emb=True).numpy()
rows = []; KD = {a: load_kd(a) for a in ("arm1", "arm2")}; Refs = np.load(f"{O}/features/raw_powerlaw/refs.npz"); jobs = pd.read_csv(f"{O}/jobs/{m7.JOB_PREFIX}{kind}_kd_final.tsv", sep="\t", dtype=str, keep_default_na=False)
for i, job in jobs.iterrows():
    arm = job.arm; Zp = KD[arm][0]; arm_tfs = FOLDS.TF[FOLDS.arm1_in].tolist() if arm == "arm1" else FOLDS.TF.tolist(); arm_tfs = [t for t in arm_tfs if np.isfinite(Zp[PTF[t]]).any()]; test_tfs = tf_lists(job.split, arm_tfs)[1]
    tt, gg = kd_rows(test_tfs, set_mask("test", arm), Zp); p = kd_predict(job, Zp, tt, gg); y = Zp[tt, gg]
    for t, s in per_tf_sp(tt, y, p).items(): rows.append(dict(kind=kind, link=LINK, split=job.split, arm=arm, centring=job.centring, model=job.model, mode=job['mode'], cond=job.cond, tf=TESTABLE[t], n_test_genes=int((tt == t).sum()), spearman=s))
    if i % 20 == 0: print(i, len(jobs), flush=True)
# references (same rows as the models; no fit)
for split in ["gene"] + [f"fold{k}" for k in range(5)]:
    for arm in ("arm1", "arm2"):
        Zp = KD[arm][0]; arm_tfs = FOLDS.TF[FOLDS.arm1_in].tolist() if arm == "arm1" else FOLDS.TF.tolist(); arm_tfs = [t for t in arm_tfs if np.isfinite(Zp[PTF[t]]).any()]; test_tfs = tf_lists(split, arm_tfs)[1]; tt, gg = kd_rows(test_tfs, set_mask("test", arm), Zp); y = Zp[tt, gg]
        for ref, M in (("ref_win100k", Refs["win100k"]), ("ref_v8", Refs["v8"].astype(float))):
            tix = np.array([TF_IDX[TESTABLE[t]] for t in tt]); p = M[tix, gg]
            for t, s in per_tf_sp(tt, y, p).items(): rows.append(dict(kind=kind, link=LINK, split=split, arm=arm, centring="na", model=ref, mode="none", cond="real", tf=TESTABLE[t], n_test_genes=int((tt == t).sum()), spearman=s))
pd.DataFrame(rows).to_csv(f"{res}/per_tf_long.tsv", sep="\t", index=False, float_format="%.5f")
# expression ladder on test chromosomes
y = m7.expr_target(); gt = np.flatnonzero(set_mask("test", "arm2") & np.isfinite(y)); ER = []; E = pd.read_csv(f"{O}/jobs/{m7.JOB_PREFIX}{kind}_expr_final.tsv", sep="\t", dtype=str, keep_default_na=False); pp = {}
for _, job in E.iterrows():
    rung = int(job.rung); p = expr_model_predict(out_dir(job), job.model, rung, job.cond, gt); ER.append(dict(kind=kind, link=LINK, model=job.model, mode="expr_only", rung=rung, cond=job.cond, r2=r2(y[gt], p))); pp[job.model, rung, job.cond] = p.astype(np.float32)
# joint network expression head (real, rung 4), from the knockdown both-mode fit on Arm 2 train-centred by-gene
import torch
ck_dir = f"{O}/fits/{kind}_{LINK}/final/kd/gene/arm2_train/mlp_both_real"; net, m, ck = mlp_load(f"{ck_dir}/model.pt"); v = json.load(open(f"{ck_dir}/val.json")); bb, xx = expr_x(F, "real", 4, gt); T = lambda a: torch.tensor(a, dtype=torch.float32)
with torch.no_grad(): p = net.f_ex(T(bb), T(xx)).numpy() * v["expr_y_sd"] + v["expr_y_mean"]
ER.append(dict(kind=kind, link=LINK, model="mlp", mode="both_joint", rung=4, cond="real", r2=r2(y[gt], p))); pp["mlp_joint", 4, "real"] = p.astype(np.float32)
pd.DataFrame(ER).to_csv(f"{res}/expr_long.tsv", sep="\t", index=False, float_format="%.5f"); np.savez(f"{res}/expr_preds.npz", y=y[gt].astype(np.float32), gi=gt, **{f"{a}__{b}__{c}": v for (a, b, c), v in pp.items()})
print("scored", kind, len(rows), len(ER))
