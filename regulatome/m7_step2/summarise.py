# m7_step2: final tables from results/<kind>/*. usage: summarise.py (run after score.py for raw, and imp if present). Writes per_tf_test.tsv, pooled.tsv, ladder.tsv, decision.tsv.
import os
import numpy as np, pandas as pd
O = "/work/users/d/y/dyy12/XuLab/regulatome/m7_step2"; NB = 2000; SEED = 20261111; F = pd.read_csv(f"{O}/splits/tf_folds.tsv", sep="\t").set_index("TF")
def boot(v, rng):
    v = np.asarray(v, float); n = len(v); b = v[rng.integers(0, n, (NB, n))].mean(1); return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
KINDS = sorted(k for k in os.listdir(f"{O}/results") if os.path.exists(f"{O}/results/{k}/per_tf_long.tsv"))  # <binding kind>_<link set>: raw_hicfixed (primary verdict), raw_powerlaw (sensitivity), imp_*
L = pd.concat([pd.read_csv(f"{O}/results/{k}/per_tf_long.tsv", sep="\t", keep_default_na=False).assign(kind=k) for k in KINDS])
L = pd.concat([L[L.split == "gene"], L[L.split != "gene"].assign(split="byTF")])
L["cc"] = L.cond.str.replace(r"\d+$", "", regex=True)
W = L.groupby(["kind", "split", "arm", "centring", "model", "mode", "tf", "cc"]).spearman.mean().unstack("cc").reset_index()  # pseudo, mis averaged over draws per TF
n_t = L.groupby(["kind", "split", "arm", "tf"]).n_test_genes.max()
ref = L[L.model.str.startswith("ref_")].pivot_table(index=["kind", "split", "arm", "tf"], columns="model", values="spearman")
M = W[~W.model.str.startswith("ref_")].copy(); M = M.join(ref, on=["kind", "split", "arm", "tf"]); M["primary"] = M.real - M.base; M["ceiling"] = M.tf.map(F.ceiling); M["cells"] = M.tf.map(F.cells); M["claim_tier"] = M.tf.map(F.claim_tier); M["fold"] = M.tf.map(F.fold); M["split_v1"] = M.tf.map(F.split_v1_status); M["n_test_genes"] = [n_t.get((a, b, c, d), np.nan) for a, b, c, d in zip(M.kind, M.split, M.arm, M.tf)]
M.to_csv(f"{O}/per_tf_test.tsv", sep="\t", index=False, float_format="%.5f")
rng = np.random.default_rng(SEED); rows = []; SETS = {"all": lambda d: d, "claim_tier": lambda d: d[d.claim_tier], "without_NRF1": lambda d: d[d.tf != "NRF1"]}
enet_base = M[(M.model == "enet") & (M["mode"] == "kd_only") & (M.centring == "train")].set_index(["kind", "split", "arm", "tf"]).base
linear_primary = M[(M.model == "enet") & (M["mode"] == "kd_only")].set_index(["kind", "split", "arm", "centring", "tf"]).primary
for k, G in M.groupby(["kind", "split", "arm", "centring", "model", "mode"]):
    for sn, fn in SETS.items():
        D = fn(G).dropna(subset=["real"]);
        if not len(D): continue
        r = dict(zip(["kind", "split", "arm", "centring", "model", "mode"], k)); r.update(tf_set=sn, n_tfs=len(D))
        r["mean_real"], r["mean_base"] = D.real.mean(), D.base.mean(); pv = D.primary.values; r["primary"], r["primary_lo"], r["primary_hi"] = boot(pv, rng)
        for c in ("pseudo", "mis"):
            if c in D and D[c].notna().all(): dv = (D.real - D[c]).values; r[f"primary_{c}"] = float((D[c] - D.base).mean()); r[f"real_minus_{c}"], r[f"real_minus_{c}_lo"], r[f"real_minus_{c}_hi"] = boot(dv, rng)
        for c in ("ref_win100k", "ref_v8"):
            dv = (D.real - D[c]).values; r[f"minus_{c}"], r[f"minus_{c}_lo"], r[f"minus_{c}_hi"] = boot(dv, rng); r[f"mean_{c}"] = float(D[c].mean())
        if not (k[4] == "enet" and k[5] == "kd_only"):
            lp = D.set_index(["kind", "split", "arm", "centring", "tf"]).index; lv = linear_primary.reindex(lp).values
            if np.isfinite(lv).all(): r["minus_linear"], r["minus_linear_lo"], r["minus_linear_hi"] = boot(pv - lv, rng)
        rows.append(r)
# reference rows
Rr = ref.reset_index().melt(id_vars=["kind", "split", "arm", "tf"], var_name="model", value_name="spearman").dropna(subset=["spearman"]); Rr["base"] = [enet_base.get((a, b, c, d), np.nan) for a, b, c, d in zip(Rr.kind, Rr.split, Rr.arm, Rr.tf)]; Rr["claim_tier"] = Rr.tf.map(F.claim_tier)
for k, G in Rr.groupby(["kind", "split", "arm", "model"]):
    for sn, fn in SETS.items():
        D = fn(G); v = (D.spearman - D.base).dropna().values
        if not len(v): continue
        a, lo, hi = boot(v, rng); rows.append(dict(kind=k[0], split=k[1], arm=k[2], centring="na", model=k[3], mode="none", tf_set=sn, n_tfs=len(v), mean_real=D.spearman.mean(), mean_base=D.base.mean(), primary=a, primary_lo=lo, primary_hi=hi))
P = pd.DataFrame(rows); P.to_csv(f"{O}/pooled.tsv", sep="\t", index=False, float_format="%.5f")
# expression ladder
LADDER = []
for kind in KINDS:
    p = f"{O}/results/{kind}/expr_long.tsv"
    if not os.path.exists(p): continue
    E = pd.read_csv(p, sep="\t", keep_default_na=False); Z = np.load(f"{O}/results/{kind}/expr_preds.npz"); y = Z["y"]; n = len(y); rngE = np.random.default_rng(SEED + 5); idx = rngE.integers(0, n, (NB, n))
    def r2_of(p_, y_=y): return 1 - ((y_ - p_) ** 2).sum() / ((y_ - y_.mean()) ** 2).sum()
    def boot_r2(p):
        out = np.empty(NB)
        for b in range(NB): i = idx[b]; yi = y[i]; out[b] = 1 - ((yi - p[i]) ** 2).sum() / ((yi - yi.mean()) ** 2).sum()
        return out
    for (m, mo), G in E.groupby(["model", "mode"]):
        pk = "mlp_joint" if mo == "both_joint" else m; prev_boot = None; prev_r = np.nan
        for r_, Gr in G.groupby("rung"):
            real = Gr[Gr.cond == "real"]
            if not len(real): continue
            pr = Z[f"{pk}__{r_}__real"]; R = float(real.r2.iloc[0]); bo = boot_r2(pr); row = dict(kind=kind, model=m, mode=mo, rung=int(r_), rung_name={1: "baseline", 2: "+background", 3: "+per-TF binding", 4: "+pairs"}[int(r_)], r2=R, r2_lo=float(np.percentile(bo, 2.5)), r2_hi=float(np.percentile(bo, 97.5)))
            if prev_boot is not None: row["gain_vs_prev"] = R - prev_r; d = bo - prev_boot; row["gain_lo"], row["gain_hi"] = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            ps = Gr[Gr.cond.str.startswith("pseudo")]
            if len(ps):
                row["pseudo_r2_mean"] = float(ps.r2.mean()); pb = np.mean([boot_r2(Z[f"{pk}__{r_}__{c}"]) for c in ps.cond], axis=0); row["real_minus_pseudo"] = R - float(ps.r2.mean()); d = bo - pb; row["real_minus_pseudo_lo"], row["real_minus_pseudo_hi"] = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            LADDER.append(row); prev_boot, prev_r = bo, R
Lad = pd.DataFrame(LADDER); Lad.to_csv(f"{O}/ladder.tsv", sep="\t", index=False, float_format="%.5f")
# decision rule (section 7, Amendments 1-5)
D = []
def get(kind, split, arm, centring, m, mo, tset="all"):
    q = P[(P.kind == kind) & (P.split == split) & (P.arm == arm) & (P.centring == centring) & (P.model == m) & (P["mode"] == mo) & (P.tf_set == tset)]; return q.iloc[0] if len(q) else None
for kind in P.kind.unique():
    for split in ("gene", "byTF"):
        for m, mo in [(a, b) for a in ("enet", "lgb", "mlp") for b in ("kd_only", "both")]:
            r = get(kind, split, "arm2", "train", m, mo)
            if r is None: continue
            a1 = get(kind, split, "arm1", "train", m, mo); raw = get(kind, split, "arm2", "raw", m, mo)
            d = dict(kind=kind, split=split, model=m, mode=mo, primary=r.primary, primary_lo=r.primary_lo, primary_hi=r.primary_hi, primary_pseudo=r.get("primary_pseudo"), primary_mis=r.get("primary_mis"), arm1_primary=a1.primary if a1 is not None else np.nan, raw_centring_primary=raw.primary if raw is not None else np.nan, raw_centring_lo=raw.primary_lo if raw is not None else np.nan, raw_centring_hi=raw.primary_hi if raw is not None else np.nan)
            d["interval_above_0"] = bool(r.primary_lo > 0); d["exceeds_pseudo"] = bool(r.primary > r.get("primary_pseudo", np.inf)); d["exceeds_mis"] = bool(r.primary > r.get("primary_mis", np.inf)); d["arm1_same_sign"] = bool(a1 is not None and np.sign(a1.primary) == np.sign(r.primary))
            d["shows_binding_signal"] = d["interval_above_0"] and d["exceeds_pseudo"] and d["exceeds_mis"] and d["arm1_same_sign"]; d["beats_win100k"] = bool(r.minus_ref_win100k_lo > 0); d["beats_v8"] = bool(r.minus_ref_v8_lo > 0); d["beats_linear"] = bool("minus_linear_lo" in r and r.get("minus_linear_lo", np.nan) > 0)
            d["centrings_agree"] = bool(raw is not None and np.sign(raw.primary) == np.sign(r.primary) and (raw.primary_lo > 0) == (r.primary_lo > 0))  # Amendment 5
            d["replaces_linear"] = bool(split == "gene" and not (m == "enet" and mo == "kd_only") and d["shows_binding_signal"] and d["beats_linear"] and d["beats_win100k"] and d["beats_v8"]); D.append(d)
Dd = pd.DataFrame(D)
# adopted Regulome model per binding kind and link set (Amendment 5): the linear model unless exactly the rule lets a complex one replace it
Dd["adopted"] = False
for kind, G in Dd[Dd.split == "gene"].groupby("kind"):
    c = G[G.replaces_linear]
    pick = c.sort_values("primary", ascending=False).index[0] if len(c) else G[(G.model == "enet") & (G["mode"] == "kd_only")].index[0]
    Dd.loc[pick, "adopted"] = True
Dd.to_csv(f"{O}/decision.tsv", sep="\t", index=False, float_format="%.5f"); print(Dd.to_string())
# expression verdict (Amendment 5): binding adds to an accessibility model of steady-state expression if rung 3 gain over rung 2 and rung 3 real minus pseudo both have intervals above 0
if len(Lad):
    X = Lad[(Lad["mode"] == "expr_only") & (Lad.rung == 3)].copy(); X["gain_interval_above_0"] = X.gain_lo > 0; X["pseudo_interval_above_0"] = X.real_minus_pseudo_lo > 0; X["binding_adds_to_expression"] = X.gain_interval_above_0 & X.pseudo_interval_above_0
    X[["kind", "model", "r2", "gain_vs_prev", "gain_lo", "gain_hi", "real_minus_pseudo", "real_minus_pseudo_lo", "real_minus_pseudo_hi", "gain_interval_above_0", "pseudo_interval_above_0", "binding_adds_to_expression"]].to_csv(f"{O}/decision_expr.tsv", sep="\t", index=False, float_format="%.5f")
