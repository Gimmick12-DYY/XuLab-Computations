# DESCRIPTIVE ONLY (SPEC_SIGNED.md step 3). Per TF and arm: mean SIGNED knockdown z over the TF's v8 targets (r6_v8 lists, top 1,000) within the
# >= 10% NTC universe (z loaded as b3b4/b6/m6 load(): units averaged per panel TF), raw and double-centred (gene then TF means removed over the
# arm's TF x universe block), minus the mean over 1,000 expression-matched random gene sets (20 NTC-mean bins, seed 20261109); 95% CI from
# 1,000 bootstrap resamples of the list genes (seed 20261109). Negative = targets fall on knockdown = activator-like.
# Callability: uncalled if no primary unit has phase-1 knockdown q < 0.05 (guide_qc phase1_kd_q) or the TF's split-half ceiling
# (b3b4 reliability_ceilings) is in the arm's bottom tertile. Called sign (per arm) = sign of the DC effect when callable and its CI excludes 0.
# Combined call = the called sign if all called arms agree, "conflict" otherwise. Cross-tab vs source A (signed_v1/tf_sign_annotation.tsv).
# Writes signed_v1/per_tf_signed_DESCRIPTIVE.tsv and signed_v1/crosstab_sourceA_DESCRIPTIVE.tsv. usage: signed_readout.py
import os, subprocess
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; M4 = f"{X}/regulatome/m4_validation"; O = f"{X}/regulatome/m6_improve/signed_v1"
assert not os.path.exists(f"{O}/per_tf_signed_DESCRIPTIVE.tsv")
v = subprocess.run("sha256sum -c --quiet manifest_v8.sha256", shell=True, cwd=f"{X}/regulatome/r6_v8", capture_output=True, text=True); assert v.returncode == 0, v.stdout + v.stderr
SEED = 20261109; NR = 1000; NB = 1000; ARMS = {"arm1": "arm1_high_knockdown", "arm2": "arm2_all_cell"}; ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; panel = lambda t: ALIAS.get(t, t)
L8 = {f[:-4]: pd.read_csv(f"{X}/regulatome/r6_v8/lists/{f}", sep="\t").gene.tolist() for f in os.listdir(f"{X}/regulatome/r6_v8/lists")}
C = pd.read_csv(f"{M4}/b3b4/reliability_ceilings.tsv", sep="\t"); SA = pd.read_csv(f"{O}/tf_sign_annotation.tsv", sep="\t").set_index("TF"); Sp = pd.read_csv(f"{X}/regulatome/m6_improve/split_v1.tsv", sep="\t").set_index("tf")
rows = []
for a in ARMS:
    d = np.load(f"{M4}/kd_response/out_v2/{ARMS[a]}/kd_response.npz", allow_pickle=True); q = pd.read_csv(f"{M4}/kd_response/out_v2/{ARMS[a]}/guide_qc.tsv", sep="\t"); u = d["units"].tolist(); ui = {x: k for k, x in enumerate(u)}
    uni = d["ntc_detect"] >= .1; g = d["genes"][uni]; Z = d["z"][:, uni].astype(float); kp = q[q.kept == 1]; prim = kp[kp.kind.isin(["all", "construct"])]; tf_ = sorted(prim.tf.unique())
    P_ = np.nan_to_num(np.vstack([np.nanmean(Z[[ui[x] for x in prim.unit[prim.tf == t]]], axis=0) for t in tf_])); pts = sorted({panel(t) for t in tf_})
    Zp = np.vstack([P_[[i for i, t in enumerate(tf_) if panel(t) == p]].mean(0) for p in pts]); Zdc = Zp - Zp.mean(0); Zdc = Zdc - Zdc.mean(1, keepdims=True)
    gi = {x: k for k, x in enumerate(g)}; mean = d["ntc_mean"][uni].astype(float); bins = pd.qcut(mean, 20, labels=False, duplicates="drop"); BG = [np.flatnonzero(bins == b) for b in range(bins.max() + 1)]
    kdsig = {p: bool((q[q.tf.map(panel) == p].phase1_kd_q < .05).any()) for p in pts}
    ce = C[C.arm == a].groupby("panel_tf").ceiling.mean(); cut = ce.reindex(pts).quantile(1 / 3)
    rng = np.random.default_rng(SEED + (a == "arm2"))
    for i, p in enumerate(pts):
        if p not in L8: continue
        ix = np.array(sorted({gi[x] for x in L8[p] if x in gi}), int)
        if len(ix) < 50: continue
        cnt = np.bincount(bins[ix], minlength=len(BG)); rs = np.hstack([BG[b][rng.integers(0, len(BG[b]), (NR, c))] for b, c in enumerate(cnt) if c]); bsi = rng.integers(0, len(ix), (NB, len(ix)))
        r = dict(DESCRIPTIVE="DESCRIPTIVE", arm=a, tf=p, split=Sp.set.get(p, "na"), n_genes=len(ix), ceiling=ce.get(p, np.nan), kd_significant=kdsig[p])
        for nm, M in (("raw", Zp), ("DC", Zdc)):
            row = M[i]; nullm = row[rs].mean(1); eff = row[ix].mean() - nullm.mean(); bs = row[ix][bsi].mean(1) - nullm.mean()
            r.update({f"effect_{nm}": eff, f"effect_{nm}_lo": np.percentile(bs, 2.5), f"effect_{nm}_hi": np.percentile(bs, 97.5), f"z_vs_null_{nm}": eff / nullm.std()})
        callable_ = kdsig[p] and not (np.isnan(r["ceiling"]) or r["ceiling"] <= cut); r["callable"] = callable_
        r["call"] = ("activator-like" if r["effect_DC"] < 0 else "repressor-like") if callable_ and (r["effect_DC_lo"] > 0 or r["effect_DC_hi"] < 0) else ("uncalled_weak_KD" if not callable_ else "uncalled_CI")
        rows.append(r)
P = pd.DataFrame(rows)
comb = {}
for t, G in P.groupby("tf"):
    c = set(G.call[G.call.isin(["activator-like", "repressor-like"])]); comb[t] = c.pop() if len(c) == 1 else ("conflict" if len(c) > 1 else "uncalled")
P["combined_call"] = P.tf.map(comb); P["sourceA"] = P.tf.map(SA.predicted_sign); P["sourceA_conf"] = P.tf.map(SA.confidence)
P.to_csv(f"{O}/per_tf_signed_DESCRIPTIVE.tsv", sep="\t", index=False, float_format="%.4g")
T = pd.DataFrame(dict(tf=list(comb), combined_call=list(comb.values()))); T["sourceA"] = T.tf.map(SA.predicted_sign)
X_ = pd.crosstab(T.sourceA, T.combined_call, margins=True); X_.insert(0, "DESCRIPTIVE", "DESCRIPTIVE"); X_.to_csv(f"{O}/crosstab_sourceA_DESCRIPTIVE.tsv", sep="\t")
