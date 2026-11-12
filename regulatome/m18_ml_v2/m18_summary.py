# m18_ml_v2 summary: ladder.tsv, contrasts.tsv (paired TF-bootstrap, 2,000 draws, seed 20261112), per_tf_oof.tsv, references.tsv, bytf.tsv. Cross-validated, benchmark-developed.
import glob, os
import numpy as np, pandas as pd
from scipy.stats import rankdata
RG = "/work/users/d/y/dyy12/XuLab/regulatome"; O = f"{RG}/m18_ml_v2"; RUN = f"{O}/out/runs"; OUT = f"{O}/out"; LAB = "cross-validated, benchmark-developed"
rng = np.random.default_rng(20261112); NB = 2000
T = {os.path.basename(f)[:-11]: pd.read_csv(f, sep="\t").set_index("tf").spearman_oof for f in glob.glob(f"{RUN}/*_per_tf.tsv")}
RUNGS = ["r1_A", "r2_AR", "r3_ARH", "r4_ARHC", "r5_B", "r6_D", "r7_F", "r8_G_full"]
def boot(d):
    d = d.dropna().values; b = d[rng.integers(0, len(d), (NB, len(d)))].mean(1); return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(d)
L = []; C = []
for tset in ("primary", "all"):
    for model in ("lgb", "enet"):
        for k, r in enumerate(RUNGS):
            n = f"{tset}_{model}_{r}"
            if n not in T: continue
            m, lo, hi, nt = boot(T[n]); row = dict(label=LAB, tfs=tset, model=model, rung=r, n_tfs=nt, mean_spearman=m, lo=lo, hi=hi)
            if k: 
                p = f"{tset}_{model}_{RUNGS[k - 1]}"
                if p in T: dm, dlo, dhi, _ = boot(T[n] - T[p]); row.update(minus_previous=dm, mp_lo=dlo, mp_hi=dhi)
            L.append(row)
        def con(name, a, b):
            if a in T and b in T: m, lo, hi, nt = boot(T[a] - T[b]); C.append(dict(label=LAB, tfs=tset, model=model, contrast=name, n_tfs=nt, mean_diff=m, lo=lo, hi=hi, interval_above_0=lo > 0))
        f, base = f"{tset}_{model}_r8_G_full", f"{tset}_{model}_r4_ARHC"
        con("richer model adds: full - A", f, f"{tset}_{model}_r1_A"); con("HITAG contributes: full - (A+R+H+C)", f, base)
        con("+B+BC (this TF binding)", f"{tset}_{model}_r5_B", base); con("+D (79-TF context)", f"{tset}_{model}_r6_D", f"{tset}_{model}_r5_B")
        con("coordinated binding adds: +F", f"{tset}_{model}_r7_F", f"{tset}_{model}_r6_D"); con("complex pairs add (provisional): +G", f, f"{tset}_{model}_r7_F")
        if tset == "primary":
            con("full real - full swapped-TF", f, f"primary_{model}_full_swap")
            ps = [T[x] for x in (f"primary_{model}_full_pseudo{d}" for d in range(3)) if x in T]
            if ps and f in T:
                pm = pd.concat(ps, axis=1).mean(1); m, lo, hi, nt = boot(T[f] - pm); C.append(dict(label=LAB, tfs=tset, model=model, contrast="full real - full pseudo-TF (mean of 3)", n_tfs=nt, mean_diff=m, lo=lo, hi=hi, interval_above_0=lo > 0))
                if base in T:
                    for nm, ctl in (("swap", T.get(f"primary_{model}_full_swap")), ("pseudo", pm)):
                        if ctl is None: continue
                        m, lo, hi, nt = boot((T[f] - T[base]) - (ctl - T[base])); C.append(dict(label=LAB, tfs=tset, model=model, contrast=f"HITAG gain real minus HITAG gain {nm}", n_tfs=nt, mean_diff=m, lo=lo, hi=hi, interval_above_0=lo > 0))
            for extra in ("full_permtarget", "r2_AR_half1", "full_half1", "full_byTF_keepE", "full_byTF_dropE"):
                n = f"primary_{model}_{extra}"
                if n in T: m, lo, hi, nt = boot(T[n]); L.append(dict(label=LAB, tfs="primary", model=model, rung=extra, n_tfs=nt, mean_spearman=m, lo=lo, hi=hi))
Ld = pd.DataFrame(L); Ld.to_csv(f"{OUT}/ladder.tsv", sep="\t", index=False, float_format="%.4f"); Cd = pd.DataFrame(C); Cd.to_csv(f"{OUT}/contrasts.tsv", sep="\t", index=False, float_format="%.4f")
# per-TF out-of-fold Spearman beside the reliability ceiling
FT = pd.read_csv(f"{RG}/m7_step2/splits/tf_folds.tsv", sep="\t").set_index("TF").ceiling
P = pd.DataFrame({k: T[k] for k in T if k.startswith("primary_lgb_") or k.startswith("primary_enet_")}); P.insert(0, "reliability", FT.reindex(P.index)); P.insert(1, "max_r_sqrt_reliability", np.sqrt(P.reliability.clip(lower=0)))
P.to_csv(f"{OUT}/per_tf_oof_primary.tsv", sep="\t", float_format="%.4f")
A = pd.DataFrame({k: T[k] for k in T if k.startswith("all_")}); A.insert(0, "reliability", FT.reindex(A.index)); A.to_csv(f"{OUT}/per_tf_oof_all.tsv", sep="\t", float_format="%.4f")
# unfitted references (no fitting): per-TF Spearman with z over all genes
IX = np.load(f"{O}/features/index.npz", allow_pickle=True); TT = list(IX["tfs"].astype(str)); ug = IX["genes"].astype(str); NG = len(ug)
Rz = np.load(f"{RG}/m9_abc_diagnosis/work/responses_arm2.npz", allow_pickle=True); Z = Rz["r1"].astype(float); OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}; gi = {g: k for k, g in enumerate(ug)}
for i, t in enumerate(TT):
    for o in OWN.get(t, [t]):
        if o in gi: Z[i, gi[o]] = np.nan
pr = np.load(f"{O}/features/pair_real.npz", allow_pickle=True); pn = list(pr["names"]); X = pr["X"].reshape(len(TT), NG, len(pn)); c = {n: k for k, n in enumerate(pn)}
def sp(a, b): m = np.isfinite(a) & np.isfinite(b); return np.nan if m.sum() < 10 or np.std(a[m]) == 0 else float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1])
R = []
for i, t in enumerate(TT):
    R.append(dict(tf=t, ref_peak_x_abc=sp(X[i, :, c["B_pct_peak_score"]], Z[i]), ref_promoter_reads=sp(X[i, :, c["B_prom_reads"]], Z[i]),
                  ref_motif=sp(X[i, :, c["C_prom_sites"]] + X[i, :, c["C_distal_sites_abc"]], Z[i]) if X[i, 0, c["C_has_motif"]] else np.nan))
R = pd.DataFrame(R).set_index("tf"); R.to_csv(f"{OUT}/references.tsv", sep="\t", float_format="%.4f")
print(Ld.drop(columns="label").round(4).to_string(index=False)); print(Cd.drop(columns="label").round(4).to_string(index=False))
