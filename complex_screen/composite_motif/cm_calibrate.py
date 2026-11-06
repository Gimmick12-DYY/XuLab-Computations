# Calibration: CTCF Gate 1 residual at elements with vs without the motif (FIMO p < 1e-4), real MA0139.2 and 3 shuffled copies. The real motif must shift, shuffles must not.
import numpy as np, pandas as pd
from cm_common import *
E = load_elements(); tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]
Y = {t: np.load(f"{U}/counts/{t}.npz") for t in tfs}; tot = sum(Y[t]["y"] for t in tfs); N = sum(int(Y[t]["total"]) for t in tfs)
y = Y["CTCF"]["y"].astype(float); R = int(Y["CTCF"]["total"]); mu = residual(y, R, tot - y, N - R); lr = np.log2((y + 0.5) / (mu + 0.5)); C = pd.read_csv(f"{U}/covariates.tsv", sep="\t")
F = pd.read_csv(f"{O}/calibration/fimo.tsv", sep="\t", comment="#"); rows = []
np.savez_compressed(f"{O}/calibration/ctcf_residual.npz", y=y, mu=mu)
for m in ["MA0139.2", "shuffle1", "shuffle2", "shuffle3"]:
    h = E.eid.isin(F.loc[F.motif_id == m, "sequence_name"]).values; a, b = h, ~h
    d = dict(motif=m, elements_with=int(a.sum()), elements_without=int(b.sum()), frac_with=round(a.mean(), 4), reads_with=int(y[a].sum()), expected_with=round(mu[a].sum(), 1),
             obs_over_exp_with=round(y[a].sum() / mu[a].sum(), 3), obs_over_exp_without=round(y[b].sum() / mu[b].sum(), 3))
    for k, v in (("resid_reads", y - mu), ("log2_resid", lr), ("cpm", y / R * 1e6), ("atac", np.log1p(C.atac.values)), ("gc", C.gc.values)):
        x, lo, hi = boot_diff(v[a], v[b]); d.update({f"{k}_diff": round(x, 4), f"{k}_lo": round(lo, 4), f"{k}_hi": round(hi, 4)})
    rows.append(d)
T = pd.DataFrame(rows); T.to_csv(f"{O}/calibration/ctcf_calibration.tsv", sep="\t", index=False); pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
print(f"CTCF reads {R:,}, in union elements {int(y.sum()):,}, sum mu {mu.sum():,.0f}"); print(T.T.to_string())
