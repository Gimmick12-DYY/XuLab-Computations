# M1 binding layer: raw obs/exp (NB GLM), imputed residual, blend; writes elements x TF matrices
import sys, json
import numpy as np, pandas as pd
from m1_common import load_elements

elements, cdir, out = sys.argv[1:4]
E, _ = load_elements(elements)
tfs = [l.strip() for l in open(f"{out}/tfs.txt") if l.strip()]
n, T = len(E), len(tfs)
raw = np.zeros((n, T), np.float64); imp = np.zeros((n, T), np.float64); tot = []
for j, tf in enumerate(tfs):
    z = np.load(f"{cdir}/{tf}.npz"); raw[:, j] = z["raw"]; imp[:, j] = z["imp"]
    tot.append(json.load(open(f"{cdir}/{tf}.json")))
tot = pd.DataFrame(tot).set_index("tf").loc[tfs]
a = np.load(f"{cdir}/atac_counts.npz")
atac_cpm = a["atac"] / a["total_cuts"] * 1e6
width = (E.end - E.start).values.astype(float)

# leave-one-out panel background: mean of the other TFs' CPMs (equal weight per TF, so deep TFs do not dominate)
cpm = raw / tot.reads.values[None, :] * 1e6
loo = (cpm.sum(1, keepdims=True) - cpm) / (T - 1)

def nb_glm(y, X, off, iters=25, rounds=4):
    """NB2 GLM by IRLS with a moment estimate of alpha alternated with the coefficient fit."""
    b = np.zeros(X.shape[1]); b[0] = np.log(y.mean() + 1e-9) - off.mean(); alpha = 0.0
    for _ in range(rounds):
        for _ in range(iters):
            eta = X @ b + off; mu = np.exp(np.clip(eta, -30, 30))
            w = mu / (1 + alpha * mu)
            zz = eta - off + (y - mu) / np.maximum(mu, 1e-12)
            XtW = X.T * w
            b_new = np.linalg.solve(XtW @ X + 1e-8 * np.eye(X.shape[1]), XtW @ zz)
            if np.max(np.abs(b_new - b)) < 1e-7: b = b_new; break
            b = b_new
        mu = np.exp(np.clip(X @ b + off, -30, 30))
        alpha = max(1e-6, float(((y - mu) ** 2 - mu).sum() / (mu ** 2).sum()))
    return b, alpha, mu

B_lfc = np.zeros((n, T), np.float32); B_pr = np.zeros((n, T), np.float32); MU = np.zeros((n, T), np.float32); fits = []
for j, tf in enumerate(tfs):
    y = raw[:, j]
    X = np.column_stack([np.ones(n), np.log1p(atac_cpm), np.log1p(loo[:, j]), np.log(width)])
    off = np.full(n, np.log(tot.reads.iloc[j]))
    b, alpha, mu = nb_glm(y, X, off)
    B_lfc[:, j] = np.log2((y + 0.5) / (mu + 0.5)); B_pr[:, j] = (y - mu) / np.sqrt(mu + alpha * mu ** 2); MU[:, j] = mu
    fits.append(dict(tf=tf, alpha=alpha, b0=b[0], b_atac=b[1], b_loo=b[2], b_width=b[3],
                     dev_explained=float(1 - ((y - mu) ** 2 / (mu + alpha * mu ** 2)).sum() / ((y - y.mean()) ** 2 / (y.mean() + alpha * y.mean() ** 2)).sum())))
    print(tf, round(alpha, 3), np.round(b, 3), flush=True)

# imputed: overlap-weighted cell calls -> CPM -> log2 ratio vs leave-one-out mean CPM -> z per TF
imp_cpm = imp / np.maximum(imp.sum(0, keepdims=True), 1e-9) * 1e6
imp_loo = (imp_cpm.sum(1, keepdims=True) - imp_cpm) / (T - 1)
r = np.log2((imp_cpm + 1) / (imp_loo + 1))
B_imp = ((r - r.mean(0)) / np.where(r.std(0) > 0, r.std(0), 1)).astype(np.float32)
zr = ((B_lfc - B_lfc.mean(0)) / B_lfc.std(0)).astype(np.float32)
blend = {}
for k in (3000, 10000, 30000):
    w = (tot.cells.values / (tot.cells.values + k))[None, :]
    blend[k] = (w * zr + (1 - w) * B_imp).astype(np.float32)

np.savez_compressed(f"{out}/binding.npz", tfs=np.array(tfs), element_id=E.element_id.values,
                    raw=raw.astype(np.int32), mu=MU, B_lfc=B_lfc, B_pr=B_pr, z_lfc=zr, imp_cpm=imp_cpm.astype(np.float32),
                    B_imp=B_imp, B_blend_k3000=blend[3000], B_blend_k10000=blend[10000], B_blend_k30000=blend[30000],
                    atac_cpm=atac_cpm.astype(np.float32), raw_cpm=cpm.astype(np.float32))
f = pd.DataFrame(fits).set_index("tf").join(tot)
for k in (3000, 10000, 30000): f[f"w_raw_k{k}"] = f.cells / (f.cells + k)
f.to_csv(f"{out}/tf_fits.tsv", sep="\t", float_format="%.4g")
print(f.to_string())
