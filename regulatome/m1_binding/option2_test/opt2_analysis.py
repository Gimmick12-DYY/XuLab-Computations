# Option-2 test: (a) TF-level motif enrichment coefficient (TF reads vs panel reads, ATAC-stratified NB GLM) and
# ATAC x motif scoring vs bulk ChIP; (b) pooled raw signal at bulk peaks within ATAC deciles; (c) B_imp within ATAC deciles
import sys, glob, os, subprocess
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

O, M1, elements = sys.argv[1:4]
rng = np.random.default_rng(1)
z = np.load(f"{M1}/binding.npz", allow_pickle=True)
tfs = list(z["tfs"]); eid = z["element_id"]; n = len(eid); pos = {e: i for i, e in enumerate(eid)}
raw = z["raw"].astype(float); atac = z["atac_cpm"].astype(float); Bimp = z["B_imp"]; rawcpm = z["raw_cpm"]
E = pd.read_csv(elements, sep="\t", header=None); width = (E[2] - E[1]).values.astype(float)
reads = pd.read_csv(f"{M1}/tf_fits.tsv", sep="\t", index_col=0).loc[tfs, "reads"].values.astype(float)
# strata: ATAC density decile x width tertile
dens = np.log1p(atac / width * 1000)
dec = pd.qcut(dens + rng.uniform(0, 1e-9, n), 10, labels=False)
wt = pd.qcut(width + rng.uniform(0, 1e-6, n), 3, labels=False)
strata = dec * 3 + wt
S = np.zeros((n, 30)); S[np.arange(n), strata] = 1
SIDX = [np.where(strata == k)[0] for k in range(30)]

def nb_fit(y, X, off, iters=50):
    """NB2 IRLS with moment alpha; returns beta, SE (model-based), alpha."""
    b = np.zeros(X.shape[1]); b[:] = 0; alpha = 0.0
    b0 = np.linalg.lstsq(X, np.log((y + 0.5)) - off, rcond=None)[0]; b = b0
    for rnd in range(4):
        for _ in range(iters):
            eta = X @ b + off; mu = np.exp(np.clip(eta, -30, 30)); w = mu / (1 + alpha * mu)
            zz = eta - off + (y - mu) / np.maximum(mu, 1e-12)
            XtW = X.T * w; H = XtW @ X + 1e-8 * np.eye(X.shape[1])
            bn = np.linalg.solve(H, XtW @ zz)
            if np.max(np.abs(bn - b)) < 1e-8: b = bn; break
            b = bn
        mu = np.exp(np.clip(X @ b + off, -30, 30))
        alpha = max(1e-6, float(((y - mu) ** 2 - mu).sum() / (mu ** 2).sum()))
    w = mu / (1 + alpha * mu); H = (X.T * w) @ X
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return b, se, alpha

# ---- (a) motif coefficient -----------------------------------------------------------------
src = pd.read_csv(f"{O}/motif_sources.tsv", sep="\t", index_col=0)
rows, motif = [], {}
for f in sorted(glob.glob(f"{O}/fimo/*.best_p.tsv")):
    tf = os.path.basename(f).split(".")[0]
    d = pd.read_csv(f, sep="\t", header=None, names=["eid", "p"])
    m = np.zeros(n, bool); m[[pos[e] for e in d.eid[d.p < 1e-4]]] = True; motif[tf] = m
    j = tfs.index(tf)
    X = np.column_stack([S, np.log(width), m.astype(float)])
    bt, st, at = nb_fit(raw[:, j], X, np.full(n, np.log(reads[j])))
    panel = raw.sum(1) - raw[:, j]
    bp, sp_, ap = nb_fit(panel, X, np.full(n, np.log(reads.sum() - reads[j])))
    d_ = bt[-1] - bp[-1]; se_d = np.sqrt(st[-1] ** 2 + sp_[-1] ** 2)
    # observed vs panel-expected reads at motif elements (direct, model-free check)
    rows.append(dict(tf=tf, motif=src.loc[tf, "motif_source"], frac_elements_with_motif=m.mean(), tf_reads_in_elements=int(raw[:, j].sum()),
                     beta_tf=bt[-1], ci_lo=bt[-1] - 1.96 * st[-1], ci_hi=bt[-1] + 1.96 * st[-1], p_tf=2 * stats.norm.sf(abs(bt[-1] / st[-1])),
                     beta_panel=bp[-1], delta_tf_minus_panel=d_, delta_ci_lo=d_ - 1.96 * se_d, delta_ci_hi=d_ + 1.96 * se_d,
                     p_delta=2 * stats.norm.sf(abs(d_ / se_d)), fold_specific=np.exp(d_)))
A = pd.DataFrame(rows).set_index("tf")
A["q_delta"] = stats.false_discovery_control(A.p_delta.values)
A = A.sort_values("delta_tf_minus_panel", ascending=False)
A.to_csv(f"{O}/motif_coefficients_45tf.tsv", sep="\t", float_format="%.4g")
print(A[["frac_elements_with_motif", "beta_tf", "ci_lo", "ci_hi", "beta_panel", "delta_tf_minus_panel", "p_delta", "q_delta", "fold_specific"]].round(3).to_string())

# ---- bulk truth ---------------------------------------------------------------------------------
D = "/work/users/d/y/dyy12/XuLab/data"
BULK = {"CTCF": "CTCF_majority2of3.bed", "MAZ": "MAZ_HEK293_peaks.bed", "NFYA": "NFYA_K562_peaks.bed", "ZBTB7A": "ZBTB7A_HEK293_peaks.bed",
        "ZIC2": "ZIC2_HEK293_peaks.bed", "ZNF282": "ZNF282_HEK293_peaks.bed", "ZNF777": "ZNF777_HEK293_peaks.bed"}
def hits(bed):
    o = subprocess.run(["bedtools", "intersect", "-u", "-a", elements, "-b", bed], capture_output=True, text=True, check=True).stdout
    y = np.zeros(n, bool)
    for l in o.splitlines(): y[pos[l.split("\t")[3]]] = True
    return y

def strat_auc(y, s):
    """Mean within-ATAC-decile AUROC (deciles with both classes), weighted by positives."""
    a, w = [], []
    for k in range(10):
        i = dec == k
        if y[i].any() and (~y[i]).any(): a.append(roc_auc_score(y[i], s[i])); w.append(y[i].sum())
    return float(np.average(a, weights=w))

rows_a, rows_b = [], []
panel_all = raw.sum(1)
for tf, bed in BULK.items():
    y = hits(f"{D}/{bed}"); j = tfs.index(tf)
    r = dict(tf=tf, ATAC=roc_auc_score(y, atac))
    if tf in motif:
        m = motif[tf]; beta = A.loc[tf, "delta_tf_minus_panel"]
        r["ATAC_x_motif"] = roc_auc_score(y, atac * m + 1e-9 * atac)  # motif elements ranked by ATAC first
        r["ATAC_x_motif_x_coef"] = roc_auc_score(y, np.log(atac + 1e-3) + max(beta, 0) * m)
        r["motif_only"] = roc_auc_score(y, m.astype(float))
        r["coef_delta"] = beta
    r["B_imp"] = roc_auc_score(y, Bimp[:, j]); r["raw_cpm"] = roc_auc_score(y, rawcpm[:, j])
    r["within_decile_ATAC"] = strat_auc(y, atac); r["within_decile_B_imp"] = strat_auc(y, Bimp[:, j])
    r["within_decile_raw_cpm"] = strat_auc(y, rawcpm[:, j]); r["within_decile_B_lfc"] = strat_auc(y, z["B_lfc"][:, j])
    if tf in motif: r["within_decile_motif"] = strat_auc(y, motif[tf].astype(float))
    rows_a.append(r)
    # ---- (b) pooled raw reads at bulk peaks within ATAC decile x width strata, TF vs panel -----
    t = raw[:, j]; pnl = panel_all - t
    def fold(lab):
        obs_t = exp_t = obs_p = exp_p = 0.0
        for k in range(30):
            i = SIDX[k]; yi = lab[i]
            if not yi.any() or yi.all(): continue
            bp_pos, bp_neg = width[i][yi].sum(), width[i][~yi].sum()
            obs_t += t[i][yi].sum(); exp_t += t[i][~yi].sum() / bp_neg * bp_pos
            obs_p += pnl[i][yi].sum(); exp_p += pnl[i][~yi].sum() / bp_neg * bp_pos
        return obs_t / exp_t, obs_p / exp_p
    ft, fp = fold(y); stat = np.log(ft / fp)
    null = []
    for _ in range(1000):
        yp = y.copy()
        for k in range(30):
            i = SIDX[k]; yp[i] = rng.permutation(y[i])
        a_, b_ = fold(yp); null.append(np.log(a_ / b_))
    null = np.array(null)
    rows_b.append(dict(tf=tf, n_pos=int(y.sum()), fold_tf=ft, fold_panel=fp, specific_ratio=ft / fp,
                       perm_p=(1 + (null >= stat).sum()) / 1001, null_ratio_95=float(np.exp(np.percentile(null, 95)))))
R = pd.DataFrame(rows_a).set_index("tf"); R.loc["median"] = R.median(numeric_only=True)
B = pd.DataFrame(rows_b).set_index("tf")
R.to_csv(f"{O}/bulk_auroc_option2.tsv", sep="\t", float_format="%.3f"); B.to_csv(f"{O}/pooled_bulk_signal.tsv", sep="\t", float_format="%.4g")
print(R.round(3).to_string()); print(B.round(3).to_string())
