# Gate 1 v3 (binned units), raw reads only. usage: gate1_v3.py <unit_type> tf <TF> | <unit_type> module <MODULE>
# unit_type: bin10k, bin50k (fixed genome bins, mappable >= 0.5), reprpc (ReprPC + ReprPCWk domains merged within 5 kb),
#            genebody (protein-coding gene body +/- 10 kb).
# Expected per unit: Poisson GLM log mu = log(reads) + b0 + b1 log1p(leave-unit-out pooled panel CPM) + b2 log(width)
#   + b3 GC + b4 GC^2 + b5 k50 mappable fraction (no ATAC).
# Split-half: 20 random cell splits; residual r = log2((O+0.5)/(E+0.5)) per unit per half; count-aware filter = units with
#   expected >= 5 in BOTH halves (the same count rule for every unit type: residual stability depends on counts, not width).
# Statistic: Spearman(r_A, r_B) over eligible units, mean over splits. Null: 100 pseudo-units, cells drawn per member from the
#   pooled cells of non-member TFs in that member's library group at the member's cell count (v1-style), 5 splits each.
# Saves per-split O_A, E_A, O_B, E_B (all units) for the specificity margin in the summary.
import json, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import rankdata
X = "/work/users/d/y/dyy12/XuLab"; V3 = f"{X}/regulatome/gate1/v3_binned"; U = f"{V3}/units"
N_SPLIT, N_NULL, N_NULL_SPLIT, EMIN = 20, 100, 5, 5.0
utype, mode, name = sys.argv[1:4]
t0 = time.time()
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]
cells = {t: json.load(open(f"{X}/regulatome/m1_binding/v2/counts/{t}.json"))["cells"] for t in tfs}
F = pd.read_csv(f"{X}/regulatome/m1_binding/v2/frac_reads_by_class.tsv", sep="\t", index_col=0)
group = (F.all_elements > 0.22).map({True: "high", False: "low"})
SHALLOW = sorted([t for t in tfs if cells[t] < 3400], key=lambda t: cells[t])
LOWDEPTH = ["SETDB1", "NEUROD4", "SOX12", "ZNF79", "ZNF606", "ZNF777", "ZNF581"]
ADJ = ["MTF2", "EZH2", "DR1", "THAP6", "CHAMP1", "WIZ", "ZNF644", "SRCAP"]
MODS = {"PRC2core": ["EZH2", "MTF2", "PCGF6"], "NuRDcore": ["MBD3"],
        "lowdepth": LOWDEPTH, "lowdepth_noNEUROD4": [t for t in LOWDEPTH if t != "NEUROD4"],
        "lowdepth_noZNF777": [t for t in LOWDEPTH if t != "ZNF777"],
        "prc2adj": ADJ, "prc2adj_noEZH2MTF2": [t for t in ADJ if t not in ("EZH2", "MTF2")]}
for t in ADJ: MODS[f"prc2adj_no{t}"] = [x for x in ADJ if x != t]
rank = pd.Series(cells).rank()
for k in range(20):  # random shallow control sets: each low-depth member swapped for a non-member shallow TF within +/-4 ranks
    rg = np.random.default_rng(1000 + k); pick = []
    for m in LOWDEPTH:
        cand = [t for t in tfs if cells[t] < 3400 and t not in LOWDEPTH and t not in pick and abs(rank[t] - rank[m]) <= 4]
        if not cand: cand = [t for t in tfs if cells[t] < 3400 and t not in LOWDEPTH and t not in pick]
        pick.append(rg.choice(cand))
    MODS[f"randshallow{k + 1:02d}"] = pick
def matched_pool(template, seed):
    """each template member -> random non-member TF of the same library group within +/-4 cell-count ranks (no repeats);
    redraw (up to 500 tries) until the pool's total cell count is within +/-20% of the template's"""
    rg = np.random.default_rng(seed); target = sum(cells[t] for t in template); best = None
    for _ in range(500):
        pick = []
        for m in template:
            cand = [t for t in tfs if t not in ADJ and t not in pick and group[t] == group[m] and abs(rank[t] - rank[m]) <= 4]
            if not cand: cand = [t for t in tfs if t not in ADJ and t not in pick and group[t] == group[m]]
            pick.append(rg.choice(cand))
        dev = abs(sum(cells[t] for t in pick) / target - 1)
        if best is None or dev < best[0]: best = (dev, pick)
        if dev <= 0.2: break
    return best[1]
for k in range(20):  # pooling controls for PRC2-adjacent (8 TFs) and PRC2-adjacent minus EZH2/MTF2 (6 TFs)
    MODS[f"randadj8_{k + 1:02d}"] = matched_pool(ADJ, 2000 + k)
    MODS[f"randadj6_{k + 1:02d}"] = matched_pool([t for t in ADJ if t not in ("EZH2", "MTF2")], 3000 + k)
members = [name] if mode == "tf" else MODS[name]
label = name

cv = np.load(f"{U}/{utype}_cov.npz"); gc, mp, wd = cv["gc"], cv["map"], cv["width"]; nU = len(gc)
raw = {t: sp.load_npz(f"{V3}/cellmats/{t}.{utype}.npz").tocsr() for t in tfs}
reads = {t: np.load(f"{V3}/cellmats/{t}.reads.npy").astype(float) for t in tfs}
tf_sum = {t: np.asarray(raw[t].sum(0)).ravel() for t in tfs}; tf_reads = {t: reads[t].sum() for t in tfs}
tot_sum = sum(tf_sum.values()); tot_reads = sum(tf_reads.values())
gpool = {}
for g_ in set(group[m] for m in members):
    ds = [d for d in tfs if d not in members and group[d] == g_]
    gpool[g_] = (sp.vstack([raw[d] for d in ds]).tocsr(), np.concatenate([reads[d] for d in ds]))
print(f"{utype} {label}: members {members}, units {nU:,}, load {time.time() - t0:.0f}s", flush=True)

def glm_mu(y, off, Xd):
    b = np.zeros(Xd.shape[1]); b[0] = np.log(max(y.sum(), 1) / np.exp(off).sum())
    for _ in range(30):
        eta = off + Xd @ b; mu = np.exp(eta); z = eta - off + (y - mu) / mu
        XtW = Xd.T * mu; bn = np.linalg.solve(XtW @ Xd + 1e-6 * np.eye(len(b)), XtW @ z)
        if np.max(np.abs(bn - b)) < 1e-7: b = bn; break
        b = bn
    return np.exp(off + Xd @ b)
def spearman(a, b): return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])

def unit_stats(Rm, rd, excluded, splits, rng, keep=False):
    pool = (tot_sum - sum(tf_sum[t] for t in excluded)) / (tot_reads - sum(tf_reads[t] for t in excluded)) * 1e6
    Xd = np.column_stack([np.ones(nU), np.log1p(pool), np.log(wd), gc, gc ** 2, mp])
    y = np.asarray(Rm.sum(0)).ravel(); N = rd.sum(); mu = glm_mu(y, np.full(nU, np.log(N / 1e6)), Xd)
    rs, ng, prof = [], [], []
    for _ in range(splits):
        h = rng.permutation(Rm.shape[0]) < Rm.shape[0] // 2
        oa, ob = np.asarray(Rm[h].sum(0)).ravel(), np.asarray(Rm[~h].sum(0)).ravel()
        ea, eb = mu * rd[h].sum() / N, mu * rd[~h].sum() / N
        k = (ea >= EMIN) & (eb >= EMIN); ng.append(int(k.sum()))
        if k.sum() >= 50:
            rs.append(spearman(np.log2((oa[k] + .5) / (ea[k] + .5)), np.log2((ob[k] + .5) / (eb[k] + .5))))
        if keep: prof.append(np.stack([oa, ea, ob, eb]).astype(np.float32))
    return (float(np.mean(rs)) if rs else np.nan), float(np.mean(ng)), prof, mu

rng = np.random.default_rng(20261030 + sum(map(ord, label + utype)) % 10000)
Rm = sp.vstack([raw[m] for m in members]).tocsr(); rd = np.concatenate([reads[m] for m in members])
obs, ng, prof, mu = unit_stats(Rm, rd, members, N_SPLIT, rng, keep=True)
print(f"observed {obs:.4f}, eligible units {ng:.0f}, {time.time() - t0:.0f}s", flush=True)
null = []
if np.isfinite(obs):
    for k in range(N_NULL):
        parts = []
        for m in members:
            PR, PD = gpool[group[m]]; c = rng.choice(PR.shape[0], size=cells[m], replace=cells[m] > PR.shape[0])
            parts.append((PR[c], PD[c]))
        s_, _, _, _ = unit_stats(sp.vstack([p[0] for p in parts]).tocsr(), np.concatenate([p[1] for p in parts]), members, N_NULL_SPLIT, rng)
        null.append(s_)
nd = np.array([x for x in null if np.isfinite(x)])
row = dict(unit_type=utype, mode=mode, unit=label, members=",".join(members), n_cells=sum(cells[m] for m in members),
           groups=",".join(sorted(set(group[m] for m in members))), eligible_units=ng, obs=obs,
           null_mean=nd.mean() if len(nd) else np.nan, null_p90=np.percentile(nd, 90) if len(nd) else np.nan,
           p=(1 + (nd >= obs).sum()) / (1 + len(nd)) if len(nd) else np.nan)
outdir = f"{V3}/results" if mode == "tf" else f"{V3}/results_modules"
pd.DataFrame([row]).to_csv(f"{outdir}/{utype}.{label}.tsv", sep="\t", index=False, float_format="%.4f")
np.savez_compressed(f"{outdir}/{utype}.{label}.profiles.npz", OE=np.stack(prof), mu=mu.astype(np.float32))
print(row, f"total {time.time() - t0:.0f}s")
