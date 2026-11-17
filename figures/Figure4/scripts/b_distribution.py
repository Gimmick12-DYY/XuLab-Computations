"""Fig 4b follow-up: why the 3,081 TF-pair r sit near 0. Rebuilds the five ChromHMM element-set matrices with the exact steps of
regulatome/r2_targets/chromhmm_corrected_sets.py and repressive_chromhmm_corrected.py (data_min20): reads/cell per ChromHMM segment ->
equal per-TF totals (CPM) -> element / across-TF mean (obs/exp) -> Pearson across elements -> SpQN by cell count.
Reports per set: units, sparsity (per TF zero fraction, raw reads per unit; medians over TFs), off-diagonal r quantiles at
(a) Pearson on CPM before obs/exp, (b) after obs/exp, (c) after SpQN, r at the 0.60/0.90/0.95 percentile cut-offs, and a noise ceiling:
each TF's raw reads split in two by binomial thinning (seed 0), both halves through the same pipeline, r(half1, half2) per TF.
usage: b_distribution.py out_tsv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
X = Path("/work/users/d/y/dyy12/XuLab"); TFC = X / "tf_complex"
sys.path.insert(0, str(TFC / "scripts")); sys.path.insert(0, str(X / "downstream"))
from compartment_rpkm_correlation import load_cells, spqn
tfs = [t.strip() for t in (TFC / "results_compartment_counts_spqn_domain_sm2/tf_similarity_genome.tsv").read_text().splitlines()[0].split("\t")[1:] if t.strip()]
cells = load_cells(X / "data/TF1000cells.meta.csv"); depth = np.array([cells[t.lower()] for t in tfs], float); W = TFC / "work/chromhmm_elements_counts"
M = sp.load_npz(W / "element_matrix.npz").tocsr(); E = pd.read_csv(W / "elements.tsv", sep="\t"); state = E.state.values
tot = np.asarray(M.sum(0)).ravel(); raw_all = M @ sp.diags(depth); rawsum = np.asarray(raw_all.sum(1)).ravel()
SUB = {"all": np.ones(len(state), bool), "noquies": state != "Quies", "tss": np.isin(state, ["TssA", "TssFlnk", "TssFlnkU", "TssFlnkD", "TssBiv"]),
       "enhancer_biv": np.isin(state, ["EnhA1", "EnhA2", "EnhG1", "EnhG2", "EnhWk", "EnhBiv"]), "repressive": np.isin(state, ["ReprPC", "ReprPCWk", "Het", "ZNF/Rpts"])}
off = ~np.eye(len(tfs), dtype=bool); iu = np.triu_indices(len(tfs), 1); Q = [0, .05, .25, .5, .75, .95, .99, 1]
def obsexp(A):
    mu = A.mean(1, keepdims=True); mu[mu == 0] = 1.0; return A / mu
rng = np.random.default_rng(0); rows = []
for name, m in SUB.items():
    keep = m & (rawsum >= 20) & (np.asarray(M.sum(1)).ravel() > 0)
    Mk = np.asarray(M[keep].todense(), float); A = Mk / tot * 1e6; raw = np.rint(np.asarray(raw_all[keep].todense(), float))
    Ra = np.nan_to_num(np.corrcoef(A, rowvar=False)); Rb = np.nan_to_num(np.corrcoef(obsexp(A), rowvar=False)); Rc = spqn(Rb, depth)
    h1 = rng.binomial(raw.astype(np.int64), 0.5).astype(float); h2 = raw - h1                     # split-half noise ceiling
    B1 = obsexp(h1 / h1.sum(0) * 1e6); B2 = obsexp(h2 / h2.sum(0) * 1e6)
    ceil = np.array([np.corrcoef(B1[:, t], B2[:, t])[0, 1] for t in range(len(tfs))])
    base = dict(set=name, units=int(keep.sum()), unit="ChromHMM-18 segment, >=20 raw reads summed over TFs", median_bp=int(np.median(E.bp.values[keep])),
                tf_zero_frac_median=round(float(np.median((raw == 0).mean(0))), 3), tf_reads_per_unit_median=round(float(np.median(raw.mean(0))), 3),
                tf_reads_per_unit_min=round(float(raw.mean(0).min()), 3), tf_reads_per_unit_max=round(float(raw.mean(0).max()), 3),
                splithalf_r_median=round(float(np.median(ceil)), 3), splithalf_r_min=round(float(ceil.min()), 3), splithalf_r_max=round(float(ceil.max()), 3))
    for stage, R in (("a_pearson_cpm", Ra), ("b_obs_exp", Rb), ("c_spqn", Rc)):
        v = R[iu]; d = dict(base, stage=stage, **{f"q{int(q*100):02d}": round(float(np.quantile(v, q)), 4) for q in Q}, sd=round(float(v.std()), 4))
        for c in (.60, .90, .95): d[f"r_at_pct{int(c*100)}"] = round(float(np.quantile(Rc[iu] if stage == "c_spqn" else v, c)), 4)
        rows.append(d); print({k: d[k] for k in ("set", "stage", "units", "q50", "q95", "q99", "r_at_pct90", "splithalf_r_median")}, flush=True)
pd.DataFrame(rows).to_csv(sys.argv[1], sep="\t", index=False)
