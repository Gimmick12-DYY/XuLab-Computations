"""Fig 4b input: SpQN-normalised Pearson r matrices for the all / non-quiescent / TSS element sets, recomputed with the exact code of
regulatome/r2_targets/chromhmm_corrected_sets.py (which saved only percentiles). Reports how closely pct(Rn) reproduces the stored
regulatome/chromhmm_corrected_sets/pct_*.tsv (the files candidates_frozen_v1 reads; current inputs give max |diff| ~0.005), then writes spqn_<set>.tsv
to out_dir and the comparison to out_dir/spqn_check.tsv.
Enhancer and repressive SpQN r are already stored (repressive_chromhmm_corrected/data_min20/spqn_*.tsv). usage: b_export_spqn.py out_dir
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
X = Path("/work/users/d/y/dyy12/XuLab"); TFC = X / "tf_complex"; O = Path(sys.argv[1]); O.mkdir(exist_ok=True, parents=True)
sys.path.insert(0, str(TFC / "scripts")); sys.path.insert(0, str(X / "downstream"))
from compartment_rpkm_correlation import load_cells, spqn
tfs = [t.strip() for t in (TFC / "results_compartment_counts_spqn_domain_sm2/tf_similarity_genome.tsv").read_text().splitlines()[0].split("\t")[1:] if t.strip()]
cells = load_cells(X / "data/TF1000cells.meta.csv"); depth = np.array([cells[t.lower()] for t in tfs], float); W = TFC / "work/chromhmm_elements_counts"; assert (W / "tfs.txt").read_text().split() == tfs
M = sp.load_npz(W / "element_matrix.npz").tocsr(); E = pd.read_csv(W / "elements.tsv", sep="\t"); state = E.state.values
tot = np.asarray(M.sum(0)).ravel(); rawsum = np.asarray((M @ sp.diags(depth)).sum(1)).ravel()
def pct(R):
    off = ~np.eye(len(R), dtype=bool); v = np.sort(R[off]); P = np.searchsorted(v, R, side="right") / len(v); np.fill_diagonal(P, 1.0); return P
CHK = []
SUB = {"all": np.ones(len(state), bool), "noquies": state != "Quies", "tss": np.isin(state, ["TssA", "TssFlnk", "TssFlnkU", "TssFlnkD", "TssBiv"])}
for name, m in SUB.items():
    keep = m & (rawsum >= 20); A = np.asarray(M[keep].todense(), float) / tot * 1e6; mu = A.mean(1, keepdims=True); mu[mu == 0] = 1.0; R = np.nan_to_num(np.corrcoef(A / mu, rowvar=False)); Rn = spqn(R, depth)
    ref = pd.read_csv(X / f"regulatome/chromhmm_corrected_sets/pct_{name}.tsv", sep="\t", index_col=0).loc[tfs, tfs].values
    D = np.abs(np.round(pct(Rn), 5) - ref); iu = np.triu_indices(len(tfs), 1); from scipy.stats import spearmanr
    d = D.max(); CHK.append(dict(set=name, max_abs_diff=round(float(d), 5), mean_abs_diff=round(float(D[iu].mean()), 6), n_pairs_diff_gt_001=int((D[iu] > .001).sum()), spearman=round(float(spearmanr(pct(Rn)[iu], ref[iu])[0]), 6)))
    pd.DataFrame(Rn, index=tfs, columns=tfs).to_csv(O / f"spqn_{name}.tsv", sep="\t", float_format="%.6f"); print(name, "elements", int(keep.sum()), "pct max diff vs stored", d, flush=True)
pd.DataFrame(CHK).to_csv(O / "spqn_check.tsv", sep="\t", index=False); print(pd.DataFrame(CHK).to_string(index=False))
