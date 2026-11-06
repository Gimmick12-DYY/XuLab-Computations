# Accessibility-corrected percentile matrices for the remaining ChromHMM element sets: all, no quiescent, TSS (TssA, TssFlnk, TssFlnkU, TssFlnkD, TssBiv). Same method as repressive_chromhmm_corrected.py: equal per-TF totals, element / its across-TF mean, Pearson, SpQN, percentile; elements with >= 20 raw reads.
import sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
X = Path("/work/users/d/y/dyy12/XuLab"); TFC = X / "tf_complex"; O = X / "regulatome/chromhmm_corrected_sets_pearson"; O.mkdir(exist_ok=True)
sys.path.insert(0, str(TFC / "scripts")); sys.path.insert(0, str(X / "downstream"))
from compartment_rpkm_correlation import load_cells, spqn, _confound
tfs = [t.strip() for t in (TFC / "results_compartment_counts_spqn_domain_sm2/tf_similarity_genome.tsv").read_text().splitlines()[0].split("\t")[1:] if t.strip()]
cells = load_cells(X / "data/TF1000cells.meta.csv"); depth = np.array([cells[t.lower()] for t in tfs], float); W = TFC / "work/chromhmm_elements_counts"; assert (W / "tfs.txt").read_text().split() == tfs
M = sp.load_npz(W / "element_matrix.npz").tocsr(); E = pd.read_csv(W / "elements.tsv", sep="\t"); state = E.state.values
tot = np.asarray(M.sum(0)).ravel(); rawsum = np.asarray((M @ sp.diags(depth)).sum(1)).ravel()
def pct(R):
    off = ~np.eye(len(R), dtype=bool); v = np.sort(R[off]); P = np.searchsorted(v, R, side="right") / len(v); np.fill_diagonal(P, 1.0); return P
SUB = {"all": np.ones(len(state), bool), "noquies": state != "Quies", "tss": np.isin(state, ["TssA", "TssFlnk", "TssFlnkU", "TssFlnkD", "TssBiv"])}; off = ~np.eye(len(tfs), dtype=bool); rows = []
for name, m in SUB.items():
    keep = m & (rawsum >= 20); A = np.asarray(M[keep].todense(), float) / tot * 1e6; mu = A.mean(1, keepdims=True); mu[mu == 0] = 1.0; R = np.nan_to_num(np.corrcoef(A / mu, rowvar=False)); pd.DataFrame(R, index=tfs, columns=tfs).to_csv(O / f"pearson_{name}.tsv", sep="\t", float_format="%.6f"); Rn = spqn(R, depth)
    pd.DataFrame(pct(Rn), index=tfs, columns=tfs).to_csv(O / f"pct_{name}.tsv", sep="\t", float_format="%.5f"); rows.append(dict(set=name, elements_used=int(keep.sum()), median_r=round(float(np.median(R[off])), 4), median_spqn=round(float(np.median(Rn[off])), 4), confound_spqn=round(_confound(Rn, depth), 3))); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(O / "sets.tsv", sep="\t", index=False)
