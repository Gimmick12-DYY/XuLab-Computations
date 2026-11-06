# TF x TF correlation on REPRESSIVE ChromHMM-18 elements, built exactly like the ChromHMM enhancer-element matrices
# (tf_complex/scripts/chromhmm_element_correlation.py + spearman_pearson_matrices.py): signal = pseudobulk reads per ChromHMM
# element / TF cell count (tf_complex/work/chromhmm_elements_counts/element_matrix.npz), Pearson across the subset's elements with
# any read, SpQN by cell count (tf_complex spqn), percentile = rank of each off-diagonal value among all off-diagonal values.
# Subsets: enhancer (EnhA1, EnhA2, EnhG1, EnhG2, EnhWk) and enhancer_biv (+ EnhBiv; this is the deck's stored enhancer matrix),
# both checked against the stored deck SpQN matrix; repressive
# (ReprPC + ReprPCWk + Het + ZNF/Rpts), polycomb (ReprPC + ReprPCWk), het (Het), znf_rpts (ZNF/Rpts). Quies and TssBiv/EnhBiv excluded.
import sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
X = Path("/work/users/d/y/dyy12/XuLab"); TFC = X / "tf_complex"; O = X / "regulatome/repressive_chromhmm_matrix"
sys.path.insert(0, str(TFC / "scripts")); sys.path.insert(0, str(X / "downstream"))
from compartment_rpkm_correlation import load_cells, spqn, _confound
tfs = [t.strip() for t in (TFC / "results_compartment_counts_spqn_domain_sm2/tf_similarity_genome.tsv").read_text().splitlines()[0].split("\t")[1:] if t.strip()]
cells = load_cells(X / "data/TF1000cells.meta.csv"); depth = np.array([cells[t.lower()] for t in tfs], float)
W = TFC / "work/chromhmm_elements_counts"
assert (W / "tfs.txt").read_text().split() == tfs, "element matrix TF order differs"
M = sp.load_npz(W / "element_matrix.npz").tocsr()
E = pd.read_csv(W / "elements.tsv", sep="\t"); state = E.state.values; bp = E.bp.values
SUB = {"enhancer": ["EnhA1", "EnhA2", "EnhG1", "EnhG2", "EnhWk"], "enhancer_biv": ["EnhA1", "EnhA2", "EnhG1", "EnhG2", "EnhWk", "EnhBiv"], "repressive": ["ReprPC", "ReprPCWk", "Het", "ZNF/Rpts"],
       "polycomb": ["ReprPC", "ReprPCWk"], "het": ["Het"], "znf_rpts": ["ZNF/Rpts"]}
def pct(R):
    off = ~np.eye(len(R), dtype=bool); v = np.sort(R[off]); P = np.searchsorted(v, R, side="right") / len(v); np.fill_diagonal(P, 1.0); return P
info = []
for name, st in SUB.items():
    m = np.isin(state, st); X_ = np.asarray(M[m].todense()); keep = X_.sum(1) > 0; X_ = X_[keep]
    R = np.nan_to_num(np.corrcoef(X_, rowvar=False)); Rn = spqn(R, depth)
    for k, A in (("pearson", R), ("spqn", Rn), ("pct", pct(Rn))):
        pd.DataFrame(A, index=tfs, columns=tfs).to_csv(O / f"{k}_{name}.tsv", sep="\t", float_format="%.5f")
    off = ~np.eye(len(tfs), dtype=bool)
    info.append(dict(set=name, states="+".join(st), elements=int(m.sum()), elements_with_reads=int(keep.sum()), bp=int(bp[m].sum()),
                     median_offdiag_pearson=float(np.median(R[off])), median_offdiag_spqn=float(np.median(Rn[off])),
                     depth_confound_pearson=round(_confound(R, depth), 3), depth_confound_spqn=round(_confound(Rn, depth), 3)))
    if name in ("enhancer", "enhancer_biv"):
        lines = (TFC / "results_chromhmm_elements_counts_spqn_enhancer/spqn_similarity_enhancer.tsv").read_text().splitlines()
        lab = lines[0].split("\t")[1:]; S = np.array([[float(v) for v in l.split("\t")[1:]] for l in lines[1:]])
        S = S[np.ix_([lab.index(t) for t in tfs], [lab.index(t) for t in tfs])]
        print(f"{name} check vs stored deck SpQN matrix: max |diff| = {np.abs(S - Rn).max():.2e}", flush=True)
I = pd.DataFrame(info); I.to_csv(O / "element_sets.tsv", sep="\t", index=False, float_format="%.4f")
pd.DataFrame({"tf": tfs, "cells": depth.astype(int)}).to_csv(O / "tf_cells.tsv", sep="\t", index=False)
pd.set_option("display.width", 220); print(I.to_string(index=False))
