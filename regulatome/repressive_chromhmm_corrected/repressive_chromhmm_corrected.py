# Accessibility/activity-corrected TF x TF matrices on ChromHMM element sets, using the per-domain ENRICHMENT RATIO from
# tf_complex/scripts/compartment_rpkm_correlation.py --transform enrichment (journal 2026-09-24; the run that confirmed PRC2,
# MAZ/ZNF143, cohesin, NuRD): signal per TF scaled to equal totals (RPKM-like; per-element length cancels), then each element /
# its across-TF mean (observed / expected), which removes the shared accessibility/activity profile every TF follows.
# Then Pearson across the subset's elements with any read, SpQN by cell count, percentile among off-diagonals (as uncorrected).
# Sensitivity: same, restricted to elements with >= 20 raw reads summed over TFs (obs/exp is noisy on 1-2 read elements).
import sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
X = Path("/work/users/d/y/dyy12/XuLab"); TFC = X / "tf_complex"; O = X / "regulatome/repressive_chromhmm_corrected"
sys.path.insert(0, str(TFC / "scripts")); sys.path.insert(0, str(X / "downstream"))
from compartment_rpkm_correlation import load_cells, spqn, _confound
tfs = [t.strip() for t in (TFC / "results_compartment_counts_spqn_domain_sm2/tf_similarity_genome.tsv").read_text().splitlines()[0].split("\t")[1:] if t.strip()]
cells = load_cells(X / "data/TF1000cells.meta.csv"); depth = np.array([cells[t.lower()] for t in tfs], float)
W = TFC / "work/chromhmm_elements_counts"
assert (W / "tfs.txt").read_text().split() == tfs
M = np.asarray(sp.load_npz(W / "element_matrix.npz").todense(), dtype=np.float64)   # reads / cell
raw = M * depth[None, :]                                                             # raw reads
E = pd.read_csv(W / "elements.tsv", sep="\t"); state = E.state.values; bp = E.bp.values
Mn = M / M.sum(0, keepdims=True) * 1e6                                               # equal per-TF totals (genome-wide)
SUB = {"enhancer_biv": ["EnhA1", "EnhA2", "EnhG1", "EnhG2", "EnhWk", "EnhBiv"], "repressive": ["ReprPC", "ReprPCWk", "Het", "ZNF/Rpts"],
       "polycomb": ["ReprPC", "ReprPCWk"], "het": ["Het"], "znf_rpts": ["ZNF/Rpts"]}
def pct(R):
    off = ~np.eye(len(R), dtype=bool); v = np.sort(R[off]); P = np.searchsorted(v, R, side="right") / len(v); np.fill_diagonal(P, 1.0); return P
off = ~np.eye(len(tfs), dtype=bool); info = []
for variant, minr in (("", 0), ("_min20", 20)):
    for name, st in SUB.items():
        m = np.isin(state, st); A = Mn[m]; keep = (A.sum(1) > 0) & (raw[m].sum(1) >= minr); A = A[keep]
        mu = A.mean(1, keepdims=True); mu[mu == 0] = 1.0; Xe = A / mu
        R = np.nan_to_num(np.corrcoef(Xe, rowvar=False)); Rn = spqn(R, depth)
        for k, Q in (("pearson", R), ("spqn", Rn), ("pct", pct(Rn))):
            pd.DataFrame(Q, index=tfs, columns=tfs).to_csv(O / f"data{variant}/{k}_{name}.tsv", sep="\t", float_format="%.5f")
        info.append(dict(variant=variant or "all", set=name, elements_used=int(keep.sum()), bp_used=int(bp[m][keep].sum()),
                         median_offdiag_pearson=float(np.median(R[off])), median_offdiag_spqn=float(np.median(Rn[off])),
                         depth_confound_pearson=round(_confound(R, depth), 3), depth_confound_spqn=round(_confound(Rn, depth), 3)))
        print(info[-1], flush=True)
I = pd.DataFrame(info); I.to_csv(O / "element_sets_corrected.tsv", sep="\t", index=False, float_format="%.4f")
pd.DataFrame({"tf": tfs, "cells": depth.astype(int)}).to_csv(O / "data/tf_cells.tsv", sep="\t", index=False)
pd.DataFrame({"tf": tfs, "cells": depth.astype(int)}).to_csv(O / "data_min20/tf_cells.tsv", sep="\t", index=False)
pd.set_option("display.width", 220); print(I.to_string(index=False))
