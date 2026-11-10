# Extra label permutations for one matched pair (default DDIT3-NFYA in triple_support/): reads the matched discovery pools and shuffles
# pair / A-only / B-only labels within the GC x CpG x promoter strata, writing perm/<k> for k in [START, END).
import os, sys
import numpy as np, pandas as pd
O = os.path.dirname(os.path.abspath(__file__)); B = os.environ.get("BASE", f"{O}/triple_support"); pair = sys.argv[1]; k0, k1 = int(sys.argv[2]), int(sys.argv[3])
Q = pd.read_csv(f"{O}/ref/bins_qc.tsv", sep="\t"); pos = {f"b{b}": i for i, b in enumerate(Q.bin)}
prom = np.zeros(len(Q), bool); prom[[int(x) for x in open(f"{O}/ref/promoter_bins.txt").read().split()]] = True
num = Q.chrom.str.replace("chr", ""); ok = Q.n_ok & ~Q.blacklist & (Q.exonic <= 0.5) & num.isin([str(i) for i in range(1, 23)] + ["X"])
cq = np.quantile(Q.cpg_oe.values[ok.values], [0.2, 0.4, 0.6, 0.8]); strat = (np.floor(Q.gc.values / 0.025).astype(int) * 10 + np.searchsorted(cq, Q.cpg_oe.values)) * 2 + prom
L = {k: [l for l in open(f"{B}/pools/{pair}/disc/{k}.bed")] for k in ["pair", "Aonly", "Bonly"]}
U = sum(L.values(), []); lab = np.repeat([2, 0, 1], [len(L["pair"]), len(L["Aonly"]), len(L["Bonly"])]); st = np.array([strat[pos[l.split("\t")[3]]] for l in U]); rng = np.random.default_rng(1000 + k0)
for k in range(k0, k1):
    x = lab.copy()
    for s in np.unique(st): q = np.flatnonzero(st == s); x[q] = rng.permutation(x[q])
    d = f"{B}/pools/{pair}/perm/{k:03d}"; os.makedirs(d, exist_ok=True)
    for nm, v in [("pair", 2), ("Aonly", 0), ("Bonly", 1)]: open(f"{d}/{nm}.bed", "w").writelines(U[i] for i in np.flatnonzero(x == v))
print("written", k1 - k0)
