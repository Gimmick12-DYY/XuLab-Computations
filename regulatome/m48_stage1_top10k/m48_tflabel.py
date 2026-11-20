# m48: TF-label shuffle for the top 10,000 lists: each TF's Perturb-seq genes paired with another TF's Regulome list (derangement among the 63, and within list-size quintiles), pooled recall.
import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"; M = f"{R}/m48_stage1_top10k"
s = open(f"{M}/m48_tables.py").read(); exec(s[:s.index("VERS = {")])
O = f"{M}/out"; rng = np.random.default_rng(20261118); NP = 1000; pval = lambda o, d: (1 + int((np.asarray(d) >= o).sum())) / (len(d) + 1); NT = len(TFS); EX3 = {"NRF1", "E4F1", "DNMT1"}
def derange(groups):
    perm = np.arange(NT)
    for g in groups:
        g = np.asarray(g)
        if len(g) < 2: continue
        while True:
            q = rng.permutation(g)
            if not (q == g).any(): break
        perm[g] = q
    return perm
rows = []
for pad in PADS:
    B = np.vstack([LISTS[pad][t] for t in TFS]); sizes = B.sum(1); qn = pd.qcut(pd.Series(sizes).rank(method="first"), 5, labels=False).values
    for c in (2, 3, 4, 5):
        ch = OK & (AZ >= c); MX = (ch.astype(np.int32) @ (B & OK.any(0)).astype(np.int32).T)  # MX[i, j] = TF i Perturb-seq genes in list j
        MX = np.array([[int((ch[i] & B[j]).sum()) for j in range(NT)] for i in range(NT)]); den = ch.sum(1)
        for ver, groups in (("any other TF (derangement)", [np.arange(NT)]), ("same list-size quintile", [np.flatnonzero(qn == k) for k in range(5)])):
            perms = [derange(groups) for _ in range(NP)]
            for sn, ex_ in (("all TFs", set()), ("without NRF1, E4F1, DNMT1", EX3)):
                sel = np.array([t not in ex_ for t in TFS]) & (den > 0); o = 100 * np.trace(MX)[()] if False else 100 * np.diag(MX)[sel].sum() / den[sel].sum()
                dd = np.array([100 * MX[np.arange(NT), pm][sel].sum() / den[sel].sum() for pm in perms])
                rows.append({"padding": pad, "cutoff": f"|z| >= {c}", "pairing": ver, "TFs": sn, "n TFs": int(sel.sum()), "Perturb-seq genes": int(den[sel].sum()), "recall own lists (%)": o, "shuffle mean (%)": dd.mean(),
                             "shuffle 2.5-97.5 (%)": f"{np.percentile(dd,2.5):.1f}-{np.percentile(dd,97.5):.1f}", "gap (points)": o - dd.mean(), "p (one-sided)": pval(o, dd)})
        print(pad, c, flush=True)
T = pd.DataFrame(rows); T.to_csv(f"{O}/8_tf_label_shuffle.tsv", sep="\t", index=False, float_format="%.4g")
pd.set_option("display.width", 250); print(T.round(3).to_string(index=False)); print("done")
