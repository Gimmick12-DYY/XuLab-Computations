# Descriptive companion to the held-out table (not part of the pass condition): own-KD z minus mean z over non-partner KDs (kNN5 partners from the
# binding-only graph, alpha = 0 lists), ceiling-weighted (b3b4 weights), bootstrap over TFs 10,000, seed 20261116; Arm 2 and Arm 1, with and
# without NRF1; dev (net1b/per_tf.tsv, dev KDs) and held-out (heldout_v1/per_tf.tsv, held-out KDs). Writes heldout_v1/own_vs_nonpartner.tsv.
import numpy as np, pandas as pd
O = "/work/users/d/y/dyy12/XuLab/regulatome/m6_improve"; C = pd.read_csv(f"{O}/../m4_validation/b3b4/reliability_ceilings.tsv", sep="\t"); out = []
SRC = {"dev": (f"{O}/net1b/per_tf.tsv", {"win100k raw": "win100k|alpha0", "ABC-sum": "abc|alpha0"}), "heldout": (f"{O}/heldout_v1/per_tf.tsv", {"win100k raw": "win100k_raw@1000", "ABC-sum": "abc_raw@1000"})}
for split, (f, V) in SRC.items():
    P = pd.read_csv(f, sep="\t")
    for excl in (False, True):
        rng = np.random.default_rng(20261116)
        for a in ("arm2", "arm1"):
            w = C[C.arm == a].groupby("panel_tf").weight.mean()
            for lab, v in V.items():
                G = P[(P.arm == a) & (P.variant == v) & ~(excl & (P.tf == "NRF1"))]; d = (G.z_own - G.z_nonpartner).values; wt = w.reindex(G.tf).fillna(0).values
                if len(d) == 0: continue
                ix = rng.integers(0, len(d), (10000, len(d))); b = (d[ix] * wt[ix]).sum(1) / np.maximum(wt[ix].sum(1), 1e-12)
                out.append(dict(split=split, nrf1="excluded" if excl else "included", arm=a, score=lab, n_tfs=len(d), n_weighted=int((wt > 0).sum()), own_minus_nonpartner=round(np.average(d, weights=wt), 3) if wt.sum() > 0 else np.nan, ci=f"[{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]"))
pd.DataFrame(out).to_csv(f"{O}/heldout_v1/own_vs_nonpartner.tsv", sep="\t", index=False)
