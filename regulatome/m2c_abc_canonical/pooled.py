# Pooled summary of overlap/<biosample>/per_tf.tsv (SPEC.md step 4): per arm x split set (development / heldout / all) x NRF1 (in / out):
# n tested, median OR, fraction OR > 1, sign test on OR > 1, n BH <= 0.10, TF-minus-null sign test (OR > null median OR), median OR - OR_ATAC with its
# sign test, n null_p <= 0.05; plus mean B6 effect and ceiling-weighted specificity (b3b4 weights, bootstrap 10,000, seed 20261116).
# Held-out TFs were used once before (m6 heldout_v1); they are reported as a separate split, no selection is made on these numbers.
import sys
import numpy as np, pandas as pd
from scipy import stats
X = "/work/users/d/y/dyy12/XuLab/regulatome"; O = f"{X}/m2c_abc_canonical/overlap/{sys.argv[1]}"
P = pd.read_csv(f"{O}/per_tf.tsv", sep="\t"); C = pd.read_csv(f"{X}/m4_validation/b3b4/reliability_ceilings.tsv", sep="\t"); out = []
sign = lambda x: float(stats.binomtest(int((x > 0).sum()), int((x != 0).sum()), .5).pvalue) if (x != 0).sum() else np.nan
for a in ["arm2", "arm1"]:
    w = C[C.arm == a].groupby("panel_tf").weight.mean()
    for st in ["development", "heldout", "all"]:
        for excl in (False, True):
            G = P[(P.arm == a) & (P.status == "tested") & ((P.set == st) if st != "all" else True) & ~(excl & (P.tf == "NRF1"))]
            if not len(G): continue
            rng = np.random.default_rng(20261116); e = G.specificity.values; wt = w.reindex(G.tf).fillna(0).values; ix = rng.integers(0, len(e), (10000, len(e)))
            bs = (e[ix] * wt[ix]).sum(1) / np.maximum(wt[ix].sum(1), 1e-12); ws = np.average(e, weights=wt) if wt.sum() > 0 else np.nan
            lo = np.log(G.OR.replace(0, np.nan).values)
            out.append(dict(arm=a, set=st, nrf1="excluded" if excl else "included", n_tested=len(G), n_untestable=int(((P.arm == a) & (P.status == "untestable_lt20") & ((P.set == st) if st != "all" else True)).sum()),
                median_targets=float(G.n_targets_universe.median()), median_OR=round(G.OR.median(), 3), frac_OR_gt1=round((G.OR > 1).mean(), 3), sign_p_OR_gt1=sign(G.OR.values - 1),
                n_BH_le_0_10=int((G.BH <= .10).sum()), n_OR_gt1_BH_le_0_10=int(((G.BH <= .10) & (G.OR > 1)).sum()),
                median_null_OR=round(G.null_OR_median.median(), 3), frac_OR_gt_null=round((G.OR > G.null_OR_median).mean(), 3), sign_p_vs_null=sign(G.OR.values - G.null_OR_median.values), n_null_p_le_0_05=int((G.null_p <= .05).sum()),
                median_OR_ATAC=round(G.OR_ATAC.median(), 3), median_OR_minus_ATAC=round(G.OR_minus_ATAC.median(), 3), frac_OR_gt_ATAC=round((G.OR_minus_ATAC > 0).mean(), 3), sign_p_vs_ATAC=sign(G.OR_minus_ATAC.values),
                mean_b6_effect=round(G.b6_effect_z_own.mean(), 3), wspec=round(ws, 3), wspec_ci=f"[{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]", median_own_rank=float(G.own_rank.median())))
S = pd.DataFrame(out); S.to_csv(f"{O}/pooled.tsv", sep="\t", index=False)
pd.set_option("display.width", 400); pd.set_option("display.max_columns", 40); print(S.to_string(index=False))
