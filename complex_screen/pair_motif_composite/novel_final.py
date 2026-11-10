# B4 deliverable: one table of novel pairs (pct >= 0.60) joining the motif layers tried overnight, each with its random-pair null:
#   raw-summit known sharing (per_tf_raw/novel_pairs_ranked.tsv), reversed-design sharing (reversed/pairs.tsv),
#   de novo family sharing (per_tf_raw/denovo_pairs.tsv) and the Track A co-bound test (triple_support/gate_levels.tsv via novel_pairs.py).
# Rank: AF3 strong, weak, then the rest; within that, the number of layers with empirical p <= 0.05, then the best empirical p, then pct.
# Output: novel_pairs_ranked.tsv and controls.tsv in pair_motif_composite/
import os
import numpy as np, pandas as pd
O = os.path.dirname(os.path.abspath(__file__))
def key(p): return "-".join(sorted(p.split("-")))
A = pd.read_csv(f"{O}/per_tf_raw/novel_pairs_ranked.tsv", sep="\t"); A["k"] = A.pair.map(key)
R = pd.read_csv(f"{O}/reversed/pairs.tsv", sep="\t"); R["k"] = R.pair.map(key)
N = pd.read_csv(f"{O}/per_tf_raw/denovo_pairs.tsv", sep="\t"); N["k"] = N.pair.map(key)
C = pd.read_csv(f"{O}/per_tf_raw/controls.tsv", sep="\t"); C["k"] = C.pair.map(key)
def join(T):
    T = T.rename(columns={"n_shared_strict": "raw_n_strict", "n_shared_loose": "raw_n_loose", "emp_p_strict": "raw_emp_strict", "emp_p_loose": "raw_emp_loose", "hyper_p_strict": "raw_hyper_strict",
                          "hyper_p_loose": "raw_hyper_loose", "shared_strict": "raw_shared_strict", "shared_loose_only": "raw_shared_loose_only", "shared_loose_val_even_both": "raw_shared_val_even_both", "denovo_shared": "raw_denovo_similar"})
    T = T.merge(R[["k", "n_strict", "emp_strict", "n_loose", "emp_loose", "shared_strict"]].rename(columns=lambda c: c if c == "k" else f"rev_{c}"), on="k", how="left")
    T = T.merge(N[["k", "n_shared", "n_shared_novel", "emp_p", "shared"]].rename(columns=lambda c: c if c == "k" else f"denovo_{c}"), on="k", how="left")
    e = T[["raw_emp_strict", "raw_emp_loose", "rev_emp_strict", "rev_emp_loose", "denovo_emp_p"]]
    T["layers_emp_le05"] = ((T[["raw_emp_strict", "raw_emp_loose"]].min(1) <= .05).astype(int) + (T[["rev_emp_strict", "rev_emp_loose"]].min(1) <= .05).astype(int) + (T.denovo_emp_p <= .05).astype(int))
    T["best_emp_p"] = e.min(1); return T
T = join(A); T["_a"] = T.af3_tier.map({"strong": 0, "weak": 1}).fillna(2)
T = T.sort_values(["_a", "layers_emp_le05", "best_emp_p", "pct"], ascending=[True, False, True, False]).drop(columns=["_a", "k"])
first = ["pair", "corr_tier", "pct", "pearson_domain_pct", "spqn_noquies_pct", "af3_tier", "af3_iptm", "af3_ipsae", "af3_source", "layers_emp_le05", "best_emp_p",
         "raw_n_strict", "raw_emp_strict", "raw_n_loose", "raw_emp_loose", "raw_hyper_strict", "rev_n_strict", "rev_emp_strict", "rev_n_loose", "rev_emp_loose", "denovo_n_shared", "denovo_n_shared_novel", "denovo_emp_p",
         "cobound_ownA", "cobound_ownB", "cobound_strict", "cobound_relaxed", "cobound_composite_validated"]
first = [c for c in first if c in T.columns]; T = T[first + [c for c in T.columns if c not in first]]
T.to_csv(f"{O}/novel_pairs_ranked.tsv", sep="\t", index=False); CT = join(C).drop(columns="k"); CT = CT[[c for c in first if c in CT.columns] + [c for c in CT.columns if c not in first]]; CT.to_csv(f"{O}/controls.tsv", sep="\t", index=False)
rates = T.groupby("corr_tier").apply(lambda s: pd.Series(dict(n=len(s), raw=(s[["raw_emp_strict", "raw_emp_loose"]].min(1) <= .05).mean(), reversed=(s[["rev_emp_strict", "rev_emp_loose"]].min(1) <= .05).mean(),
                                                               denovo=(s.denovo_emp_p <= .05).mean(), two_plus_layers=(s.layers_emp_le05 >= 2).mean())), include_groups=False).round(3)
print(rates.to_string()); print("two+ layers among all panel pairs is the null reference; per-layer null = 0.05 (two-sided union of strict/loose ~0.08)")
pd.set_option("display.width", 250); print(T.head(20)[["pair", "corr_tier", "pct", "af3_tier", "af3_ipsae", "layers_emp_le05", "best_emp_p", "raw_n_strict", "rev_n_strict", "denovo_n_shared", "cobound_ownB", "cobound_relaxed"]].to_string())
print(CT[["pair", "pct", "af3_tier", "layers_emp_le05", "best_emp_p", "raw_n_strict", "rev_n_strict", "denovo_n_shared"]].to_string())
