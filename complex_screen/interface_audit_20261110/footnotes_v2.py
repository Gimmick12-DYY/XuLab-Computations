import pandas as pd
O = "/work/users/d/y/dyy12/XuLab/complex_screen/interface_audit_20261110"; f = f"{O}/triple_support_flagged_v2.tsv"
T = pd.read_csv(f, sep="\t", keep_default_na=False)
FN = {"EPAS1-NEUROD4": "borderline: pair ipTM 0.390 against a 0.40 cut; 5/5 seeds strong per-seed",
      "RBBP4-RBBP7": "AF3 none is structurally expected: paralogues occupying the same position in NuRD/PRC2, mutually exclusive within a complex; correlation reflects shared complex membership, not contact (positive control for correlation ≠ contact)"}
for p, t in FN.items():
    m = T.pair == p; assert m.sum() == 1, p
    old = T.loc[m, "interface_footnote"].iloc[0]; T.loc[m, "interface_footnote"] = (old + "; " if old else "") + t
T.to_csv(f, sep="\t", index=False); print(T[T.pair.isin(FN)][["pair", "tier", "af3_tier", "interface_footnote"]].to_string(index=False))
