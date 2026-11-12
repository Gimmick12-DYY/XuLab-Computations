# CAP-SELEX 2025 (Xie et al., Nature 641:1329) Supplementary Table S2 section a: TF1 (row) x TF2 (column) matrix; 0 = tested, no cooperative
# signal; 1 = composite motif; 2 = spacing / orientation preference; blank = not tested. Lookup for our AF3-supported frozen candidates and
# for all frozen candidates / known pairs. Output external/capselex2025/capselex_lookup.tsv
import pandas as pd, numpy as np
X = "/work/users/d/y/dyy12/XuLab"; E = f"{X}/complex_screen/pair_motif_composite/external/capselex2025"
d = pd.read_excel(f"{E}/MOESM4.xlsx", header=None); s = d.index[d.iloc[:, 0].astype(str).str.startswith("Section a: Interaction")][0]; e = d.index[d.iloc[:, 0].astype(str).str.startswith("Section b  :")][0]
blk = d.iloc[s + 1:e].dropna(how="all").dropna(axis=1, how="all"); print(blk.iloc[:3, :8].to_string())
# find the header row: first row whose cells are mostly gene-like strings
hdr = blk.iloc[0]; M = blk.iloc[1:].copy(); M.index = M.iloc[:, 0].astype(str).str.upper().str.strip(); M = M.iloc[:, 1:]; M.columns = [str(c).upper().strip() for c in hdr.iloc[1:]]
print("matrix", M.shape, "rows e.g.", list(M.index[:5]), "cols e.g.", list(M.columns[:5]))
alias = {"HP1": ["CBX3", "CBX1", "CBX5"], "ZUFSP": ["ZUP1"]}
tested_tfs = set(M.index) | set(M.columns)
def look(a, b):
    best = None
    for A in alias.get(a, [a]):
        for B in alias.get(b, [b]):
            for r, c in [(A, B), (B, A)]:
                if r in M.index and c in M.columns:
                    v = M.at[r, c]
                    v = v.iloc[0] if isinstance(v, pd.Series) else v
                    if pd.notna(v) and str(v).strip() != "": v = str(v).strip(); best = v if best is None or v in ("1", "2") else best
    return best if best is not None else "not_tested"
F = pd.read_csv(f"{X}/complex_screen/candidates_frozen_v1/candidates_all3081.tsv", sep="\t")
F["af3"] = [b if b != "not_run" else a for a, b in zip(F.af3_1seed_tier, F.af3_5seed_tier)]
F["capselex"] = [look(a, b) for a, b in zip(F.tf_a, F.tf_b)]
F["a_in_screen"] = [any(x in tested_tfs for x in alias.get(a, [a])) for a in F.tf_a]; F["b_in_screen"] = [any(x in tested_tfs for x in alias.get(b, [b])) for b in F.tf_b]
F[["pair", "tf_a", "tf_b", "best_pct", "known", "known_reason", "candidate", "af3", "capselex", "a_in_screen", "b_in_screen"]].to_csv(f"{E}/capselex_lookup.tsv", sep="\t", index=False)
lab = {"0": "tested, no cooperativity", "1": "composite motif", "2": "spacing/orientation"}
for name, sub in [("AF3 strong/weak candidates", F[(F.candidate.astype(str) == "True") & F.af3.isin(["strong", "weak"])]), ("all frozen candidates", F[F.candidate.astype(str) == "True"]), ("all 3,081 panel pairs", F)]:
    print(f"== {name}: n={len(sub)}; both TFs in CAP-SELEX screen: {int((sub.a_in_screen & sub.b_in_screen).sum())}"); print(sub.capselex.map(lambda v: lab.get(v, v)).value_counts().to_string())
s = F[F.capselex.isin(["1", "2"])]; print(s[["pair", "best_pct", "known_reason", "candidate", "af3", "capselex"]].to_string(index=False))
print("panel TFs in screen:", sorted({t for t in set(F.tf_a) | set(F.tf_b) if any(x in tested_tfs for x in alias.get(t, [t]))}))
