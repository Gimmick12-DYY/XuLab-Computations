# Interface audit, Amendment 1 (README): min-chain CIR flag (< 5, fixed) and Boltz-only column. Reads audit.tsv; writes flags.tsv and
# triple_support_flagged.tsv (copy; triple_support_20261109/triple_support.tsv untouched). No row is removed or relabelled.
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab/complex_screen"; O = f"{X}/interface_audit_20261110"; THR = 5
A = pd.read_csv(f"{O}/audit.tsv", sep="\t").drop(columns=["audit_label"])  # strong_noCIR relabel withdrawn (Amendment 1.1)
hp = lambda t: "HP1" if t in ("CBX1", "CBX3", "CBX5") else t
key = lambda p: "__".join(sorted(hp(t) for t in p.replace("-", "__").split("__")))
A["panel"] = A.pair.map(key)
def weak_side(r):
    a, b = r.pair.split("__")
    if pd.isna(r.CIR_a): return ""
    return a if r.CIR_a < r.CIR_b else (b if r.CIR_b < r.CIR_a else f"{a}/{b}")
A["minchain"] = A[["CIR_a", "CIR_b"]].min(axis=1); A["minchain_tf"] = A.apply(weak_side, axis=1)
note = lambda m, tf: f"one-sided interface, peptide-scale ({int(m)} confident residues on {tf})" if pd.notna(m) and m < THR else ""
af3_strong = set(A[(A.tool == "AF3") & (A.tier == "strong")].panel)
F = A[A.tier == "strong"].copy()
F["footnote"] = [note(m, t) for m, t in zip(F.minchain, F.minchain_tf)]
F["boltz_only_no_interface_support"] = (F.tool == "Boltz") & ~F.panel.isin(af3_strong)
F[["tool", "source", "pair", "panel", "tier", "iptm", "ipsae", "CIR", "CIR_a", "CIR_b", "minchain", "minchain_tf", "footnote",
   "boltz_only_no_interface_support", "CIR_per_seed"]].assign(DESCRIPTIVE="DESCRIPTIVE").to_csv(f"{O}/flags.tsv", sep="\t", index=False)
T = pd.read_csv(f"{X}/triple_support_20261109/triple_support.tsv", sep="\t", keep_default_na=False); T["_k"] = T.pair.map(key)
SRC = {"sweep_s1": lambda s: s == "af3_s1_sweep", "af3_local": lambda s: s.startswith("af3_local"), "af3_novel_20261103": lambda s: s.startswith("af3_novel")}
def best(k, pred, tool="AF3"):
    g = A[(A.panel == k) & (A.tool == tool) & A.source.map(pred) & A.minchain.notna()]
    if not len(g): return np.nan, ""
    r = g.loc[g.minchain.idxmax()]; return r.minchain, r.minchain_tf
mc, fn, bo = [], [], []
for r in T.itertuples():
    s1 = r.af3_tier == "strong"; s5 = r.af3_s5_tier == "strong"; boltz = "strong" in (r.boltz_trimmed, r.boltz_fulllength)
    m, tf = best(r._k, SRC[r.af3_source]) if r.af3_source in SRC else (np.nan, "")
    if s5: m5, tf5 = best(r._k, lambda s: s == "af3_s5"); m, tf = (m5, tf5) if not s1 or np.isnan(m) else (m, tf)
    only = boltz and not (s1 or s5)
    if only: m, tf = best(r._k, lambda s: s == "boltz_screen", "Boltz")
    mc.append(m); fn.append(note(m, tf) if (s1 or s5 or only) else ""); bo.append(only)
T = T.drop(columns="_k"); T["af3_minchain_CIR"] = mc; T["interface_footnote"] = fn; T["boltz_only_no_interface_support"] = bo
T.to_csv(f"{O}/triple_support_flagged.tsv", sep="\t", index=False)
print("flags rows", len(F), "footnoted", (F.footnote != "").sum(), "boltz_only rows", int(F.boltz_only_no_interface_support.sum()),
      "boltz_only panel pairs", F[F.boltz_only_no_interface_support].panel.nunique())
print(F[F.footnote != ""][["tool", "source", "pair", "footnote"]].to_string(index=False))
print("triple: footnoted", (T.interface_footnote != "").sum(), "boltz_only", int(T.boltz_only_no_interface_support.sum()))
print(T[(T.interface_footnote != "") | T.boltz_only_no_interface_support][["pair", "tier", "af3_tier", "af3_s5_tier", "boltz_trimmed", "boltz_fulllength", "af3_minchain_CIR", "interface_footnote", "boltz_only_no_interface_support"]].to_string(index=False))
