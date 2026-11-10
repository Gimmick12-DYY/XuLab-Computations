# Interface audit, Amendment 2: af3_status (af3_not_tested / af3_tested_not_strong / af3_strong) replaces boltz_only_no_interface_support;
# boltz_note kept as its own column. Reads audit.tsv, flags.tsv, triple_support_flagged.tsv; writes flags_v2.tsv, triple_support_flagged_v2.tsv.
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab/complex_screen"; O = f"{X}/interface_audit_20261110"; S = f"{X}/af3_corr060_20261104"
hp = lambda t: "HP1" if t in ("CBX1", "CBX3", "CBX5") else t
key = lambda p: "__".join(sorted(hp(t) for t in p.replace("-", "__").split("__")))
A = pd.read_csv(f"{O}/audit.tsv", sep="\t"); A["panel"] = A.pair.map(key); af = A[(A.tool == "AF3") & A.tier.isin(["strong", "weak", "none"])].copy()
rk = {"strong": 2, "weak": 1, "none": 0}; af["rk"] = af.tier.map(rk); af["seeds"] = af.n_models_used
af = af.sort_values(["rk", "ipsae"], ascending=False); B = af.drop_duplicates("panel").set_index("panel")
submitted = set(af.panel) | {key(l.strip()) for l in open(f"{S}/seeds5/all.txt") if l.strip()}
def status(k):
    if k in B.index: return "af3_strong" if B.loc[k, "tier"] == "strong" else "af3_tested_not_strong"
    return "af3_submitted_unscored" if k in submitted else "af3_not_tested"
def info(k):
    if k not in B.index: return dict(af3_best_tier="", af3_best_source="", af3_best_ipsae=np.nan, af3_best_iptm=np.nan, af3_best_seeds=np.nan, af3_best_n_seeds_strong=np.nan)
    r = B.loc[k]; return dict(af3_best_tier=r.tier, af3_best_source=r.source, af3_best_ipsae=r.ipsae, af3_best_iptm=r.iptm, af3_best_seeds=r.seeds, af3_best_n_seeds_strong=r.get("n_seeds_strong", np.nan))
NOTE = "Boltz score not interface evidence (CIR unrelated to Boltz ipSAE/ipTM; Amendment 1.3)"
F = pd.read_csv(f"{O}/flags.tsv", sep="\t").drop(columns=["boltz_only_no_interface_support"])
F["af3_status"] = F.panel.map(status); F = pd.concat([F, pd.DataFrame([info(k) for k in F.panel], index=F.index)], axis=1)
F["boltz_note"] = np.where(F.tool == "Boltz", NOTE, ""); F.to_csv(f"{O}/flags_v2.tsv", sep="\t", index=False)
T = pd.read_csv(f"{O}/triple_support_flagged.tsv", sep="\t", keep_default_na=False); was = T.boltz_only_no_interface_support.astype(str) == "True"
T = T.drop(columns=["boltz_only_no_interface_support"]); k = T.pair.map(key)
T["af3_status"] = k.map(status); T = pd.concat([T, pd.DataFrame([info(x) for x in k], index=T.index)], axis=1)
T["boltz_note"] = np.where((T.boltz_trimmed == "strong") | (T.boltz_fulllength == "strong"), NOTE, ""); T.to_csv(f"{O}/triple_support_flagged_v2.tsv", sep="\t", index=False)
bz = F[F.tool == "Boltz"]; print("(a) Boltz strong rows", len(bz)); print(bz.af3_status.value_counts().to_string()); print(bz[bz.af3_status == "af3_tested_not_strong"].af3_best_tier.value_counts().to_string())
print("panel pairs:", bz.drop_duplicates("panel").af3_status.value_counts().to_dict())
print("(b) v1 Boltz-only deck rows", int(was.sum()))
print(T[was][["pair", "tier", "af3_status", "af3_best_tier", "af3_best_source", "af3_best_ipsae", "af3_best_iptm", "af3_best_seeds", "af3_best_n_seeds_strong"]].to_string(index=False))
