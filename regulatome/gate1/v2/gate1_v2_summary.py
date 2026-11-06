# Gate 1 v2 summary: reproducibility (above null p90, BH q <= 0.10) AND specificity (own half A vs own half B beats every
# other TF's half B; margin = own - best other, > 0 in >= 18/20 splits and on average), NEUROD4 floor, v1 vs v2, modules.
import glob, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
G = "/work/users/d/y/dyy12/XuLab/regulatome/gate1"; V = f"{G}/v2"

def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out

R = pd.concat([pd.read_csv(f, sep="\t") for f in glob.glob(f"{V}/results/*.tsv") if not f.endswith(".null.tsv")])
R["q"] = R.groupby(["rep", "arm"]).p.transform(bh)
R["reproducible"] = (R.obs > R.null_p90) & (R.q <= 0.10)

# specificity from saved raw x ABC half profiles (rank-transform once; Pearson of ranks = Spearman)
prof = {}
for f in glob.glob(f"{V}/results/*.profiles.npz"):
    z = np.load(f); t = f.split("/")[-1].replace(".profiles.npz", "")
    rk = lambda M: np.apply_along_axis(rankdata, 1, M)
    prof[t] = (rk(z["A"]), rk(z["B"]))
tfs = sorted(prof); K = min(prof[t][0].shape[0] for t in tfs)
def zs(M): M = M - M.mean(1, keepdims=True); return M / np.linalg.norm(M, axis=1, keepdims=True)
A = np.stack([zs(prof[t][0][:K]) for t in tfs]); B = np.stack([zs(prof[t][1][:K]) for t in tfs])  # tf x split x gene
spec = []
for i, t in enumerate(tfs):
    own = np.einsum("kg,kg->k", A[i], B[i])
    oth = np.einsum("kg,ukg->uk", A[i], B); oth[i] = -np.inf
    best = oth.max(0); who = np.array(tfs)[oth.argmax(0)]
    m = own - best
    spec.append(dict(tf=t, own_rho=own.mean(), best_other_rho=best.mean(), best_other_tf=pd.Series(who).mode()[0],
                     margin=m.mean(), splits_margin_pos=int((m > 0).sum()), n_splits=K))
S = pd.DataFrame(spec); S["specific"] = (S.margin > 0) & (S.splits_margin_pos >= int(np.ceil(0.9 * K)))
R = R.merge(S, left_on="unit", right_on="tf", how="left")
SHALLOW = R.groupby("unit").n_cells.first().median()
gate_arm = R.rep == "raw"
R["call"] = np.where(R.rep == "atac_only", np.where(R.reproducible, "pass", "fail"),
            np.where(R.reproducible & R.specific.fillna(False), "pass",
            np.where(R.reproducible, "reproducible_not_specific", np.where(R.n_cells < SHALLOW, "undetermined", "fail"))))
fl = R[(R.unit == "NEUROD4") & (R.rep == "raw")].set_index("arm")[["obs", "excess"]]
R["below_neurod4_floor"] = [(r.rep == "raw") and (r.arm in fl.index) and (r.excess <= fl.loc[r.arm, "excess"]) and r.call == "pass"
                            for r in R.itertuples()]
R.to_csv(f"{V}/gate1_v2_all.tsv", sep="\t", index=False, float_format="%.4f")
S.to_csv(f"{V}/gate1_v2_specificity.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 250)
print(f"TFs {R.unit.nunique()}; shallow = cells < {SHALLOW:.0f}")
print(R.groupby(["rep", "arm"]).call.value_counts().unstack(fill_value=0).to_string())
print("NEUROD4 floor (raw):", fl.round(3).to_dict("index"))
ra = R[(R.rep == "raw") & (R.arm == "abc")].sort_values("n_cells")
print("raw x ABC passes:", ", ".join(f"{t}({n}{'*' if b else ''})" for t, n, b in zip(ra[ra.call == 'pass'].unit, ra[ra.call == 'pass'].n_cells, ra[ra.call == 'pass'].below_neurod4_floor)), "(* = excess at/below NEUROD4)")
print("reproducible but not specific:", ", ".join(f"{t}(best other {b}, margin {m:.3f})" for t, b, m in zip(ra[ra.call == 'reproducible_not_specific'].unit, ra[ra.call == 'reproducible_not_specific'].best_other_tf, ra[ra.call == 'reproducible_not_specific'].margin)))
v1 = pd.read_csv(f"{G}/gate1_all.tsv", sep="\t"); v1 = v1[(v1.rep == "raw") & (v1.arm == "abc")].set_index("tf").call
cmp = ra.set_index("unit").call.rename("v2").to_frame().join(v1.rename("v1"))
print("v1 vs v2 (raw x ABC):\n", pd.crosstab(cmp.v1, cmp.v2).to_string())
print("dropped (v1 pass -> v2 not pass):", ", ".join(cmp[(cmp.v1 == "pass") & (cmp.v2 != "pass")].index))
print("gained (v1 not pass -> v2 pass):", ", ".join(cmp[(cmp.v1 != "pass") & (cmp.v2 == "pass")].index))
for t in ["NFYA", "CTCF", "MAZ", "NRF1", "NEUROD4"]:
    print(t, ra[ra.unit == t][["n_cells", "n_eff", "obs", "null_p90", "p", "q", "margin", "splits_margin_pos", "best_other_tf", "call"]].round(3).to_string(index=False, header=(t == "NFYA")))
ra = ra.copy(); ra["bin"] = pd.qcut(ra.n_cells, 5)
print(ra.groupby("bin", observed=True).call.value_counts().unstack(fill_value=0).to_string())
fig, ax = plt.subplots(figsize=(6, 4))
for arm, d in R[R.rep == "raw"].groupby("arm"):
    d = d.sort_values("n_cells"); b = pd.qcut(np.log10(d.n_cells), 5)
    pr = d.groupby(b, observed=True).apply(lambda x: (x.call == "pass").mean())
    ax.plot([10 ** i.mid for i in pr.index], pr.values, marker="o", label=f"raw x {arm}")
ax.set_xscale("log"); ax.set_xlabel("cells per TF"); ax.set_ylabel("pass rate (reproducible and specific)"); ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(f"{V}/gate1_v2_pass_vs_cells.png", dpi=200)

M = [pd.read_csv(f, sep="\t") for f in glob.glob(f"{V}/results_modules/*.tsv") if not f.endswith(".null.tsv")]
if M:
    M = pd.concat(M); M = M[M.rep == "raw"]
    M["loo"] = M.unit.str.contains("__minus_")
    M["q"] = M.groupby(["loo", "arm"]).p.transform(bh)
    M["call"] = np.where((M.obs > M.null_p90) & (M.q <= 0.10), "pass", "fail")
    M["flag"] = np.where(M.members == "DDIT3,E2F3,SAFB,XPA,ZNF703,ZNF768", "batch_module (= high-library group)", ".")
    M.to_csv(f"{V}/gate1_v2_modules.tsv", sep="\t", index=False, float_format="%.4f")
    print(M.sort_values(["loo", "arm", "unit"])[["unit", "arm", "members", "n_cells", "n_eff", "obs", "null_p90", "q", "call", "flag"]].round(3).to_string(index=False))
