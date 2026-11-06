# Gate 1 summary: TF x representation pass table (obs > null p90 and BH q <= 0.10 across TFs), undetermined = shallow fails,
# sanity checks, pass rate vs cell count plot
import glob, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
G = "/work/users/d/y/dyy12/XuLab/regulatome/gate1"
R = pd.concat([pd.read_csv(f, sep="\t") for f in glob.glob(f"{G}/results/*.tsv") if not f.endswith(".null.tsv")])
SHALLOW = R.groupby("tf").n_cells.first().median()
def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out
R["q"] = R.groupby(["rep", "arm"]).p.transform(bh)
R["call"] = np.where((R.obs > R.null_p90) & (R.q <= 0.10), "pass", np.where(R.n_cells < SHALLOW, "undetermined", "fail"))
# imputed values come from one model per TF trained on ALL its cells (shared per-bin sequence encoder + per-cell embedding),
# so half A's imputed matrix already carries information from half B: imputed and blend calls are not split-independent
R["split_independent"] = ~R.rep.isin(["imputed", "blend"])
R["note"] = np.where(R.rep == "blend", "blend (reported, not used: z-blend lifts the null)",
                     np.where(R.rep == "imputed", "not independent", "."))
R.loc[~R.split_independent & (R.call == "pass"), "call"] = "pass_not_independent"
R.to_csv(f"{G}/gate1_all.tsv", sep="\t", index=False, float_format="%.4f")
T = R.pivot_table(index=["tf", "n_cells", "group"], columns=["rep", "arm"], values="call", aggfunc="first")
T.columns = [f"{a}|{b}" for a, b in T.columns]; T = T.reset_index().sort_values("n_cells")
T.to_csv(f"{G}/gate1_pass_table.tsv", sep="\t", index=False)
print(f"TFs {R.tf.nunique()}; shallow = cells < {SHALLOW:.0f}")
print(R.groupby(["rep", "arm"]).call.value_counts().unstack(fill_value=0).to_string())
for t in ["NRF1", "NFYA", "CTCF", "MAZ"]:
    if t in set(R.tf): print(t, R[(R.tf == t)][["rep", "arm", "obs", "null_p90", "p", "call"]].round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(6, 4))
for (rep, arm), d in R[R.arm == "abc"].groupby(["rep", "arm"]):
    d = d.sort_values("n_cells"); bins = pd.qcut(np.log10(d.n_cells), 5, duplicates="drop")
    pr = d.groupby(bins, observed=True).apply(lambda x: (x.call == "pass").mean())
    ax.plot([10 ** i.mid for i in pr.index], pr.values, marker="o", label=rep)
ax.set_xscale("log"); ax.set_xlabel("cells per TF"); ax.set_ylabel("pass rate (ABC arm)"); ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(f"{G}/gate1_pass_vs_cells.png", dpi=200)
