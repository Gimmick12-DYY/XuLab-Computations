# Gate 1 v2.1 summary (count-aware gene-level residual; genes with expected >= 5 in both halves).
# Reproducibility: obs > pooled-panel null p90 and BH q <= 0.10 (per-TF and module/LOO units separately).
# Specificity per split k: own = Spearman(r_A(u), r_B(u)) on u's eligible genes; vs comparator v: Spearman(r_A(u), r_B(v)) on
#   genes eligible in both; margin = own - max over comparators. TFs: comparators = deep TFs (>= 3,400 cells), self excluded.
#   Modules/LOO: comparators = non-member deep TFs + other full modules (margin_vs_others), and separately the unit's own
#   deep members (margin_vs_members: is the module more than its dominant member?). Specific = margin > 0 in >= 18/20 splits.
# Diagnostic: with ALL TFs as comparators, how often is ZNF503 (1,050 cells) the best other TF (v2 artifact check).
import glob, json, os
import numpy as np, pandas as pd
from scipy.stats import rankdata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
G = "/work/users/d/y/dyy12/XuLab/regulatome/gate1"; V = f"{G}/v2_1"; X = "/work/users/d/y/dyy12/XuLab"
EMIN, DEEP = 5.0, 3400
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.5, "svg.fonttype": "none", "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "legend.fontsize": 6, "savefig.dpi": 600, "xtick.major.width": 0.5, "ytick.major.width": 0.5})

def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out

def load(dirn):
    R = [pd.read_csv(f, sep="\t") for f in glob.glob(f"{V}/{dirn}/*.tsv")
         if not f.endswith(".null.tsv") and os.path.getsize(f) > 2]  # shallow TFs with no eligible genes write empty files
    return pd.concat(R) if R else pd.DataFrame()
T, M = load("results"), load("results_modules")
cells = {l.strip(): json.load(open(f"{X}/regulatome/m1_binding/v2/counts/{l.strip()}.json"))["cells"]
         for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()}

prof = {}
for f in glob.glob(f"{V}/results/*.profiles.npz") + glob.glob(f"{V}/results_modules/*.profiles.npz"):
    u = f.split("/")[-1].replace(".profiles.npz", ""); OE = np.load(f)["OE"]  # splits x 4 x genes
    prof[u] = OE
NS = {u: v.shape[0] for u, v in prof.items()}  # splits with >= 50 eligible genes (saved profiles)
def half(u, k, h):
    o, e = prof[u][k, 0 if h == "A" else 2], prof[u][k, 1 if h == "A" else 3]
    return np.log2((o + 0.5) / (e + 0.5)), e >= EMIN
def elig(u, k): return (prof[u][k, 1] >= EMIN) & (prof[u][k, 3] >= EMIN)
def rho(x, y, m):
    if m.sum() < 50: return np.nan
    return float(np.corrcoef(rankdata(x[m]), rankdata(y[m]))[0, 1])

deep_tfs = [t for t in cells if cells[t] >= DEEP and t in prof]
full_modules = [u for u in M.unit.unique() if "__minus_" not in u] if len(M) else []
def specificity(u, comps):
    own, best, who = [], [], []
    for k in range(NS[u]):
        ra, _ = half(u, k, "A"); rb, _ = half(u, k, "B"); m = elig(u, k)
        own.append(rho(ra, rb, m))
        # own and other on the SAME genes (eligible in u and in v's half B); margin = smallest (own - other) over comparators
        sc = {}
        for v in comps:
            if k >= NS[v]: continue
            rv, ev = half(v, k, "B"); g = m & ev
            o_, x_ = rho(ra, rb, g), rho(ra, rv, g)
            if np.isfinite(o_) and np.isfinite(x_): sc[v] = (o_ - x_, x_)
        if sc: w = min(sc, key=lambda v: sc[v][0]); best.append(sc[w][0]); who.append(w)
        else: best.append(np.nan); who.append(".")
    own, mg = np.array(own), np.array(best)
    return dict(own_rho=np.nanmean(own), best_other_rho=np.nan, best_other=pd.Series(who).mode()[0],
                margin=np.nanmean(mg), splits_margin_pos=int(np.nansum(mg > 0)), n_splits=int(np.isfinite(mg).sum()))

rows = []
for u in sorted(set(T.unit)):
    if u not in prof: continue
    s = specificity(u, [v for v in deep_tfs if v != u]); s["unit"] = u; rows.append(s)
S = pd.DataFrame(rows)
# ZNF503 diagnostic: all TFs as comparators
z = []
for u in sorted(set(T.unit)):
    if u in prof and cells[u] >= DEEP:
        z.append(specificity(u, [v for v in cells if v != u and v in prof])["best_other"])
znf503_best = int(sum(w == "ZNF503" for w in z))

T = T[T.rep == "raw"].copy(); T["q"] = T.groupby("arm").p.transform(bh)
T["reproducible"] = (T.obs > T.null_p90) & (T.q <= 0.10)
T = T.merge(S, on="unit", how="left"); T["deep"] = T.n_cells >= DEEP
T["specific"] = (T.margin > 0) & (T.splits_margin_pos >= np.ceil(0.9 * T.n_splits)) & (T.n_splits >= 10)
T["call"] = np.where(~T.deep, "undetermined (shallow)", np.where(T.reproducible & T.specific, "pass",
            np.where(T.reproducible & (T.margin > 0) & (T.splits_margin_pos >= 0.55 * T.n_splits), "borderline", "fail")))
T.to_csv(f"{V}/gate1_v2_1_rerun_tf.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 250)
ta = T[T.arm == "abc"].sort_values("n_cells")
print("per-TF (raw x ABC):", ta.call.value_counts().to_dict(), "| genes used median", int(ta.genes_used.median()),
      "| shallow TFs with 0 eligible genes:", int(((ta.genes_used < 50) & ~ta.deep).sum()))
print("passes:", ", ".join(f"{u}({n})" for u, n in zip(ta[ta.call == "pass"].unit, ta[ta.call == "pass"].n_cells)))
print("borderline:", ", ".join(ta[ta.call == "borderline"].unit))
old8 = ["ZNF507", "MBD3", "E4F1", "MLX", "XPA", "ZNF777", "SOX4", "EZH2", "RBBP4", "E2F3", "RBBP7", "NFYA", "MTF2"]
print(ta[ta.unit.isin(old8)][["unit", "n_cells", "genes_used", "obs", "null_p90", "q", "margin", "splits_margin_pos", "n_splits", "best_other", "call"]].round(3).to_string(index=False))
z5 = T[(T.unit == "ZNF503") & (T.arm == "abc")]
print(f"ZNF503: genes used {float(z5.genes_used.iloc[0]) if len(z5) else 'NA'}; best-other for {znf503_best} of {len(z)} deep TFs when all TFs are comparators (v2: 60/79)")
for t in ["NFYA", "CTCF", "MAZ", "NRF1", "NEUROD4"]:
    print(t, ta[ta.unit == t][["n_cells", "genes_used", "obs", "null_p90", "q", "margin", "splits_margin_pos", "call"]].round(3).to_string(index=False, header=False))

if len(M):
    M = M[M.rep == "raw"].copy(); M["loo"] = M.unit.str.contains("__minus_")
    M["q"] = M.groupby(["loo", "arm"]).p.transform(bh); M["reproducible"] = (M.obs > M.null_p90) & (M.q <= 0.10)
    mrows = []
    for u in M.unit.unique():
        if u not in prof: continue
        mem = M[M.unit == u].members.iloc[0].split(",")
        comps = [v for v in deep_tfs if v not in mem] + [m for m in full_modules if m != u.split("__minus_")[0] and m in prof]
        s1 = specificity(u, comps)
        dm = [v for v in mem if v in deep_tfs]
        s2 = specificity(u, dm) if dm else dict(margin=np.nan, splits_margin_pos=0, best_other=".")
        mrows.append(dict(unit=u, margin_vs_others=s1["margin"], pos_vs_others=s1["splits_margin_pos"], n_splits=s1["n_splits"], best_other=s1["best_other"],
                          own_rho=s1["own_rho"], margin_vs_members=s2["margin"], pos_vs_members=s2["splits_margin_pos"],
                          best_member=s2["best_other"]))
    Ms = pd.DataFrame(mrows); M = M.merge(Ms, on="unit", how="left")
    M["specific"] = (M.margin_vs_others > 0) & (M.pos_vs_others >= np.ceil(0.9 * M.n_splits)) & (M.n_splits >= 10)
    M["beyond_members"] = (M.margin_vs_members > 0) & (M.pos_vs_members >= np.ceil(0.9 * M.n_splits))
    M["call"] = np.where(M.reproducible & M.specific, np.where(M.beyond_members | M.margin_vs_members.isna(), "module node", "pass (= dominant member)"),
                np.where(M.reproducible, "reproducible, not specific", "fail"))
    M["flag"] = np.where(M.members == "DDIT3,E2F3,SAFB,XPA,ZNF703,ZNF768", "batch_module (= high-library group)", ".")
    M.to_csv(f"{V}/gate1_v2_1_modules.tsv", sep="\t", index=False, float_format="%.4f")
    print(M[M.arm == "abc"].sort_values(["loo", "unit"])[["unit", "n_cells", "genes_used", "obs", "null_p90", "q", "margin_vs_others", "pos_vs_others", "n_splits",
          "best_other", "margin_vs_members", "pos_vs_members", "best_member", "call", "flag"]].round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(2.6, 2.0))
d = ta.copy(); d["bin"] = pd.qcut(np.log10(d.n_cells), 5)
for col, lab, c in (("reproducible", "reproducible", "#8C8C8C"), ("call", "pass (v2.1)", "#1A1A1A")):
    pr = d.groupby("bin", observed=True).apply(lambda x: (x[col] == True).mean() if col == "reproducible" else (x[col] == "pass").mean())
    ax.plot([10 ** i.mid for i in pr.index], pr.values, marker="o", ms=2.5, lw=0.8, color=c, label=lab)
ax.axvline(DEEP, color="#8C8C8C", lw=0.5, ls="--")
ax.set_xscale("log"); ax.set_xlabel("cells per TF"); ax.set_ylabel("fraction of TFs"); ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(f"{V}/gate1_v2_1_pass_vs_cells.svg")
