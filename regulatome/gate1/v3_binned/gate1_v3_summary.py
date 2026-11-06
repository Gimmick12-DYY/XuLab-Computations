# Gate 1 v3 (binned) summary, per unit type (bin10k, bin50k, reprpc, genebody):
#  reproducible = obs > pooled-panel null p90 and BH q <= 0.10 (TFs and modules separately);
#  specificity margin per split = r(own A, own B) - max over comparators r(own A, other B), Pearson on the residuals of units
#    eligible in both (expected >= 5 in both halves of the unit and in the comparator's half B); specific = margin > 0 in >= 90%
#    of splits (>= 10 splits). TFs: comparators = all other TFs. Modules: non-member TFs + other modules with no shared members.
#  pass = reproducible & specific; undetermined = < 50 eligible units, or < 3,400 cells and not passing.
#  collapse check: median pairwise Pearson of deep TFs' full-data residual profiles (common eligible units) per unit type.
import glob, json, os, itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
X = "/work/users/d/y/dyy12/XuLab"; V3 = f"{X}/regulatome/gate1/v3_binned"; EMIN, DEEP = 5.0, 3400
UTS = ["bin10k", "bin50k", "reprpc", "genebody"]
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.5, "svg.fonttype": "none", "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "legend.fontsize": 6, "xtick.major.width": 0.5, "ytick.major.width": 0.5})
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]
cells = {t: json.load(open(f"{X}/regulatome/m1_binding/v2/counts/{t}.json"))["cells"] for t in tfs}
def bh(p):
    p = np.asarray(p, float); ok = np.isfinite(p); out = np.full(len(p), np.nan); q_ = p[ok]
    o = np.argsort(q_); q = q_[o] * len(q_) / np.arange(1, len(q_) + 1); q = np.minimum.accumulate(q[::-1])[::-1]
    r_ = np.empty_like(q); r_[o] = np.minimum(q, 1); out[ok] = r_; return out
def load(d):
    fs = [f for f in glob.glob(f"{V3}/{d}/*.tsv") if os.path.getsize(f) > 2]
    return pd.concat([pd.read_csv(f, sep="\t") for f in fs]) if fs else pd.DataFrame()
T, M = load("results"), load("results_modules")

def resid(OE, k, h):
    o, e = (OE[k, 0], OE[k, 1]) if h == "A" else (OE[k, 2], OE[k, 3])
    return np.log2((o + 0.5) / (e + 0.5)), e >= EMIN
def pear(x, y, m):
    if m.sum() < 50: return np.nan
    a, b = x[m] - x[m].mean(), y[m] - y[m].mean(); d = np.sqrt((a * a).sum() * (b * b).sum()); return float((a * b).sum() / d) if d > 0 else np.nan

allrows, collapse = [], []
for ut in UTS:
    P = {}
    for f in glob.glob(f"{V3}/results/{ut}.*.profiles.npz") + glob.glob(f"{V3}/results_modules/{ut}.*.profiles.npz"):
        P[f.split("/")[-1][len(ut) + 1:-len(".profiles.npz")]] = np.load(f)["OE"]
    if not P: continue
    R = {u: [resid(P[u], k, "A") + resid(P[u], k, "B") for k in range(P[u].shape[0])] for u in P}  # (rA, eA, rB, eB)
    mems = {}
    if len(M): mems = {r.unit: set(r.members.split(",")) for r in M[M.unit_type == ut].itertuples()}
    def margin(u, comps):
        ms = []
        for k in range(len(R[u])):
            rA, eA, rB, eB = R[u][k]; base = eA & eB; best = None
            for v in comps:
                if v == u or k >= len(R[v]): continue
                rv, ev = R[v][k][2], R[v][k][3]; g = base & ev
                o_, x_ = pear(rA, rB, g), pear(rA, rv, g)
                if np.isfinite(o_) and np.isfinite(x_): best = o_ - x_ if best is None else min(best, o_ - x_)
            if best is not None: ms.append(best)
        return (float(np.mean(ms)) if ms else np.nan), int(np.sum(np.array(ms) > 0)), len(ms)
    tf_units = [t for t in tfs if t in P]
    if len(T):
        t = T[T.unit_type == ut].copy(); t["q"] = bh(t.p.values); t["reproducible"] = (t.obs > t.null_p90) & (t.q <= 0.10)
        mg = {u: margin(u, tf_units) for u in t.unit}
        t["margin"] = [mg[u][0] for u in t.unit]; t["splits_pos"] = [mg[u][1] for u in t.unit]; t["n_splits"] = [mg[u][2] for u in t.unit]
        t["specific"] = (t.margin > 0) & (t.splits_pos >= np.ceil(0.9 * t.n_splits)) & (t.n_splits >= 10)
        t["call"] = np.where(t.eligible_units < 50, "undetermined (few units)",
                    np.where(t.reproducible & t.specific, "pass",
                    np.where(t.n_cells < DEEP, "undetermined (shallow)",
                    np.where(t.reproducible, "reproducible, not specific", "fail"))))
        allrows.append(t)
    if len(M):
        m = M[M.unit_type == ut].copy(); m["q"] = bh(m.p.values); m["reproducible"] = (m.obs > m.null_p90) & (m.q <= 0.10)
        res = []
        for u in m.unit:
            comps = [x for x in tf_units if x not in mems[u]] + [v for v in mems if v != u and not (mems[v] & mems[u]) and v in P
                                                                and not v.startswith("randshallow")]
            res.append(margin(u, comps) if u in P else (np.nan, 0, 0))
        m["margin"] = [r[0] for r in res]; m["splits_pos"] = [r[1] for r in res]; m["n_splits"] = [r[2] for r in res]
        m["specific"] = (m.margin > 0) & (m.splits_pos >= np.ceil(0.9 * m.n_splits)) & (m.n_splits >= 10)
        m["call"] = np.where(m.eligible_units < 50, "undetermined (few units)", np.where(m.reproducible & m.specific, "pass",
                    np.where(m.reproducible, "reproducible, not specific", "fail")))
        # direct: PRC2-adjacent minus EZH2/MTF2 vs PRC2 core and NuRD core (no shared members)
        for u, v in [("prc2adj_noEZH2MTF2", "PRC2core"), ("prc2adj_noEZH2MTF2", "NuRDcore"), ("PRC2core", "NuRDcore"), ("NuRDcore", "PRC2core")]:
            if u in P and v in P:
                mg_ = margin(u, [v]); m.loc[m.unit == u, f"vs_{v}"] = f"{mg_[0]:+.3f} ({mg_[1]}/{mg_[2]})"
        allrows.append(m)
    # collapse check: deep TFs' full-data residual profiles
    deep = [u for u in tf_units if cells[u] >= DEEP]
    prof = {}
    for u in deep:
        O = P[u][:, 0] + P[u][:, 2]; E = P[u][:, 1] + P[u][:, 3]; o, e = O.mean(0), E.mean(0)
        prof[u] = (np.log2((o + .5) / (e + .5)), e >= 2 * EMIN)
    cs = [pear(prof[a][0], prof[b][0], prof[a][1] & prof[b][1]) for a, b in itertools.combinations(deep, 2)]
    cs = np.array([c for c in cs if np.isfinite(c)])
    collapse.append(dict(unit_type=ut, deep_tfs=len(deep), median_pairwise_r=np.median(cs) if len(cs) else np.nan,
                         frac_pairs_r_gt_0_5=(cs > 0.5).mean() if len(cs) else np.nan))
A = pd.concat(allrows, ignore_index=True) if allrows else pd.DataFrame()
A.to_csv(f"{V3}/gate1_v3_all.tsv", sep="\t", index=False, float_format="%.4f")
C = pd.DataFrame(collapse); C.to_csv(f"{V3}/gate1_v3_collapse_check.tsv", sep="\t", index=False, float_format="%.3f")
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 60)
TT = A[A["mode"] == "tf"]
print(TT.groupby("unit_type").call.value_counts().unstack(fill_value=0).to_string())
KEY = ["EZH2", "MBD3", "SETDB1", "HP1", "PCGF6", "ZNF143", "MAZ", "RBBP4", "NFYA", "SOX4", "ZNF507", "MLX", "E4F1", "ZNF777", "NEUROD4", "CTCF", "MTF2"]
print(TT[TT.unit.isin(KEY)].pivot(index="unit", columns="unit_type", values="call").reindex(KEY).to_string())
print(TT[TT.unit.isin(KEY)].pivot(index="unit", columns="unit_type", values="eligible_units").reindex(KEY).round(0).to_string())
print("NEUROD4:\n", TT[TT.unit == "NEUROD4"][["unit_type", "eligible_units", "obs", "null_p90", "q", "margin", "splits_pos", "n_splits", "call"]].round(3).to_string(index=False))
MM = A[A["mode"] == "module"]
if len(MM):
    cols = ["unit_type", "unit", "n_cells", "eligible_units", "obs", "null_p90", "q", "margin", "splits_pos", "n_splits", "call"] + \
           [c for c in MM.columns if c.startswith("vs_")]
    print(MM[~MM.unit.str.startswith("randshallow")][cols].round(3).to_string(index=False))
    rs = MM[MM.unit.str.startswith("randshallow")]
    for ut in UTS:
        r_ = rs[rs.unit_type == ut]; ld = MM[(MM.unit_type == ut) & (MM.unit == "lowdepth")]
        if len(r_) and len(ld):
            ex = (r_.obs - r_.null_mean).values; e0 = float((ld.obs - ld.null_mean).iloc[0])
            print(f"{ut}: lowdepth excess {e0:+.3f} vs 20 random shallow sets {np.median(ex):+.3f} (max {ex.max():+.3f}); "
                  f"random sets reproducible {int(r_.reproducible.sum())}/20; lowdepth rank {int((ex >= e0).sum()) + 1}/21")
print(C.round(3).to_string(index=False))
# diagnostics: read distribution (active elements vs ReprPC vs other), deep TFs
D = pd.DataFrame([json.load(open(f)) for f in glob.glob(f"{V3}/cellmats/*.diag.json")]).set_index("tf")
D.to_csv(f"{V3}/read_distribution.tsv", sep="\t", float_format="%.4f")
dd = D[D.cells >= DEEP]
print("EZH2:", D.loc["EZH2"].round(4).to_dict())
print("deep TFs median:", dd[["reads_per_cell", "reads_per_covered_bin", "frac_active_elements", "frac_reprpc_domains", "frac_other"]].median().round(4).to_dict())
print("EZH2 rank among deep TFs (frac_reprpc_domains, descending):", int(dd.frac_reprpc_domains.rank(ascending=False)["EZH2"]), "of", len(dd))
# pass rate vs cells per unit type
fig, ax = plt.subplots(figsize=(2.8, 2.1))
for ut, col in zip(UTS, ["#1A1A1A", "#8C8C8C", "#1F3F8F", "#D98C1F"]):
    d = TT[TT.unit_type == ut].copy()
    if not len(d): continue
    d["bin"] = pd.qcut(np.log10(d.n_cells), 5)
    pr = d.groupby("bin", observed=True).call.apply(lambda x: (x == "pass").mean())
    ax.plot([10 ** i.mid for i in pr.index], pr.values, marker="o", ms=2.5, lw=0.8, color=col, label=ut)
ax.set_xscale("log"); ax.set_xlabel("cells per TF"); ax.set_ylabel("pass rate"); ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(f"{V3}/gate1_v3_pass_vs_cells.svg")
