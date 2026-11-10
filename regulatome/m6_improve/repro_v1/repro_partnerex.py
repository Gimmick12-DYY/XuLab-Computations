# repro_v1 Amendment 2: (a) Run 1 post hoc partner-excluded J_between from the saved lists; (b) Run 2 known-partner table (run2_partners.tsv).
# Writes depth_curve_partnerex.tsv, depth_slopes_partnerex.tsv, run2_partners.tsv. usage: repro_partnerex.py
import os
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; R = f"{X}/regulatome"; O = f"{R}/m6_improve/repro_v1"; L1 = f"{O}/run1/lists"
DEEP = ["RBBP4", "NFYA", "SOX4", "XPA", "RBBP7"]; DEPTHS = [1000, 2000, 3400, 5000, 10000, 20000, 50000, 100000]
R3 = pd.read_csv(f"{R}/r1_tftf/v2_final_v3/r1_pairs_frozen_v3.tsv", sep="\t", keep_default_na=False)
P = R3[(R3["class"] == "complex") | (R3.string_physical != "")][["a", "b", "class", "string_physical"]]; P.to_csv(f"{O}/run2_partners.tsv", sep="\t", index=False)
rl = lambda p: set(open(p).read().split()); jac = lambda a, b: len(a & b) / max(len(a | b), 1); rows = []
PX = {"RBBP4": ["NFYA", "SOX4", "XPA"], "RBBP7": ["NFYA", "SOX4", "XPA"]}
for N in DEPTHS:
    for d in range(5):
        L = {(t, h): rl(f"{L1}/{t}_N{N}_d{d}_h{h}.txt") for t in DEEP for h in (1, 2) if os.path.exists(f"{L1}/{t}_N{N}_d{d}_h{h}.txt")}; have = sorted({t for t, _ in L})
        jb = lambda t, us: np.mean([jac(L[(t, h)], L[(u, h)]) for u in us for h in (1, 2)]) if us else np.nan
        for t in have:
            allp = [u for u in have if u != t]; px = [u for u in PX.get(t, allp) if u in have]; norb = [u for u in allp if u not in ("RBBP4", "RBBP7")]
            jw = jac(L[(t, 1)], L[(t, 2)]); r = dict(DESCRIPTIVE="DESCRIPTIVE", tf=t, cells=N, draw=d, J_within=jw, J_between_matched=jb(t, allp), J_between_partnerex=jb(t, px), J_between_noRBBP=jb(t, norb), J_between_XPA=jb(t, ["XPA"] if t != "XPA" and "XPA" in have else []))
            for k in ("matched", "partnerex", "noRBBP", "XPA"): r[f"diff_{k}"] = jw - r[f"J_between_{k}"]
            rows.append(r)
D = pd.DataFrame(rows); D.to_csv(f"{O}/depth_curve_partnerex.tsv", sep="\t", index=False, float_format="%.5f")
rng = np.random.default_rng(20261110); S = []
for t, G in D.groupby("tf"):
    G = G.dropna(subset=["diff_partnerex"]); x = np.log10(G.cells.values); groups = [np.flatnonzero(G.cells.values == N) for N in sorted(G.cells.unique())]; r = dict(DESCRIPTIVE="DESCRIPTIVE", tf=t, n_depths=len(groups))
    for k in ("matched", "partnerex", "noRBBP", "XPA"):
        y = G[f"diff_{k}"].values
        if np.isnan(y).any(): continue
        bs = [np.polyfit(x[ix], y[ix], 1)[0] for ix in (np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(1000))]
        r.update({f"slope_diff_{k}": np.polyfit(x, y, 1)[0], f"slope_diff_{k}_lo": np.percentile(bs, 2.5), f"slope_diff_{k}_hi": np.percentile(bs, 97.5)})
    if "slope_diff_partnerex" in r and "slope_diff_matched" in r:
        q = r["slope_diff_partnerex"] / r["slope_diff_matched"] if abs(r["slope_diff_matched"]) > 1e-9 else np.nan; r["ratio_partnerex_to_matched"] = q
        r["reading"] = "more TF-specific signal (>=2x)" if q >= 2 else ("generic convergence (within +-25%)" if abs(q - 1) <= .25 else "intermediate")
    S.append(r)
pd.DataFrame(S).to_csv(f"{O}/depth_slopes_partnerex.tsv", sep="\t", index=False, float_format="%.5f")
