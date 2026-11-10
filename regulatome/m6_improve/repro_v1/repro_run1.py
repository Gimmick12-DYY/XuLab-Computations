# repro_v1 Run 1 (README.md + Amendment 1). Task t in 0..4: one deep TF, all depths x 5 draws x 2 halves, lists written out. Task 5: full-depth
# half lists for all 79 TFs (r5_core-style split). Task "summary": J_within, J_between (matched-depth primary; full-depth secondary), slopes, gate.
# usage: repro_run1.py <task|summary>
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
X = "/work/users/d/y/dyy12/XuLab"; R = f"{X}/regulatome"; G1 = f"{R}/gate1/cellmats"; O = f"{R}/m6_improve/repro_v1/run1"
DEEP = ["RBBP4", "NFYA", "SOX4", "XPA", "RBBP7"]; DEPTHS = [1000, 2000, 3400, 5000, 10000, 20000, 50000, 100000]; ND = 5
tfs = [l.strip() for l in open(f"{R}/m1_binding/v2/tfs.txt") if l.strip()]
def links():
    E_ = pd.read_csv(f"{R}/m0_universe/elements_v2.bed", sep="\t", header=None, names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"]); Ea = E_[E_.cls.isin(["promoter", "distal_active", "bivalent"])].reset_index(drop=True)
    A = pd.read_csv(f"{R}/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene"])
    coord = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(Ea.chrom, Ea.start, Ea.end))}; e = (A.chr + ":" + A.start.astype(str) + "-" + A.end.astype(str)).map(coord).astype(int).values
    gl = pd.Index(sorted(A.TargetGene.unique())); return e, gl.get_indexer(A.TargetGene.values), gl
def toplist(y, reads, e, gidx, gl, n=1000):
    cpm = y / max(reads, 1e-12) * 1e6; s = np.bincount(gidx, cpm[e], len(gl)); o = np.argsort(-s, kind="stable"); o = o[s[o] > 0][:n]; return list(gl[o])
def write(p, L): open(p, "w").write("\n".join(L) + "\n")
task = sys.argv[1]
if task != "summary":
    e, gidx, gl = links(); k = int(task)
    if k < 5:
        t = DEEP[k]; M = sp.load_npz(f"{G1}/{t}.raw.npz").tocsr(); rd = np.load(f"{G1}/{t}.cells.npz")["reads"].astype(float); n = M.shape[0]; os.makedirs(f"{O}/lists", exist_ok=True)
        for N in [d for d in DEPTHS if d <= n]:
            for d in range(ND):
                rng = np.random.default_rng(20261110 + 1000 * k + d + 7 * N); sub = rng.choice(n, N, replace=False); h = rng.permutation(N) < N // 2
                for hi, m in ((1, sub[h]), (2, sub[~h])): write(f"{O}/lists/{t}_N{N}_d{d}_h{hi}.txt", toplist(np.asarray(M[m].sum(0)).ravel(), rd[m].sum(), e, gidx, gl))
            print(t, N, flush=True)
    else:
        os.makedirs(f"{O}/lists_full", exist_ok=True)
        for i, t in enumerate(tfs):
            M = sp.load_npz(f"{G1}/{t}.raw.npz").tocsr(); rd = np.load(f"{G1}/{t}.cells.npz")["reads"].astype(float); hh = np.random.default_rng(20261029 + i).permutation(M.shape[0]) < M.shape[0] // 2
            for hi, m in ((1, hh), (2, ~hh)): write(f"{O}/lists_full/{t}_h{hi}.txt", toplist(np.asarray(M[m].sum(0)).ravel(), rd[m].sum(), e, gidx, gl))
        print("full done", flush=True)
else:
    rl = lambda p: set(open(p).read().split()); jac = lambda a, b: len(a & b) / max(len(a | b), 1)
    FULL = {(t, h): rl(f"{O}/lists_full/{t}_h{h}.txt") for t in tfs for h in (1, 2)}; rows = []
    for N in DEPTHS:
        for d in range(ND):
            L = {(t, h): rl(f"{O}/lists/{t}_N{N}_d{d}_h{h}.txt") for t in DEEP for h in (1, 2) if os.path.exists(f"{O}/lists/{t}_N{N}_d{d}_h{h}.txt")}
            have = sorted({t for t, _ in L})
            for t in have:
                jw = jac(L[(t, 1)], L[(t, 2)]); part = [u for u in have if u != t]
                jbm = np.mean([jac(L[(t, h)], L[(u, h)]) for u in part for h in (1, 2)]) if part else np.nan
                jbf = np.mean([jac(L[(t, h)], FULL[(u, h)]) for u in tfs if u not in DEEP for h in (1, 2)])
                rows.append(dict(DESCRIPTIVE="DESCRIPTIVE", tf=t, cells=N, draw=d, J_within=jw, J_between_matched=jbm, n_matched_partners=len(part), J_between_full=jbf, diff_matched=jw - jbm, diff_full=jw - jbf))
    D = pd.DataFrame(rows); D.to_csv(f"{O}/../depth_curve.tsv", sep="\t", index=False, float_format="%.5f")
    rng = np.random.default_rng(20261110); S = []
    for t, G in D.groupby("tf"):
        x = np.log10(G.cells.values); y = G.J_within.values; b = np.polyfit(x, y, 1); bs = []
        groups = [np.flatnonzero(G.cells.values == N) for N in sorted(G.cells.unique())]
        for _ in range(1000):
            ix = np.concatenate([rng.choice(gg, len(gg)) for gg in groups]); bs.append(np.polyfit(x[ix], y[ix], 1)[0])
        bm = np.polyfit(x, G.J_between_matched.values, 1)[0] if G.J_between_matched.notna().all() else np.nan
        S.append(dict(DESCRIPTIVE="DESCRIPTIVE", tf=t, n_depths=G.cells.nunique(), slope=b[0], slope_lo=np.percentile(bs, 2.5), slope_hi=np.percentile(bs, 97.5), intercept=b[1], ci_includes_0206=bool(np.percentile(bs, 2.5) <= .206 <= np.percentile(bs, 97.5)), slope_J_between_matched=bm))
    S = pd.DataFrame(S); S.to_csv(f"{O}/../depth_slopes.tsv", sep="\t", index=False, float_format="%.5f")
    med = S.slope.median(); medrow = S.iloc[(S.slope - med).abs().argsort().iloc[0]]
    accept = bool(medrow.ci_includes_0206) or bool((S.slope >= .10).all()); flat = bool((S.slope_lo <= 0).any() and (S.slope < .10).all())
    open(f"{O}/../GATE.txt", "w").write(f"median slope {med:.4f} (TF {medrow.tf}, CI [{medrow.slope_lo:.4f}, {medrow.slope_hi:.4f}]); slopes {dict(zip(S.tf, S.slope.round(4)))}; intercept spread {S.intercept.max() - S.intercept.min():.4f}\nACCEPT_YARDSTICK={accept} FLAT={flat}\n")
