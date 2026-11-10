# Follow-up dev-TF benchmark over any named link sets (DESCRIPTIVE dev-set tuning; development TFs only, the 24 held-out TFs are spent and not scored).
# Same scoring as ../bench_k27ac.py (grid6 unchanged). usage: bench_multi.py <outdir> name=path/to/EnhancerPredictionsFull_thrX.tsv [...]
# Per link set <name>: <name>X_* (x ABC.Score) and <name>S_* (binding-only sum over links), each with raw binding and the CTRL_n_elem / CTRL_pool / CTRL_atac controls.
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
X = "/work/users/d/y/dyy12/XuLab"; M4 = f"{X}/regulatome/m4_validation"; G1 = f"{X}/regulatome/gate1/cellmats"; O = sys.argv[1]; os.makedirs(O, exist_ok=True)
SEED = 20261109; NR = 1000; ARMS = {"arm1": "arm1_high_knockdown", "arm2": "arm2_all_cell"}; ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; panel = lambda t: ALIAS.get(t, t)
E = pd.read_csv(f"{X}/regulatome/m0_universe/elements_v2.bed", sep="\t", header=None, names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"])
act = E.cls.isin(["promoter", "distal_active", "bivalent"]).values; Ea = E[act].reset_index(drop=True)
coord = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(Ea.chrom, Ea.start, Ea.end))}
LS = {}
for nm, f in (a.split("=", 1) for a in sys.argv[2:]):
    A = pd.read_csv(f, sep="\t", usecols=["chr", "start", "end", "TargetGene", "ABC.Score"])
    A["e"] = (A.chr + ":" + A.start.astype(str) + "-" + A.end.astype(str)).map(coord); assert A.e.notna().all(), nm; A["e"] = A.e.astype(int); LS[nm] = A; print(nm, "links", len(A), flush=True)
atac = np.load(f"{X}/regulatome/m1_binding/v2/counts/atac_counts.npz")["atac"][np.flatnonzero(act)].astype(float); atac_cpm = atac / atac.sum() * 1e6
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]
Sp = pd.read_csv(f"{X}/regulatome/m6_improve/split_v1.tsv", sep="\t"); DEV = set(Sp.tf[Sp.set == "development"]); assert len(DEV) == 52 and not (DEV & set(Sp.tf[Sp.set == "heldout"]))
tfs_dev = [t for t in tfs if panel(t) in DEV]; print("dev TFs", len(tfs_dev), "panel", len({panel(t) for t in tfs_dev}), flush=True)
def cpm_of(t):
    y = np.asarray(sp.load_npz(f"{G1}/{t}.raw.npz").sum(0)).ravel(); return y / max(y.sum(), 1e-12) * 1e6
RAW = {t: cpm_of(t) for t in tfs}; TOT = np.sum([RAW[t] for t in tfs], axis=0); NT = len(tfs)
def variants(t):
    c = RAW[t]; Ev = {"raw": c, "CTRL_n_elem": np.ones_like(c), "CTRL_pool": (TOT - c) / (NT - 1), "CTRL_atac": atac_cpm}; out = {}
    for nm, A in LS.items():
        e, g, w = A.e.values, A.TargetGene.values, A["ABC.Score"].values
        for k, v in Ev.items(): out[f"{nm}X_{k}"] = pd.Series(v[e] * w).groupby(g).sum(); out[f"{nm}S_{k}"] = pd.Series(v[e]).groupby(g).sum()
    return out
SC = {}
for p in sorted({panel(t) for t in tfs_dev}):
    ts = [t for t in tfs_dev if panel(t) == p]; Vs = [variants(t) for t in ts]; L = {}
    for k in Vs[0]:
        s = pd.concat([V[k] for V in Vs], axis=1).fillna(0).mean(1) if len(Vs) > 1 else Vs[0][k]  # HP1 paralogs averaged
        s = s[s > 0].sort_values(ascending=False, kind="stable"); L[f"{k}@1000"] = list(s.index[:1000])
    SC[p] = L
print("variants", list(next(iter(SC.values())).keys()), flush=True)
pd.DataFrame([dict(tf=p, variant=v, gene=g) for p, L in SC.items() for v, gs in L.items() if v.endswith("_raw@1000") for g in gs]).to_csv(f"{O}/lists_raw.tsv.gz", sep="\t", index=False)
def load(a):  # b3b4.py load() (no replicate part), as grid6
    d = np.load(f"{M4}/kd_response/out_v2/{ARMS[a]}/kd_response.npz", allow_pickle=True); q = pd.read_csv(f"{M4}/kd_response/out_v2/{ARMS[a]}/guide_qc.tsv", sep="\t"); u = d["units"].tolist(); ui = {x: k for k, x in enumerate(u)}; uni = d["ntc_detect"] >= .1; g = d["genes"][uni]; Z = d["z"][:, uni].astype(float); kp = q[q.kept == 1]; prim = kp[kp.kind.isin(["all", "construct"])]; tf_ = sorted(prim.tf.unique())
    P_ = np.vstack([np.nanmean(Z[[ui[x] for x in prim.unit[prim.tf == t]]], axis=0) for t in tf_]); P_ = np.nan_to_num(P_); Rr = P_ - P_.mean(0); Rr = Rr - Rr.mean(1, keepdims=True)
    return dict(g=g, gi={x: k for k, x in enumerate(g)}, R=Rr, tfs=tf_, mean=d["ntc_mean"][uni].astype(float))
rows = []
for a in ARMS:
    D = load(a); rng = np.random.default_rng(SEED + (a == "arm2")); g, gi, Rr, tf_ = D["g"], D["gi"], D["R"], D["tfs"]
    pts = sorted({panel(t) for t in tf_} & DEV); Rp = np.vstack([Rr[[i for i, t in enumerate(tf_) if panel(t) == p]].mean(0) for p in pts]); Ap = np.abs(Rp).astype(np.float32); pi = {p: k for k, p in enumerate(pts)}
    bins = pd.qcut(D["mean"], 20, labels=False, duplicates="drop"); BG = [np.flatnonzero(bins == b) for b in range(bins.max() + 1)]
    print(a, "dev KDs", len(pts), flush=True)
    for p in pts:
        if p not in SC: continue
        for v, genes in SC[p].items():
            ix = np.array(sorted({gi[x] for x in genes if x in gi}), int)
            if len(ix) < 50: continue
            cnt = np.bincount(bins[ix], minlength=len(BG)); rs = np.hstack([BG[b][rng.integers(0, len(BG[b]), (NR, c))] for b, c in enumerate(cnt) if c])
            obs = Ap[:, ix].mean(1); rm = np.stack([Ap[k][rs].mean(1) for k in range(len(pts))]); z = (obs - rm.mean(1)) / rm.std(1)
            own = z[pi[p]]; off = np.delete(z, pi[p]); ab = Ap[pi[p]]; ng = len(ab); resp = np.zeros(ng, bool); resp[np.argsort(-ab, kind="stable")[:int(round(ng * .05))]] = True; inl = np.zeros(ng, bool); inl[ix] = True
            t2 = [[int((inl & resp).sum()), int((~inl & resp).sum())], [int((inl & ~resp).sum()), int((~inl & ~resp).sum())]]; lor = np.log((t2[0][0] + .5) * (t2[1][1] + .5) / ((t2[0][1] + .5) * (t2[1][0] + .5)))
            rows.append(dict(arm=a, tf=p, variant=v, n=len(ix), z_own=own, z_off_mean=off.mean(), specificity=own - off.mean(), own_rank=int((z > own).sum() + 1), n_kd=len(pts), logor5=lor))
    print(a, "done", flush=True)
P = pd.DataFrame(rows); assert not (set(P.tf) - DEV); P.to_csv(f"{O}/per_tf.tsv", sep="\t", index=False, float_format="%.4g")
