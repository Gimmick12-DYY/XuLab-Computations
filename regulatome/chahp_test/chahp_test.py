# ChAHP pooled test (Gate 1 v3 machinery). Units: H3K9me3 HEK293 replicated peaks (ENCODE ENCFF037SXA) merged within 5 kb
# ("k9peaks", built here on the v3 1 kb grid, mappable >= 0.5), plus the v3 50 kb bins and gene body +/- 10 kb.
# Pools: ADNP+HP1+CHAMP1 (chahp3) and ADNP+HP1 (chahp2), each with 20 random pools (members swapped for non-member TFs of the
# same library group within +/-4 cell-count ranks, total cells within +/-20%). Statistic and nulls as in gate1_v3.py.
#   usage: chahp_test.py units | prep <TF> | stat <unit_type> <pool> | summary
import json, os, subprocess, sys
import numpy as np, pandas as pd, scipy.sparse as sp
X = "/work/users/d/y/dyy12/XuLab"; H = f"{X}/regulatome/chahp_test"; V3 = f"{X}/regulatome/gate1/v3_binned"
K9 = f"{X}/regulatome/encode_repressive_check/data/ENCFF037SXA.bed.gz"
g = np.load(f"{V3}/units/grid.npz", allow_pickle=True); ci = {c: i for i, c in enumerate(g["chrom"])}; off = g["offset"]; NB = int(off[-1])
step = sys.argv[1]

def udir(ut): return (f"{H}/units", f"{H}/cellmats") if ut == "k9peaks" else (f"{V3}/units", f"{V3}/cellmats")

if step == "units":
    cov = np.load(f"{V3}/units/bin1k_cov.npz"); GC1, MP1 = cov["gc"], cov["map"]
    txt = subprocess.run(f"zcat {K9} | cut -f1-3 | sort -k1,1 -k2,2n | bedtools merge -d 5000 -i -", shell=True, capture_output=True, text=True, check=True).stdout
    iv = pd.read_csv(pd.io.common.StringIO(txt), sep="\t", header=None, names=["chrom", "start", "end"]); iv = iv[iv.chrom.isin(ci)].reset_index(drop=True)
    rows, cols, w = [], [], []
    for u, (c, s, e) in enumerate(zip(iv.chrom, iv.start, iv.end)):
        for b in range(s // 1000, (e - 1) // 1000 + 1):
            ov = min(e, (b + 1) * 1000) - max(s, b * 1000)
            if ov > 0: rows.append(off[ci[c]] + b); cols.append(u); w.append(ov / 1000.0)
    M = sp.csr_matrix((w, (rows, cols)), shape=(NB, len(iv))); bp = np.asarray(M.sum(0)).ravel() * 1000
    gcu = (M.T @ GC1) * 1000 / np.maximum(bp, 1); mpu = (M.T @ MP1) * 1000 / np.maximum(bp, 1); keep = (mpu >= 0.5) & (bp > 0)
    sp.save_npz(f"{H}/units/k9peaks_map.npz", M[:, keep].tocsr()); np.savez(f"{H}/units/k9peaks_cov.npz", gc=gcu[keep], map=mpu[keep], width=bp[keep])
    iv[keep].to_csv(f"{H}/units/k9peaks.bed", sep="\t", header=False, index=False)
    print(f"k9peaks: {keep.sum():,} of {len(iv):,} merged peaks kept; median width {np.median(bp[keep]):,.0f} bp")

elif step == "prep":
    tf = sys.argv[2]
    r = pd.read_csv(f"/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment/TF.{tf}.bed.gz", sep="\t", header=None, usecols=[0, 1, 3],
                    names=["chrom", "start", "bc"], dtype={"chrom": str, "start": np.int64, "bc": str})
    r = r[r.chrom.isin(ci)]; bcs, cell = np.unique(r.bc.values, return_inverse=True)
    gb = off[r.chrom.map(ci).values] + r.start.values // 1000
    C1 = sp.csr_matrix((np.ones(len(gb), np.float32), (cell, gb)), shape=(len(bcs), NB)); C1.sum_duplicates()
    sp.save_npz(f"{H}/cellmats/{tf}.k9peaks.npz", (C1 @ sp.load_npz(f"{H}/units/k9peaks_map.npz").tocsc()).tocsr().astype(np.float32))
    print(tf, len(bcs), "cells")

elif step == "stat":
    ut, pool = sys.argv[2], sys.argv[3]
    tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]
    cells = {t: json.load(open(f"{X}/regulatome/m1_binding/v2/counts/{t}.json"))["cells"] for t in tfs}
    F = pd.read_csv(f"{X}/regulatome/m1_binding/v2/frac_reads_by_class.tsv", sep="\t", index_col=0)
    group = (F.all_elements > 0.22).map({True: "high", False: "low"}); rank = pd.Series(cells).rank()
    BASE = {"chahp3": ["ADNP", "HP1", "CHAMP1"], "chahp2": ["ADNP", "HP1"]}
    def matched(template, seed):
        rg = np.random.default_rng(seed); target = sum(cells[t] for t in template); best = None
        for _ in range(500):
            pick = []
            for m in template:
                cand = [t for t in tfs if t not in BASE["chahp3"] and t not in pick and group[t] == group[m] and abs(rank[t] - rank[m]) <= 4]
                if not cand: cand = [t for t in tfs if t not in BASE["chahp3"] and t not in pick and group[t] == group[m]]
                pick.append(rg.choice(cand))
            dev = abs(sum(cells[t] for t in pick) / target - 1)
            if best is None or dev < best[0]: best = (dev, pick)
            if dev <= 0.2: break
        return best[1]
    POOLS = dict(BASE)
    for k in range(20):
        POOLS[f"rand3_{k + 1:02d}"] = matched(BASE["chahp3"], 4000 + k); POOLS[f"rand2_{k + 1:02d}"] = matched(BASE["chahp2"], 5000 + k)
    members = POOLS[pool]; Ud, Cd = udir(ut)
    from scipy.stats import rankdata
    N_SPLIT, N_NULL, N_NULL_SPLIT, EMIN = 20, 100, 5, 5.0
    cv = np.load(f"{Ud}/{ut}_cov.npz"); gc, mp, wd = cv["gc"], cv["map"], cv["width"]; nU = len(gc)
    raw = {t: sp.load_npz(f"{Cd}/{t}.{ut}.npz").tocsr() for t in tfs}
    reads = {t: np.load(f"{V3}/cellmats/{t}.reads.npy").astype(float) for t in tfs}
    tf_sum = {t: np.asarray(raw[t].sum(0)).ravel() for t in tfs}; tf_reads = {t: reads[t].sum() for t in tfs}
    tot_sum = sum(tf_sum.values()); tot_reads = sum(tf_reads.values())
    gpool = {}
    for g_ in set(group[m] for m in members):
        ds = [d for d in tfs if d not in members and group[d] == g_]
        gpool[g_] = (sp.vstack([raw[d] for d in ds]).tocsr(), np.concatenate([reads[d] for d in ds]))
    def glm_mu(y, o, Xd):
        b = np.zeros(Xd.shape[1]); b[0] = np.log(max(y.sum(), 1) / np.exp(o).sum())
        for _ in range(30):
            eta = o + Xd @ b; mu = np.exp(eta); z = eta - o + (y - mu) / mu; XtW = Xd.T * mu
            bn = np.linalg.solve(XtW @ Xd + 1e-6 * np.eye(len(b)), XtW @ z)
            if np.max(np.abs(bn - b)) < 1e-7: b = bn; break
            b = bn
        return np.exp(o + Xd @ b)
    def stats(Rm, rd, excl, splits, rng, keep=False):
        pool_ = (tot_sum - sum(tf_sum[t] for t in excl)) / (tot_reads - sum(tf_reads[t] for t in excl)) * 1e6
        Xd = np.column_stack([np.ones(nU), np.log1p(pool_), np.log(wd), gc, gc ** 2, mp])
        y = np.asarray(Rm.sum(0)).ravel(); N = rd.sum(); mu = glm_mu(y, np.full(nU, np.log(N / 1e6)), Xd)
        rs, ng, prof = [], [], []
        for _ in range(splits):
            h = rng.permutation(Rm.shape[0]) < Rm.shape[0] // 2
            oa, ob = np.asarray(Rm[h].sum(0)).ravel(), np.asarray(Rm[~h].sum(0)).ravel(); ea, eb = mu * rd[h].sum() / N, mu * rd[~h].sum() / N
            k = (ea >= EMIN) & (eb >= EMIN); ng.append(int(k.sum()))
            if k.sum() >= 50:
                a, b = np.log2((oa[k] + .5) / (ea[k] + .5)), np.log2((ob[k] + .5) / (eb[k] + .5))
                rs.append(float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]))
            if keep: prof.append(np.stack([oa, ea, ob, eb]).astype(np.float32))
        return (float(np.mean(rs)) if rs else np.nan), float(np.mean(ng)), prof
    rng = np.random.default_rng(20261030 + sum(map(ord, pool + ut)) % 10000)
    Rm = sp.vstack([raw[m] for m in members]).tocsr(); rd = np.concatenate([reads[m] for m in members])
    obs, ng, prof = stats(Rm, rd, members, N_SPLIT, rng, keep=True)
    nd = []
    if np.isfinite(obs):
        for _ in range(N_NULL):
            parts = []
            for m in members:
                PR, PD = gpool[group[m]]; c = rng.choice(PR.shape[0], size=cells[m], replace=cells[m] > PR.shape[0]); parts.append((PR[c], PD[c]))
            s_, _, _ = stats(sp.vstack([p[0] for p in parts]).tocsr(), np.concatenate([p[1] for p in parts]), members, N_NULL_SPLIT, rng)
            if np.isfinite(s_): nd.append(s_)
    nd = np.array(nd)
    row = dict(unit_type=ut, pool=pool, members=",".join(members), n_cells=sum(cells[m] for m in members), eligible_units=ng, obs=obs,
               null_mean=nd.mean() if len(nd) else np.nan, null_p90=np.percentile(nd, 90) if len(nd) else np.nan,
               p=(1 + (nd >= obs).sum()) / (1 + len(nd)) if len(nd) else np.nan)
    pd.DataFrame([row]).to_csv(f"{H}/results/{ut}.{pool}.tsv", sep="\t", index=False, float_format="%.4f")
    np.savez_compressed(f"{H}/results/{ut}.{pool}.profiles.npz", OE=np.stack(prof))
    print(row)

elif step == "summary":
    EMIN = 5.0
    def bh(p):
        p = np.asarray(p, float); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]; r = np.empty_like(q); r[o] = np.minimum(q, 1); return r
    def resid(OE, k, h):
        o, e = (OE[k, 0], OE[k, 1]) if h == "A" else (OE[k, 2], OE[k, 3]); return np.log2((o + .5) / (e + .5)), e >= EMIN
    def pear(x, y, m):
        if m.sum() < 50: return np.nan
        a, b = x[m] - x[m].mean(), y[m] - y[m].mean(); d = np.sqrt((a * a).sum() * (b * b).sum()); return float((a * b).sum() / d) if d > 0 else np.nan
    def margin(P, u, comps):
        ms = []
        for k in range(P[u].shape[0]):
            rA, eA = resid(P[u], k, "A"); rB, eB = resid(P[u], k, "B"); base = eA & eB; best = None
            for v in comps:
                rv, ev = resid(P[v], k, "B"); g = base & ev; o_, x_ = pear(rA, rB, g), pear(rA, rv, g)
                if np.isfinite(o_) and np.isfinite(x_): best = o_ - x_ if best is None else min(best, o_ - x_)
            if best is not None: ms.append(best)
        return (float(np.mean(ms)) if ms else np.nan), int(np.sum(np.array(ms) > 0)), len(ms)
    rows = []
    for ut in ("k9peaks", "bin50k", "genebody"):
        for real, pref in (("chahp3", "rand3_"), ("chahp2", "rand2_")):
            units = [real] + [f"{pref}{k:02d}" for k in range(1, 21)]
            R = pd.concat([pd.read_csv(f"{H}/results/{ut}.{u}.tsv", sep="\t") for u in units if os.path.exists(f"{H}/results/{ut}.{u}.tsv")])
            R["q"] = bh(R.p.fillna(1).values); R["reproducible"] = (R.obs > R.null_p90) & (R.q <= 0.10); R["excess"] = R.obs - R.null_mean
            P = {u: np.load(f"{H}/results/{ut}.{u}.profiles.npz")["OE"] for u in R.pool}
            for r in R.itertuples():
                dm = margin(P, r.pool, [u for u in R.pool if u != r.pool])
                rows.append(dict(unit_type=ut, family=real, pool=r.pool, members=r.members, n_cells=r.n_cells, eligible_units=r.eligible_units,
                                 obs=r.obs, null_p90=r.null_p90, excess=r.excess, q=r.q, reproducible=r.reproducible,
                                 dm_margin=dm[0], dm_pos=f"{dm[1]}/{dm[2]}", dm_specific=bool(dm[0] > 0 and dm[1] >= np.ceil(0.9 * max(dm[2], 1)) and dm[2] >= 10)))
    D = pd.DataFrame(rows); D.to_csv(f"{H}/chahp_pooled_test.tsv", sep="\t", index=False, float_format="%.4f")
    for (ut, fam), d in D.groupby(["unit_type", "family"], sort=False):
        real = d[d.pool == fam].iloc[0]; rnd = d[d.pool != fam]
        print(f"{ut:9s} {fam}: units {real.eligible_units:,.0f}, obs {real.obs:.3f} vs null p90 {real.null_p90:.3f} (q {real.q:.3f}); excess {real.excess:+.3f} "
              f"rank {int((rnd.excess >= real.excess).sum()) + 1}/21 (random median {rnd.excess.median():+.3f}); random reproducible "
              f"{int(rnd.reproducible.sum())}/20; depth-matched margin {real.dm_margin:+.3f} ({real.dm_pos}); random dm-specific {int(rnd.dm_specific.sum())}/20")
    E = pd.read_csv(f"{X}/regulatome/encode_repressive_check/repressive_enrichment.tsv", sep="\t", index_col=0)
    C = {t: pd.read_csv(f"{X}/regulatome/encode_repressive_check/counts/{t}.tsv.gz", sep="\t") for t in E.index}
    tot = {k: sum(C[t][k].sum() for t in C) for k in ("total", "H3K9me3_HEK293__set")}
    for name, mem in (("ADNP+HP1+CHAMP1", ["ADNP", "HP1", "CHAMP1"]), ("ADNP+HP1", ["ADNP", "HP1"])):
        a = sum(C[t]["H3K9me3_HEK293__set"].sum() for t in mem); n = sum(C[t].total.sum() for t in mem)
        print(f"{name}: H3K9me3 fraction {a / n:.4f}; obs/exp vs leave-pool-out panel {(a / n) / ((tot['H3K9me3_HEK293__set'] - a) / (tot['total'] - n)):.3f}")
