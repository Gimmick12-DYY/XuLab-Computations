# R5 two-layer Regulome, prep (spec steps 1-2): loop anchor -> promoter tables (Peakachu 0.7, 5 kb primary, 10 kb sensitivity) and the
# replicated H3K27ac flag (GSE246314 HEK293T, peaks present in >= 2 of 3 replicates, >= 1 bp overlap) for active elements and union ATAC peaks.
# TSS table = ABC TargetGeneTSS per gene (the R2 gene universe). usage: r5_prep.py   (login node, a few minutes; writes edges/)
import os
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/regulatome/r5_two_layer"; os.makedirs(f"{O}/edges", exist_ok=True)
assert not os.path.exists(f"{O}/manifest_v7.sha256"), "manifest v7 exists; R5 outputs are frozen"
A = pd.read_csv(f"{X}/regulatome/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t", usecols=["chr", "TargetGene", "TargetGeneTSS"])
TS = A.drop_duplicates("TargetGene").rename(columns={"TargetGene": "gene", "TargetGeneTSS": "tss"}); TS["tss"] = TS.tss.astype(np.int64)
tss_by = {c: (d.tss.values[np.argsort(d.tss.values)], d.gene.values[np.argsort(d.tss.values)]) for c, d in TS.groupby("chr")}
def loop_promoters(res):
    L = pd.read_csv(f"{X}/hic/work_wt293t/loops/peakachu_{res}.loops.0.7.bedpe", sep="\t", header=None, usecols=range(6), names=["c1", "s1", "e1", "c2", "s2", "e2"]); L["loop_id"] = [f"loop{i + 1}" for i in range(len(L))]
    rows = []
    for r in L.itertuples(index=False):
        for side, (c, s, e), (pc, ps, pe) in (("L", (r.c1, r.s1, r.e1), (r.c2, r.s2, r.e2)), ("R", (r.c2, r.s2, r.e2), (r.c1, r.s1, r.e1))):
            if c not in tss_by: continue
            t, g = tss_by[c]; lo, hi = np.searchsorted(t, s - 2000, side="right"), np.searchsorted(t, e + 2000, side="left")
            for k in range(lo, hi): rows.append((r.loop_id, side, c, s, e, g[k], int(t[k]), pc, ps, pe))
    T = pd.DataFrame(rows, columns=["loop_id", "anchor_side", "chrom", "anchor_start", "anchor_end", "gene", "tss", "partner_chrom", "partner_start", "partner_end"])
    return L, T
for res, name in ((5000, "loop_anchor_promoter.tsv.gz"), (10000, "loop_anchor_promoter_10kb.tsv.gz")):
    L, T = loop_promoters(res); T.to_csv(f"{O}/edges/{name}", sep="\t", index=False)
    print(f"{res // 1000} kb loops {len(L)}: loops with >= 1 promoter anchor {T.loop_id.nunique()} ({T.loop_id.nunique() / len(L):.1%}); both anchors {T.groupby('loop_id').anchor_side.nunique().eq(2).sum()}; genes reached {T.gene.nunique()} of {len(TS)}; rows {len(T)}", flush=True)
# replicated H3K27ac: intervals covered by peaks of >= 2 of 3 replicates
R = "/work/users/d/y/dyy12/XuLab/results_3d_model/se_hub_293T_h3k27ac/raw"; reps = sorted(f for f in os.listdir(R) if "H3K27ac_CTRL_Rep" in f)
ev = []
for k, f in enumerate(reps):
    P = pd.read_csv(f"{R}/{f}", sep="\t", header=None, usecols=[0, 1, 2], names=["c", "s", "e"]).sort_values(["c", "s"])
    for c, d in P.groupby("c"):  # merge within a replicate so each replicate counts once
        s, e = d.s.values, d.e.values; m = np.r_[True, s[1:] > np.maximum.accumulate(e)[:-1]]; gid = np.cumsum(m); ms = pd.Series(s).groupby(gid).min().values; me = pd.Series(e).groupby(gid).max().values
        ev += [(c, x, 1) for x in ms] + [(c, x, -1) for x in me]
E_ = pd.DataFrame(ev, columns=["c", "x", "d"]).sort_values(["c", "x", "d"]); rep2 = []
for c, d in E_.groupby("c"):
    cov = np.cumsum(d.d.values); x = d.x.values
    for i in range(len(x) - 1):
        if cov[i] >= 2 and x[i + 1] > x[i]: rep2.append((c, x[i], x[i + 1]))
K = pd.DataFrame(rep2, columns=["c", "s", "e"]); print(f"H3K27ac replicates {[f.split('_')[0] for f in reps]}; regions in >= 2 of 3: {len(K)} ({(K.e - K.s).sum() / 1e6:.1f} Mb)", flush=True)
kb = {c: (d.s.values, d.e.values) for c, d in K.groupby("c")}
def flag(c, s, e):
    out = np.zeros(len(s), bool)
    for ch in np.unique(c):
        if ch not in kb: continue
        ks, ke = kb[ch]; m = c == ch; out[m] = (np.searchsorted(ks, e[m], side="left") - np.searchsorted(ke, s[m], side="right")) > 0
    return out
def dist_tss(c, s, e):
    mid = (s + e) // 2; d = np.full(len(s), np.inf)
    for ch in np.unique(c):
        if ch not in tss_by: continue
        t = tss_by[ch][0]; m = c == ch; i = np.searchsorted(t, mid[m]); lo = np.abs(mid[m] - t[np.clip(i - 1, 0, len(t) - 1)]); hi = np.abs(mid[m] - t[np.clip(i, 0, len(t) - 1)]); d[m] = np.minimum(lo, hi)
    return d
El = pd.read_csv(f"{X}/regulatome/m0_universe/elements_v2.bed", sep="\t", header=None, names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"]); El = El[El.cls.isin(["promoter", "distal_active", "bivalent"])].reset_index(drop=True)
At = pd.read_csv(f"{X}/data/HEK293T_ATAC_union3.bed.gz", sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"])
for nm, D in (("elements", El), ("atac", At)):
    c, s, e = D.chrom.values, D.start.values.astype(np.int64), D.end.values.astype(np.int64); f = flag(c, s, e); d = dist_tss(c, s, e); dist = d > 5000
    out = D[["chrom", "start", "end"] + (["eid", "cls"] if nm == "elements" else [])].copy(); out["h3k27ac_rep2"] = f.astype(int); out["dist_to_tss"] = d; out["distal_5kb"] = dist.astype(int)
    out.to_csv(f"{O}/edges/h3k27ac_flag_{nm}.tsv.gz", sep="\t", index=False)
    print(f"{nm}: n {len(D)}; flagged {f.mean():.1%}; distal (> 5 kb from any TSS) n {dist.sum()}, flagged {f[dist].mean():.1%}; proximal flagged {f[~dist].mean():.1%}", flush=True)
