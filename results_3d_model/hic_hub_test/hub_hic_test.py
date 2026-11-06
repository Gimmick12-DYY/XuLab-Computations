"""Direct test of the chr16p11.2 promoter hub in the WT HEK293T Hi-C matrix (GSE216399, 10 kb, ICE), no polymer model.

O/E = ICE / mean ICE at the same genomic distance (kept bins only). Hub = the 23 browser hub elements (20 distinct 10 kb bins
after merging elements in the same bin), and the 24 hub-component shared clique edges (18 "red" on a Peakachu 0.7 loop, 6 "blue").

Tests (one-sided empirical p, 10,000 draws, seed 1):
 1. Edges: mean O/E of the red / blue / all hub edges vs (a) random bin pairs at the same distance and (b) bin pairs at the same
    distance whose two bins fall in the same TF-count class as the edge's ends (class from the max TF count of the 2 kb beads in
    the 10 kb bin: 0, 1-4, 5-9, >=10). Red edges are partly circular (they are defined by Hi-C loops); blue edges are not.
 2. Hub set ("is it one hub?"): mean O/E over ALL pairs of hub bins (not only edges; >= 20 kb apart) vs random sets of the same
    size drawn bin-by-bin from the same TF-count class (all kept bins of that class in the window). Also with the 24 edge pairs
    removed (non-edge pairs only), so the Cicero edges themselves cannot drive the result.
 3. Pileup: mean O/E in the 5 x 5 pixel neighbourhood around the blue edges and the non-edge hub pairs (local enrichment, APA-style).
"""
from pathlib import Path
import numpy as np, pandas as pd, itertools, json
R = Path("/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model"); O = R / "hic_hub_test"
H = np.load(R / "tf_polymer_hic/inputs/hic_10kb.npz"); ice, keep = H["ice"].astype(float), H["keep"].astype(bool); n = ice.shape[0]
B = pd.read_csv(R / "tf_polymer/inputs/beads.tsv", sep="\t")
EL = pd.read_csv(R / "tf_polymer_hicfirst/hicfirst/figure_hub_flow/hub_elements.tsv", sep="\t")
HE = pd.read_csv(R / "tf_polymer_hicfirst/hicfirst/figure_hub_flow/hub_edges_beads.tsv", sep="\t")
rng = np.random.default_rng(1); NR = 10000
# O/E
OE = np.full((n, n), np.nan)
for d in range(n):
    i = np.arange(n - d); j = i + d; ok = keep[i] & keep[j] & np.isfinite(ice[i, j])
    if ok.any():
        mu = ice[i[ok], j[ok]].mean()
        OE[i[ok], j[ok]] = OE[j[ok], i[ok]] = ice[i[ok], j[ok]] / mu
tfbin = B.groupby(B.bead // 5).n_tfs.max().reindex(range(n)).fillna(0).values
cls = np.select([tfbin >= 10, tfbin >= 5, tfbin >= 1], [3, 2, 1], 0)
EL["bin"] = EL.bead // 5; HE["bin1"], HE["bin2"] = np.sort(np.c_[HE.b1 // 5, HE.b2 // 5], 1).T
hub = np.array(sorted(set(EL.bin) & set(np.where(keep)[0])))
res = {}
def edge_test(sub, tag):
    pr = [(a, b) for a, b in zip(sub.bin1, sub.bin2) if b - a >= 2 and keep[a] and keep[b]]
    obs = np.nanmean([OE[a, b] for a, b in pr])
    nd, nc = [], []
    for _ in range(NR):
        v1, v2 = [], []
        for a, b in pr:
            d = b - a; i = rng.integers(0, n - d); v1.append(OE[i, i + d])
            cand = [i for i in range(n - d) if {cls[i], cls[i + d]} == {cls[a], cls[b]} and np.isfinite(OE[i, i + d])] if False else None
        nd.append(np.nanmean(v1))
    # class-matched candidates (precomputed per edge)
    cands = []
    for a, b in pr:
        d = b - a; ii = np.arange(n - d); m = (((cls[ii] == cls[a]) & (cls[ii + d] == cls[b])) | ((cls[ii] == cls[b]) & (cls[ii + d] == cls[a]))) & np.isfinite(OE[ii, ii + d])
        m &= ~((ii == a) & (ii + d == b)); cands.append(ii[m] if m.any() else ii[np.isfinite(OE[ii, ii + d])])
    for _ in range(NR):
        nc.append(np.nanmean([OE[c_, c_ + (b - a)] for (a, b), c in zip(pr, cands) for c_ in [rng.choice(c)]]))
    nd, nc = np.array(nd), np.array(nc)
    res[tag] = dict(n=len(pr), mean_oe=obs, null_dist=float(np.nanmean(nd)), ratio_dist=obs / np.nanmean(nd), p_dist=float((1 + (nd >= obs).sum()) / (NR + 1)),
                    null_tfclass=float(np.nanmean(nc)), ratio_tfclass=obs / np.nanmean(nc), p_tfclass=float((1 + (nc >= obs).sum()) / (NR + 1)))
    pd.DataFrame(dict(bin1=[a for a, b in pr], bin2=[b for a, b in pr], oe=[OE[a, b] for a, b in pr])).to_csv(O / f"edges_{tag}.tsv", sep="\t", index=False)
edge_test(HE[HE["on_loop_0.7"] == 1], "red_edges"); edge_test(HE[HE["on_loop_0.7"] == 0], "blue_edges"); edge_test(HE, "all_edges")
# hub set test
edge_set = set(zip(HE.bin1, HE.bin2))
def setmean(bins, drop_edges=False):
    v = [OE[a, b] for a, b in itertools.combinations(sorted(bins), 2) if b - a >= 2 and not (drop_edges and (a, b) in edge_set)]
    return np.nanmean(v)
pool = {c: np.where(keep & (cls == c))[0] for c in range(4)}
for drop, tag in ((False, "hub_all_pairs"), (True, "hub_nonedge_pairs")):
    obs = setmean(hub, drop); null = []
    for _ in range(NR):
        s = set()
        for b in hub:
            c = rng.choice(pool[cls[b]])
            while c in s: c = rng.choice(pool[cls[b]])
            s.add(c)
        null.append(setmean(s))
    null = np.array(null)
    res[tag] = dict(n_bins=len(hub), mean_oe=obs, null_tfclass=float(null.mean()), ratio_tfclass=obs / null.mean(),
                    p_tfclass=float((1 + (null >= obs).sum()) / (NR + 1)), null_q95=float(np.quantile(null, 0.95)))
    np.save(O / f"null_{tag}.npy", null)
# pileup 5x5 around blue edges and non-edge hub pairs
def pile(pairs):
    acc = []
    for a, b in pairs:
        if a - 2 < 0 or b + 3 > n: continue
        w = OE[a - 2:a + 3, b - 2:b + 3]
        if np.isfinite(w).sum() > 15: acc.append(w)
    return np.nanmean(acc, 0), len(acc)
blue = [(a, b) for a, b in zip(HE[HE["on_loop_0.7"] == 0].bin1, HE[HE["on_loop_0.7"] == 0].bin2) if b - a >= 4]
nonedge = [(a, b) for a, b in itertools.combinations(hub, 2) if b - a >= 4 and (a, b) not in edge_set]
for nm, prs in (("blue_edges", blue), ("hub_nonedge_pairs", nonedge)):
    m, k = pile(prs); np.save(O / f"pileup_{nm}.npy", m)
    res[f"pileup_{nm}"] = dict(n=k, centre=float(m[2, 2]), corners=float(np.nanmean([m[0, 0], m[0, 4], m[4, 0], m[4, 4]])))
json.dump(res, open(O / "results.json", "w"), indent=1, default=float)
pd.set_option("display.width", 200); print(pd.DataFrame(res).T.round(3).to_string())
print("hub bins:", len(hub), "TF classes of hub bins:", np.bincount(cls[hub], minlength=4).tolist(), "pool sizes:", {c: len(v) for c, v in pool.items()})
