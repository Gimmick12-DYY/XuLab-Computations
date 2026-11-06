# Are panel TF libraries enriched at known repressive sites? Peak sets: SUZ12 (ENCODE HEK293T), H3K9me3 (ENCODE HEK293),
# ReprPC domains (HEK293T ChromHMM; H3K27me3 proxy, no H3K27me3/EZH2 in ENCODE for HEK293/293T).
#   step "sets":  1 kb genome bins overlapping each set (mappable >= 0.5) + a background of non-set bins matched on
#                 GC x k50-mappability deciles, 5x the set's bin count (seed 1)
#   step "count": per TF (raw read 5' ends, fragments on /vast read-only): per-cell reads total / in set bins / in background bins
#   step "summary": fraction of reads in each set; obs/exp vs leave-one-out pooled panel (TF fraction / other TFs' fraction);
#                 enrichment vs matched background (reads per bin in set / per bin in background); ranks among 79;
#                 depth-matched version: TFs with > 3,400 cells downsampled to 3,400 cells (20 draws, mean).
import json, subprocess, sys
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; E = f"{X}/regulatome/encode_repressive_check"; U = f"{X}/regulatome/gate1/v3_binned/units"
SETS = {"SUZ12_HEK293T": f"{E}/data/ENCFF708RSP.bed.gz", "H3K9me3_HEK293": f"{E}/data/ENCFF037SXA.bed.gz", "ReprPC_domains": f"{U}/reprpc.bed"}
step = sys.argv[1]
g = np.load(f"{U}/grid.npz", allow_pickle=True); ci = {c: i for i, c in enumerate(g["chrom"])}; off = g["offset"]; NB = int(off[-1])

if step == "sets":
    cov = np.load(f"{U}/bin1k_cov.npz"); gc, mp = cov["gc"], cov["map"]; ok = mp >= 0.5
    qg = np.quantile(gc[ok], np.linspace(0, 1, 11)); qm = np.quantile(mp[ok], np.linspace(0, 1, 11))
    cls = np.clip(np.digitize(gc, qg[1:-1]), 0, 9) * 10 + np.clip(np.digitize(mp, qm[1:-1]), 0, 9)
    rng = np.random.default_rng(1); out = {}
    for name, path in SETS.items():
        cat = "zcat" if path.endswith(".gz") else "cat"
        txt = subprocess.run(f"{cat} {path} | cut -f1-3", shell=True, capture_output=True, text=True, check=True).stdout
        iv = pd.read_csv(pd.io.common.StringIO(txt), sep="\t", header=None, names=["chrom", "start", "end"])
        iv = iv[iv.chrom.isin(ci)]
        bins = set()
        for c, s, e in zip(iv.chrom, iv.start, iv.end):
            bins.update(range(off[ci[c]] + s // 1000, off[ci[c]] + (e - 1) // 1000 + 1))
        sb = np.array(sorted(b for b in bins if ok[b]))
        pool = np.flatnonzero(ok); pool = pool[~np.isin(pool, sb)]
        bg = []
        for k, n in zip(*np.unique(cls[sb], return_counts=True)):
            cand = pool[cls[pool] == k]
            if len(cand): bg += list(rng.choice(cand, size=min(5 * n, len(cand)), replace=False))
        out[f"{name}__set"] = sb; out[f"{name}__bg"] = np.array(sorted(bg))
        print(f"{name}: {len(iv):,} intervals -> {len(sb):,} mappable 1 kb bins; background {len(bg):,} bins; "
              f"GC {gc[sb].mean():.3f} vs {gc[np.array(bg)].mean():.3f}; map {mp[sb].mean():.3f} vs {mp[np.array(bg)].mean():.3f}")
    np.savez(f"{E}/set_bins.npz", **out)

elif step == "count":
    tf = sys.argv[2]; S = np.load(f"{E}/set_bins.npz")
    r = pd.read_csv(f"/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment/TF.{tf}.bed.gz", sep="\t", header=None, usecols=[0, 1, 3],
                    names=["chrom", "start", "bc"], dtype={"chrom": str, "start": np.int64, "bc": str})
    r = r[r.chrom.isin(ci)]
    _, cell = np.unique(r.bc.values, return_inverse=True); nc = cell.max() + 1
    gb = off[r.chrom.map(ci).values] + r.start.values // 1000
    cols = {"total": np.bincount(cell, minlength=nc)}
    for k in S.files:
        mask = np.zeros(NB, bool); mask[S[k]] = True
        cols[k] = np.bincount(cell[mask[gb]], minlength=nc)
    pd.DataFrame(cols).to_csv(f"{E}/counts/{tf}.tsv.gz", sep="\t", index=False)
    print(tf, nc, "cells", {k: int(v.sum()) for k, v in cols.items()})

elif step == "summary":
    S = np.load(f"{E}/set_bins.npz"); names = list(SETS)
    tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]
    C = {t: pd.read_csv(f"{E}/counts/{t}.tsv.gz", sep="\t") for t in tfs}
    tot = {k: sum(C[t][k].sum() for t in tfs) for k in C[tfs[0]].columns}
    def stats(df, t, loo=True):
        o = {}
        for n in names:
            fr = df[f"{n}__set"].sum() / df.total.sum()
            oth = (tot[f"{n}__set"] - C[t][f"{n}__set"].sum()) / (tot["total"] - C[t].total.sum())
            per_set = df[f"{n}__set"].sum() / len(S[f"{n}__set"]); per_bg = df[f"{n}__bg"].sum() / len(S[f"{n}__bg"])
            o[f"{n}_frac"] = fr; o[f"{n}_oe_panel"] = fr / oth; o[f"{n}_enr_bg"] = per_set / per_bg if per_bg > 0 else np.nan
        return o
    rows, rng = [], np.random.default_rng(1)
    for t in tfs:
        r = dict(tf=t, cells=len(C[t]), reads=int(C[t].total.sum())); r.update(stats(C[t], t))
        if len(C[t]) > 3400:
            ds = [stats(C[t].iloc[rng.choice(len(C[t]), 3400, replace=False)], t) for _ in range(20)]
            r.update({f"{k}_ds3400": np.mean([d[k] for d in ds]) for k in ds[0]})
        else:
            r.update({f"{k}_ds3400": v for k, v in stats(C[t], t).items()})
        rows.append(r)
    D = pd.DataFrame(rows).set_index("tf")
    for c in [c for c in D.columns if "_oe_panel" in c or "_enr_bg" in c]:
        D[f"rank_{c}"] = D[c].rank(ascending=False).astype(int)
    D.to_csv(f"{E}/repressive_enrichment.tsv", sep="\t", float_format="%.4f")
    pd.set_option("display.width", 250)
    PRC = ["EZH2", "MTF2", "PCGF6", "RBBP4", "RBBP7", "MBD3"]
    K9 = ["SETDB1", "HP1", "ADNP", "CHAMP1", "WIZ", "ZNF644", "ZNF282", "ZNF746", "ZNF768", "ZNF606", "ZNF581", "ZNF79", "ZNF777", "ZNF512B"]
    for n, sel in (("SUZ12_HEK293T", PRC), ("ReprPC_domains", PRC), ("H3K9me3_HEK293", K9)):
        cols = [f"{n}_frac", f"{n}_oe_panel", f"rank_{n}_oe_panel", f"{n}_enr_bg", f"rank_{n}_enr_bg", f"{n}_oe_panel_ds3400",
                f"rank_{n}_oe_panel_ds3400", f"{n}_enr_bg_ds3400", f"rank_{n}_enr_bg_ds3400"]
        print(f"\n== {n} (panel median frac {D[f'{n}_frac'].median():.4f}, oe {D[f'{n}_oe_panel'].median():.2f}, enr_bg {D[f'{n}_enr_bg'].median():.2f})")
        print(D.loc[sel, ["cells"] + cols].round(3).to_string())
        print("top 6 by depth-matched enrichment vs bg:", D[f"{n}_enr_bg_ds3400"].sort_values(ascending=False).head(6).round(2).to_dict())
