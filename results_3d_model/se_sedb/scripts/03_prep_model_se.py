"""Hi-C-first model inputs around the exact SEdb SE SE_02_195100173 (same settings as tf_polymer_hicfirst). Edges come from edges_exact_se.tsv (anchors strictly inside the SE). Loop-edges (WT Peakachu 0.7 loop, +-30 kb) are always-on restraints; Cicero-only edges are not restrained and serve as a held-out test. Hi-C pixels (10 kb, O/E >= 2,
2-100 bins) are probabilistic restraints (p 0.1-0.6 by O/E rank; 20% held out, seed 7); Peakachu 0.7 loops in the window
(5 + 10 kb, de-duplicated within +-5 beads) probabilistic (p 0.2-0.8 by O/E rank). No TF term (beads carry zero TF weights).
Writes inputs/: beads.tsv, hic_10kb.npz, restraints.tsv, heldout_pixels.tsv, hub_edges.tsv (browser-style), hub_elements.tsv.
"""
import argparse, glob, gzip, os, re
from pathlib import Path
import numpy as np, pandas as pd, cooler

ap = argparse.ArgumentParser()
ap.add_argument("--edges", required=True); ap.add_argument("--se-region", required=True); ap.add_argument("--chrom", required=True); ap.add_argument("--start", type=int, required=True)
ap.add_argument("--end", type=int, required=True); ap.add_argument("--mcool", required=True)
ap.add_argument("--loops", nargs="+", required=True); ap.add_argument("--peaks-dir", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args(); O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
CH, L0, L1 = a.chrom, a.start, a.end; nb = (L1 - L0) // 2000; n10 = (L1 - L0) // 10000
def parse(r): c, s = r.split(":"); s, e = map(int, s.split("-")); return c, s, e
def mid(r): _, s, e = parse(r); return (s + e) // 2
tb = lambda p: (p - L0) // 2000
E0 = pd.read_csv(a.edges, sep="\t"); E0 = E0[[L0 <= mid(x) < L1 for x in E0.partner]]
E = pd.DataFrame(dict(region1=E0.anchor, region2=E0.partner, kb=E0.kb, n_tfs=E0.n_tfs, **{"on_loop_0.7": E0.on_loop}, tfs=E0.tfs, cls=E0.cls, label=E0.label))
E.to_csv(O / "hub_edges.tsv", sep="\t", index=False); E[["region1", "region2", "kb", "n_tfs", "on_loop_0.7", "tfs"]].to_csv(O / "hub_edges_shared.tsv", sep="\t", index=False)
EL = pd.DataFrame(dict(element=[a.se_region] + list(E.region2), bead=[tb(mid(a.se_region))] + [tb(mid(r)) for r in E.region2], label=["super-enhancer (SEdb, HEK293T)"] + list(E.label), cls=["se"] + list(E.cls), on_loop=[1] + list(E["on_loop_0.7"]), is_core=[True] + [False] * len(E)))
# TF counts per bead
M = {}
for f in sorted(glob.glob(os.path.join(a.peaks_dir, "TF.*_peaks.q005.narrowPeak"))):
    tf = re.sub(r"^TF\.|_peaks\.q005\.narrowPeak$", "", os.path.basename(f))
    for ln in open(f):
        x = ln.split("\t")
        if x[0] != CH: continue
        p = int(x[1]) + int(x[9])
        if L0 <= p < L1: M.setdefault(tb(p), set()).add(tf)
B = pd.DataFrame(dict(bead=np.arange(nb), start=L0 + 2000 * np.arange(nb))); B["end"] = B.start + 2000
B["n_tfs"] = [len(M.get(b, ())) for b in B.bead]; B["tfs"] = [",".join(sorted(M.get(b, ()))) for b in B.bead]
B["cls"] = np.select([B.n_tfs >= 5, B.n_tfs >= 2, B.n_tfs == 1], [3, 2, 1], 0); B["bind"] = False
for k in range(6): B[f"w_{k}"] = 0.0
B["hub_label"] = ""
for r in EL.itertuples(): B.at[r.bead, "hub_label"] = r.label
B.to_csv(O / "beads.tsv", sep="\t", index=False)
EL["n_tfs_bead"] = [B.n_tfs[b] for b in EL.bead]; EL.to_csv(O / "hub_elements.tsv", sep="\t", index=False)
# Hi-C
clr = cooler.Cooler(f"{a.mcool}::/resolutions/10000"); reg = f"{CH}:{L0}-{L1}"
ice = clr.matrix(balance=True).fetch(reg); raw = clr.matrix(balance=False).fetch(reg)
cov = raw.sum(1) - np.diag(raw); w = clr.bins().fetch(reg)["weight"].values
keep = np.isfinite(w) & (cov >= 0.2 * np.median(cov[cov > 0]))
np.savez_compressed(O / "hic_10kb.npz", ice=ice, raw=raw, keep=keep)
exp = np.full(n10, np.nan)
for d in range(n10):
    i = np.arange(n10 - d); j = i + d; ok = keep[i] & keep[j] & np.isfinite(ice[i, j])
    if ok.any(): exp[d] = ice[i[ok], j[ok]].mean()
def oe(i, j): return ice[i, j] / exp[j - i] if keep[i] and keep[j] and np.isfinite(ice[i, j]) else np.nan
# loops
rows = []
for f in a.loops:
    for ln in open(f):
        x = ln.split()
        if x[0] != CH or x[3] != CH: continue
        s1, e1, s2, e2 = map(int, (x[1], x[2], x[4], x[5]))
        if min(s1, s2) < L0 or max(e1, e2) > L1: continue
        c1, c2 = sorted(((s1 + e1) // 2, (s2 + e2) // 2)); rows.append(dict(b1=tb(c1), b2=tb(c2), score=float(x[6]) if len(x) > 6 else np.nan))
Lp = pd.DataFrame(rows).sort_values("score", ascending=False); kept = []
for r in Lp.itertuples():
    if r.b2 - r.b1 < 3 or any(abs(r.b1 - k.b1) <= 5 and abs(r.b2 - k.b2) <= 5 for k in kept): continue
    kept.append(r)
LR = pd.DataFrame(kept)[["b1", "b2"]]; LR["bin1"], LR["bin2"] = LR.b1 // 5, LR.b2 // 5
LR["oe"] = [oe(i, j) for i, j in zip(LR.bin1, LR.bin2)]; LR["oe"] = LR.oe.fillna(LR.oe.median())
LR["p_on"] = 0.2 + 0.6 * (LR.oe.rank(pct=True) - 1 / len(LR)) / (1 - 1 / len(LR)) if len(LR) > 1 else 0.5
red = []
for r in E[E["on_loop_0.7"] == 1].itertuples():
    b1, b2 = sorted((tb(mid(r.region1)), tb(mid(r.region2)))); red.append(dict(kind="red_edge", b1=b1, b2=b2, bin1=b1 // 5, bin2=b2 // 5, oe=oe(b1 // 5, b2 // 5), p_on=1.0, k=10.0, r0=1.0))
loops = [dict(kind="loop", b1=r.b1, b2=r.b2, bin1=r.bin1, bin2=r.bin2, oe=r.oe, p_on=r.p_on, k=10.0, r0=1.0) for r in LR.itertuples()]
taken = np.zeros((n10, n10), bool)
for x in red + loops: taken[max(0, x["bin1"] - 1):x["bin1"] + 2, max(0, x["bin2"] - 1):x["bin2"] + 2] = True
pix = [dict(bin1=i, bin2=i + d, oe=oe(i, i + d)) for d in range(2, 101) for i in range(n10 - d) if not taken[i, i + d] and np.isfinite(oe(i, i + d)) and oe(i, i + d) >= 2]
Px = pd.DataFrame(pix); ho = np.random.default_rng(7).random(len(Px)) < 0.2
Px[ho].to_csv(O / "heldout_pixels.tsv", sep="\t", index=False); Px = Px[~ho].copy()
Px["p_on"] = 0.1 + 0.5 * Px.oe.rank(pct=True); Px["kind"] = "pixel"; Px["k"] = 3.0; Px["r0"] = 1.5; Px["b1"] = Px.bin1 * 5 + 2; Px["b2"] = Px.bin2 * 5 + 2
R = pd.concat([pd.DataFrame(red), pd.DataFrame(loops), Px[["kind", "b1", "b2", "bin1", "bin2", "oe", "p_on", "k", "r0"]]], ignore_index=True)
R.to_csv(O / "restraints.tsv", sep="\t", index=False)
print(f"{reg}: {nb} beads; kept 10 kb bins {keep.sum()}/{n10}; hub edges {len(E)} (SE {a.se_region}); elements {len(EL)}; "
      f"loops {len(LR)}; pixels O/E>=2 {len(Px) + ho.sum()} (held out {ho.sum()}); restraints {R.kind.value_counts().to_dict()}")
print(EL.to_string(index=False))
