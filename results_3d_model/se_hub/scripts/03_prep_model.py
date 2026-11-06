"""SE-hub step 2 prep: Hi-C-first model inputs for a window around the chosen SE hub (same settings as tf_polymer_hicfirst).
Hub = one loop-supported anchor-hub core (inside a HEK293 ROSE SE) and all its loop-supported partners (anchor_hub_partners.tsv,
Cicero edges on a WT Peakachu 0.7 loop, +-30 kb); these hub loop-edges are always-on restraints. Hi-C pixels (10 kb, O/E >= 2,
2-100 bins) are probabilistic restraints (p 0.1-0.6 by O/E rank; 20% held out, seed 7); Peakachu 0.7 loops in the window
(5 + 10 kb, de-duplicated within +-5 beads) probabilistic (p 0.2-0.8 by O/E rank). No TF term (beads carry zero TF weights).
Writes inputs/: beads.tsv, hic_10kb.npz, restraints.tsv, heldout_pixels.tsv, hub_edges.tsv (browser-style), hub_elements.tsv.
"""
import argparse, glob, gzip, os, re
from pathlib import Path
import numpy as np, pandas as pd, cooler

ap = argparse.ArgumentParser()
ap.add_argument("--core", required=True); ap.add_argument("--chrom", required=True); ap.add_argument("--start", type=int, required=True)
ap.add_argument("--end", type=int, required=True); ap.add_argument("--hubdir", required=True); ap.add_argument("--mcool", required=True)
ap.add_argument("--loops", nargs="+", required=True); ap.add_argument("--peaks-dir", required=True); ap.add_argument("--gtf", required=True)
ap.add_argument("--se", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
CH, L0, L1 = a.chrom, a.start, a.end; nb = (L1 - L0) // 2000; n10 = (L1 - L0) // 10000
def parse(r): c, s = r.split(":"); s, e = map(int, s.split("-")); return c, s, e
def mid(r): _, s, e = parse(r); return (s + e) // 2
tb = lambda p: (p - L0) // 2000
# genes (TSS) for labels
G = []
with gzip.open(a.gtf, "rt") as g:
    for ln in g:
        if not ln.startswith(CH + "\t"): continue
        x = ln.split("\t", 9)
        if x[2] != "gene" or not re.search(r'gene_type "(protein_coding|lncRNA)"', x[8]): continue
        G.append((int(x[3]) if x[6] == "+" else int(x[4]), re.search(r'gene_name "([^"]+)"', x[8]).group(1)))
G = pd.DataFrame(G, columns=["tss", "gene"])
def label(r):
    _, s, e = parse(r); d = G[(G.tss >= s - 2500) & (G.tss <= e + 2500)]
    if len(d): return "/".join(sorted(set(d.gene)))[:22]
    k = (G.tss - (s + e) // 2).abs().idxmin(); return f"{G.gene[k]} {'+' if G.tss[k] < s else '-'}{abs(G.tss[k] - (s + e) // 2) // 1000} kb"
SE = pd.read_csv(a.se, sep="\t"); SE = SE[SE.chrom == CH]
HP = pd.read_csv(Path(a.hubdir) / "anchor_hub_partners.tsv", sep="\t"); P = HP[HP.core == a.core]
P = P[[L0 <= mid(x) < L1 for x in P.partner]]
E = pd.DataFrame(dict(region1=a.core, region2=P.partner.values, kb=P.distance_bp.values / 1000, n_tfs=P.n_tfs_linking.values,
                      **{"on_loop_0.7": 1}, tfs=P.tfs_linking.values))
E = E[E.kb > 25]
E.to_csv(O / "hub_edges.tsv", sep="\t", index=False)
els = [a.core] + sorted(set(E.region2), key=mid)
def in_se(r):
    _, s, e = parse(r); h = SE[(SE.end > s) & (SE.start < e)]; return h.se_id.iloc[0] if len(h) else ""
EL = pd.DataFrame(dict(element=els, bead=[tb(mid(r)) for r in els], label=["SE " + label(els[0])] + [label(r) for r in els[1:]], se=[in_se(r) for r in els],
                       is_core=[True] + [False] * (len(els) - 1)))
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
for r in E.itertuples():
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
print(f"{reg}: {nb} beads; kept 10 kb bins {keep.sum()}/{n10}; hub edges {len(E)} (SE core {a.core}); elements {len(EL)}; "
      f"loops {len(LR)}; pixels O/E>=2 {len(Px) + ho.sum()} (held out {ho.sum()}); restraints {R.kind.value_counts().to_dict()}")
print(EL.to_string(index=False))
