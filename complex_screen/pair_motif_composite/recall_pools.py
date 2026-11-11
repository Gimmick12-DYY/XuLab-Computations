# Peak re-calling, gate pools for one (variant, TF). Variants: raw_lab (lab MACS2, no control), raw_tf (control = other 78 TFs),
# raw_atac (control = HEK293T ATAC), imp_loo, imp_atac (imputed re-scored; recall_prep.py), imp_binom (binomial peak calling on the imputed bin x cell matrices vs the other 78 TFs; recall_imp_binom.py). Foreground: top 1,000 summits by -log10 p
# (raw; summit +/- 100 bp inside an eligible, non-HOT union-ATAC bin) or top 1,000 eligible non-HOT bins by score (imputed; bin centre
# +/- 100 bp). Background: eligible non-HOT bins outside the TF's top set, 2 per foreground site, matched on GC 0.025 x CpG quintile x
# promoter x ATAC-level quintile (chromVAR-style). Discovery = odd autosomes, held-out = even + chrX.
# Usage: python recall_pools.py <variant> <TF>  -> recall/pools/<variant>/<TF>/{disc,val}/{fg,bg}.bed, recall/pools/<variant>/<TF>.json
import os, sys, json, subprocess
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; R = f"{O}/recall"; N = 1000; v, t = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(abs(hash((v, t))) % 2**32)
Q = pd.read_csv(f"{O}/ref/bins_qc.tsv", sep="\t", usecols=["chrom", "s", "e", "bin"]); F = np.load(f"{R}/ref.npz"); strat, elig, part, hot = F["strat"], F["elig"], F["part"], F["hot"]
ok = elig & ~hot; out = f"{R}/pools/{v}/{t}"; info = dict(variant=v, tf=t)
def wb(fn, rows):
    with open(fn, "w") as f:
        for r in rows: f.write("\t".join(map(str, r)) + "\n")
if v.startswith("raw"):
    src = {"raw_lab": f"/vast/som/xujie_lab/TAGATG-293T/bigwig/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", "raw_tf": f"{R}/peaks_tf/TF.{t}_peaks.narrowPeak", "raw_atac": f"{R}/peaks_atac/TF.{t}_peaks.narrowPeak"}[v]
    pk = pd.read_csv(src, sep="\t", header=None, usecols=[0, 1, 7, 9], names=["c", "s", "lp", "sm"]); pk = pk[pk.lp >= 2].sort_values("lp", ascending=False); pk["m"] = pk.s + pk.sm
    info["n_peaks_p01"] = len(pk); info["n_peaks_p1e5"] = int((pk.lp >= 5).sum())
    B = pd.read_csv(f"{O}/ref/bins1kb.bed", sep="\t", header=None, names=["c", "s", "e", "i"]); idx = np.full(len(pk), -1)
    for c, g in pk.groupby("c"):
        b = B[B.c == c]
        if not len(b): continue
        k = np.searchsorted(b.s.values, g.m.values, side="right") - 1; inb = (k >= 0) & (g.m.values < b.e.values[np.maximum(k, 0)]); idx[np.flatnonzero((pk.c == c).values)[inb]] = b.i.values[k[inb]]
    pk["bin"] = idx; pk = pk[(pk.bin >= 0)]; pk = pk[ok[pk.bin.values]].drop_duplicates("bin").head(N)
    fg = [(r.c, r.m - 100, r.m + 100, f"{t}_p{k}", int(r.bin)) for k, r in enumerate(pk.itertuples())]; info["min_lp_fg"] = round(float(pk.lp.min()), 2) if len(pk) else None
else:
    key = v.split("_")[1]; fz = f"{R}/imputed_scores_{key}.npz"; z = np.load(fz if os.path.exists(fz) else f"{R}/imputed_scores.npz"); s = z[key][list(z["tfs"]).index(t)]; cand = np.flatnonzero(ok); top = cand[np.argsort(-s[cand], kind="stable")[:N]]
    fg = [(Q.chrom[i], Q.s[i], Q.e[i], f"b{Q.bin[i]}", int(i)) for i in top]
topbins = np.array([r[4] for r in fg], int); used = np.zeros(len(ok), bool); used[topbins] = True; free = np.flatnonzero(ok & ~used)
for pt in ["disc", "val"]:
    f_ = [r for r in fg if part[r[4]] == pt]; rest = free[part[free] == pt]; bg = []
    for s_ in np.unique([strat[r[4]] for r in f_]):
        m = sum(strat[r[4]] == s_ for r in f_); c = rest[strat[rest] == s_]
        if len(c): bg += list(rng.choice(c, min(2 * m, len(c)), replace=False))
    os.makedirs(f"{out}/{pt}", exist_ok=True); wb(f"{out}/{pt}/fg.bed", [(a, b, c, d, ".", "+") for a, b, c, d, _ in f_]); wb(f"{out}/{pt}/bg.bed", [(Q.chrom[i], Q.s[i], Q.e[i], f"b{Q.bin[i]}", ".", "+") for i in sorted(bg)])
    info[f"{pt}_fg"] = len(f_); info[f"{pt}_bg"] = len(bg)
json.dump(info, open(f"{out}.json", "w")); print(info)
