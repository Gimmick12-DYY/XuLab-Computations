# Guards for the DDIT3-NFYA shared-motif call: NFY (and Chop) enrichment in the co-bound pool vs partner-only for 20 random NFYA partners
# and 20 random DDIT3 partners (nfya_null/), compared with DDIT3-NFYA; NFY-only permutation p over 200 permutations; CentriMo per pool.
# Output: nfya_null/nfy_guard.tsv and a printed summary.
import os, glob, math
import numpy as np, pandas as pd
import summarize as S
O = os.path.dirname(os.path.abspath(__file__)); N = f"{O}/nfya_null"; T = f"{O}/triple_support"
def get(k, m):
    if k is None: return (np.nan, np.nan)
    k = k.copy(); k.index = [S.short(x) for x in k.index]; k = k[~k.index.duplicated()]
    return (round(k.loc[m].fold, 2), round(k.loc[m].log10p, 2)) if m in k.index else (np.nan, np.nan)
rows = []
for p in pd.read_csv(f"{N}/pool_sizes.tsv", sep="\t").pair:
    d, v = S.known(f"{N}/results/{p}/homer_vsB/known/knownResults.txt"), S.known(f"{N}/results/{p}/val_vsB/knownResults.txt")
    for m in ["NFY(CCAAT)", "Chop(bZIP)"]:
        f, lp = get(d, m); vf, vp = get(v, m); rows.append(dict(pair=p, focal=p.split("-")[0], motif=m, fold=f, log10p=lp, val_fold=vf, val_log10p=vp))
G = pd.DataFrame(rows)
d, v = S.known(f"{T}/results/DDIT3-NFYA/homer_vsA/known/knownResults.txt"), S.known(f"{T}/results/DDIT3-NFYA/val_vsA/knownResults.txt")
obs = {m: get(d, m) + get(v, m) for m in ["NFY(CCAAT)", "Chop(bZIP)"]}
d2, v2 = S.known(f"{T}/results/DDIT3-NFYA/homer_vsB/known/knownResults.txt"), S.known(f"{T}/results/DDIT3-NFYA/val_vsB/knownResults.txt")
obsB = {m: get(d2, m) + get(v2, m) for m in ["NFY(CCAAT)", "Chop(bZIP)"]}
perm = []
for dd in sorted(glob.glob(f"{T}/results/DDIT3-NFYA/perm/*")):
    k = S.known(f"{dd}/vsA/knownResults.txt")
    if k is not None: perm.append(get(k, "NFY(CCAAT)")[1])
perm = np.array(perm); pp = (1 + np.sum(perm <= obs["NFY(CCAAT)"][1])) / (1 + len(perm))
G.to_csv(f"{N}/nfy_guard.tsv", sep="\t", index=False)
nf = G[(G.focal == "NFYA") & (G.motif == "NFY(CCAAT)")]
print("DDIT3-NFYA NFY vs DDIT3-only: fold %.2f log10p %.1f | held-out fold %.2f log10p %.1f" % obs["NFY(CCAAT)"])
print("DDIT3-NFYA NFY vs NFYA-only:  fold %.2f log10p %.1f | held-out fold %.2f log10p %.1f" % obsB["NFY(CCAAT)"])
print("DDIT3-NFYA Chop vs NFYA-only: fold %.2f log10p %.1f | held-out fold %.2f log10p %.1f" % obsB["Chop(bZIP)"])
print(f"NFY-only permutation p = {pp:.4f} over {len(perm)} permutations (min perm log10p {perm.min():.1f})")
for lab, s in [("NFYA + 20 random partners: NFY vs partner-only", nf), ("DDIT3 + 20 random partners: NFY vs partner-only", G[(G.focal == "DDIT3") & (G.motif == "NFY(CCAAT)")]), ("DDIT3 + 20 random partners: Chop vs partner-only", G[(G.focal == "DDIT3") & (G.motif == "Chop(bZIP)")])]:
    print(lab, "| n", len(s), "| fold>=1.3 & p<=1e-3:", int(((s.fold >= 1.3) & (s.log10p <= -3)).sum()), "| fold>=1.5 & p<=1e-5:", int(((s.fold >= 1.5) & (s.log10p <= -5)).sum()),
          "| + held-out log2FC>0 & P<0.05:", int(((s.fold >= 1.3) & (s.log10p <= -3) & (s.val_fold > 1) & (s.val_log10p < math.log10(.05))).sum()))
r = 1 + int((nf.fold >= obs["NFY(CCAAT)"][0]).sum()); print(f"DDIT3-NFYA NFY fold {obs['NFY(CCAAT)'][0]} ranks {r} of {len(nf) + 1} among NFYA partners (fold); by p rank {1 + int((nf.log10p <= obs['NFY(CCAAT)'][1]).sum())}")
print(nf.sort_values("fold", ascending=False).to_string(index=False))
for pl in ["pair", "Aonly", "Bonly"]:
    f = f"{T}/results/DDIT3-NFYA/centrimo_nfy_{pl}/centrimo.tsv"
    if os.path.exists(f):
        c = pd.read_csv(f, sep="\t", comment="#"); print("CentriMo", pl, c[["motif_id", "motif_alt_id", "E-value", "bin_width", "sites_in_bin", "total_sites"]].to_string(index=False) if len(c) else "no central enrichment reported")
