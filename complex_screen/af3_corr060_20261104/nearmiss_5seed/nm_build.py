# Near-miss list for 5 seeds: 1-seed sweep pairs (scores_s1) with tier "none" and (max ipSAE > 0 or top ipTM >= 0.3) -- set (c) of 2026-11-09,
# recomputed on all pairs scored now (includes big pairs scored since). Splits regular vs big (pairs_big.txt, need 128G). usage: nm_build.py
import glob, os
O = "/work/users/d/y/dyy12/XuLab/complex_screen/af3_corr060_20261104"; N = f"{O}/nearmiss_5seed"; big = set(open(f"{O}/pairs_big.txt").read().split()); reg, bg = [], []
for f in sorted(glob.glob(f"{O}/scores_s1/*.tsv")):
    x = open(f).read().split("\t"); p = os.path.basename(f)[:-4]
    if x[4] == "none" and (float(x[2]) > 0 or float(x[1]) >= .3): (bg if p in big else reg).append(p)
open(f"{N}/pairs_nm.txt", "w").write("\n".join(reg) + "\n"); open(f"{N}/pairs_nm_big.txt", "w").write("\n".join(bg) + "\n"); print("regular", len(reg), "big", len(bg))
