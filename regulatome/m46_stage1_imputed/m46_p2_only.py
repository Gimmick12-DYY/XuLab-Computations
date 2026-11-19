# m46 Amendment 6: P2 redone (complete treatment track, bdgcmp -m ppois, bdgpeakcall -c 5) for one mask and TF. usage: m46_p2_only.py <mask> <TF index in work/tfs_mask_ok.txt or tfs63.txt for M0>
import os, sys, subprocess
import numpy as np, pandas as pd, scipy.sparse as sp, scipy.io as sio
X = "/work/users/d/y/dyy12/XuLab"; U = f"{X}/unified/work"; M = f"{X}/regulatome/m46_stage1_imputed"; W = f"{M}/work"; MACS = "macs3"
MK = sys.argv[1]; L = f"{M}/tfs63.txt" if MK == "M0" else f"{W}/tfs_mask_ok.txt"; TFS = [l.strip() for l in open(L) if l.strip()]; i = int(sys.argv[2])
if i >= len(TFS): sys.exit(0)
t = TFS[i]; tl = t.lower(); PK = f"{M}/peaks" if MK == "M0" else f"{M}/peaks_{MK}"; wd = f"{W}/p2_{MK}_{t}"; os.makedirs(wd, exist_ok=True); os.makedirs(f"{PK}/{t}", exist_ok=True)
run = lambda c: subprocess.run(c, shell=True, check=True, executable="/bin/bash")
reg = [l.strip() for l in open(f"{U}/nfya/impute/regions.tsv")]; R = pd.DataFrame({"c": [r.split(":")[0] for r in reg], "s": [int(r.split(":")[1].split("-")[0]) for r in reg]}); R["e"] = R.s + 1000
fp = f"{U}/{tl}/impute/matrix_csr.npz.pre_openmask"
if MK == "M0":
    m = sp.load_npz(fp if os.path.exists(fp) else f"{U}/{tl}/impute/matrix_csr.npz").tocsr(); m.data[:] = 1
else:
    mk = np.load(f"{W}/masks.npz")[MK] if MK in np.load(f"{W}/masks.npz").files else np.load(f"{W}/masks_extra.npz")[MK]
    raw = sio.mmread(f"{U}/{tl}/mm/matrix.mtx.gz").tocsr(); raw.data[:] = 1; pre = sp.load_npz(fp).tocsr(); pre.data[:] = 1
    only = (pre - pre.multiply(raw)).tocsr(); only.eliminate_zeros(); m = (only.multiply(mk[:, None]) + raw).tocsr(); m.data[:] = 1
v = np.asarray(m.sum(1)).ravel().astype(float); del m
A = np.load(f"{W}/atac_per_bin.npy"); sc = A * v.sum() / A.sum(); lam = sc.copy(); gm = sc.mean()
for ch, g in R.groupby("c"):
    ii = g.index.values; x = sc[ii]; k = np.convolve(x, np.ones(11) / 11, mode="same"); lam[ii] = np.maximum(np.maximum(x, k), gm)
T = R.copy(); T["v"] = v; T.sort_values(["c", "s"]).to_csv(f"{wd}/treat.bdg", sep="\t", header=False, index=False, float_format="%.5g")
C = R.copy(); C["v"] = lam; C.sort_values(["c", "s"]).to_csv(f"{wd}/lambda.bdg", sep="\t", header=False, index=False, float_format="%.5g")
run(f"{MACS} bdgcmp -t {wd}/treat.bdg -c {wd}/lambda.bdg -m ppois --outdir {wd} -o ppois.bdg > /dev/null 2>&1")
run(f"{MACS} bdgpeakcall -i {wd}/ppois.bdg -c 5 -l 1000 -g 1000 --outdir {PK}/{t} -o P2.narrowPeak > /dev/null 2>&1")
n = sum(1 for l in open(f"{PK}/{t}/P2.narrowPeak") if not l.startswith("track")); print(t, MK, "P2 peaks", n)
for f in ("treat.bdg", "lambda.bdg", "ppois.bdg"): os.remove(f"{wd}/{f}")
os.rmdir(wd)
