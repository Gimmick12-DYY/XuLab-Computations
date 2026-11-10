# repro_v1 Run 2c step 1 (README Amendment 4): per-half MACS3 peaks for one TF, from that half's reads only (Run 2b split).
# usage: run2c_peaks.py <task index 0..78>
import os, sys, gzip, subprocess, numpy as np
X = "/work/users/d/y/dyy12/XuLab"; R = f"{X}/regulatome"; O = f"{R}/m6_improve/repro_v1/run2c"; FR = "/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment"
MACS = "/work/users/d/y/dyy12/conda/envs/snapatac2/bin/macs3"
tfs = [l.strip() for l in open(f"{R}/m1_binding/v2/tfs.txt") if l.strip()]; i = int(sys.argv[1]); t = tfs[i]
bc = np.load(f"{R}/gate1/cellmats/{t}.cells.npz", allow_pickle=True)["barcodes"].astype(str); n = len(bc)
hh = np.random.default_rng(20261029 + i).permutation(n) < n // 2; H = {b: (0 if h else 1) for b, h in zip(bc, hh)}
tmp = f"{O}/tmp/{t}"; os.makedirs(tmp, exist_ok=True); os.makedirs(f"{O}/peaks", exist_ok=True)
fh = [open(f"{tmp}/h{k + 1}.bed", "w") for k in (0, 1)]; nr = [0, 0]
with gzip.open(f"{FR}/TF.{t}.bed.gz", "rt") as f:
    for l in f:
        k = H.get(l.split("\t", 4)[3])
        if k is not None: fh[k].write(l); nr[k] += 1
for x in fh: x.close()
for k in (0, 1):
    subprocess.run([MACS, "callpeak", "-t", f"{tmp}/h{k + 1}.bed", "-f", "BED", "-g", "hs", "--nomodel", "--shift", "-75", "--extsize", "150",
                    "--keep-dup", "all", "-p", "0.01", "-n", f"{t}_h{k + 1}", "--outdir", f"{O}/peaks"], check=True, stderr=subprocess.DEVNULL)
    for ext in ("_peaks.xls", "_summits.bed"):
        p = f"{O}/peaks/{t}_h{k + 1}{ext}"; os.path.exists(p) and os.remove(p)
    os.remove(f"{tmp}/h{k + 1}.bed")
os.rmdir(tmp)
print(t, n, "reads per half", nr, "peaks", [sum(1 for _ in open(f"{O}/peaks/{t}_h{k + 1}_peaks.narrowPeak")) for k in (0, 1)])
