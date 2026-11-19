# m46 Amendment 4: remasked imputed matrices for the 7 benchmark TFs (Mpeak, M0, Mdrop5, Mzero, M5, M50), written as impute-style dirs under m46 work/bench_inputs.
import os, json
import numpy as np, scipy.sparse as sp, scipy.io as sio
X = "/work/users/d/y/dyy12/XuLab"; U = f"{X}/unified/work"; W = f"{X}/regulatome/m46_stage1_imputed/work"; OUT = f"{W}/bench_inputs"
MK = dict(np.load(f"{W}/masks.npz")); cpm = np.load(f"{U}/ctcf/impute_atac_cpm_drop5/bin_atac_mean_cpm.npy"); assert len(cpm) == len(MK["M0"]); MK["Mdrop5"] = cpm > 0.0781542
print({k: int(v.sum()) for k, v in MK.items()}, flush=True)
for t in ("ctcf", "maz", "zic2", "zbtb7a", "znf777", "nfya", "znf282"):
    raw = sio.mmread(f"{U}/{t}/mm/matrix.mtx.gz").tocsr(); raw.data[:] = 1; pre = sp.load_npz(f"{U}/{t}/impute/matrix_csr.npz.pre_openmask").tocsr(); pre.data[:] = 1
    only = (pre - pre.multiply(raw)).tocsr(); only.eliminate_zeros(); cur = sp.load_npz(f"{U}/{t}/impute/matrix_csr.npz").tocsr()
    for k in ("M0", "Mdrop5", "Mzero", "M5", "M50", "Mpeak"):
        d = f"{OUT}/{t}/{k}"; os.makedirs(d, exist_ok=True)
        m = (only.multiply(MK[k][:, None]) + raw).tocsr(); m.data[:] = 1; m = m.astype(np.float32)
        if k == "Mpeak": assert abs(m - (cur > 0).astype(np.float32)).sum() == 0, t
        sp.save_npz(f"{d}/matrix_csr.npz", m)
        for f in ("regions.tsv", "barcodes.tsv"):
            if not os.path.exists(f"{d}/{f}"): os.symlink(f"{U}/{t}/impute/{f}", f"{d}/{f}")
        json.dump({"source": f"{U}/{t}/impute/matrix_csr.npz.pre_openmask", "mask": k, "raw_nnz": int(raw.nnz), "nnz": int(m.nnz), "keep_raw": True}, open(f"{d}/meta.json", "w"))
        print(t, k, m.nnz, flush=True)
print("done")
