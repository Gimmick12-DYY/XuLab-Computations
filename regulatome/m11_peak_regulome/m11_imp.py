# m11 Amendment 5: imputed-data peaks for six TFs. usage: m11_imp.py prep -> recall/imputed/pseudobulk.npz ; m11_imp.py call <i> (i = TF x type x variant) -> recall/imputed/<variant>_<type>/<TF>/TF.<TF>_peaks.narrowPeak
import json, os, subprocess, sys
import numpy as np, pandas as pd, scipy.sparse as sp
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; U = f"{X}/unified/work"; IM = f"{RG}/m11_peak_regulome/recall/imputed"; os.makedirs(IM, exist_ok=True)
MACS = "/work/users/d/y/dyy12/conda/envs/atac/bin/macs2"; TFS = ["NRF1", "CTCF", "ZNF143", "MAZ", "NFYA", "E4F1"]; tfs79 = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]
if sys.argv[1] == "prep":
    reg = pd.read_csv(f"{U}/nrf1/impute/regions.tsv", header=None)[0].str.extract(r"(chr[^:]+):(\d+)-(\d+)"); reg.columns = ["c", "s", "e"]; reg[["s", "e"]] = reg[["s", "e"]].astype(int)
    UA = pd.read_csv(f"{X}/data/HEK293T_ATAC_union3.bed.gz", sep="\t", header=None, usecols=[0, 1, 2]); open_ = np.zeros(len(reg), bool)
    for c, G in UA.groupby(0):
        m = np.flatnonzero(reg.c.values == c)
        if not len(m): continue
        iv = G[[1, 2]].values; iv = iv[np.argsort(iv[:, 0])]; me = np.maximum.accumulate(iv[:, 1]); k = np.searchsorted(iv[:, 0], reg.e.values[m], side="left") - 1; ok = k >= 0; ok[ok] = me[k[ok]] > reg.s.values[m][ok]; open_[m] = ok
    oi = np.flatnonzero(open_); FR = {}; NC = {}; CO = {}
    for t in tfs79:
        f = f"{U}/{t.lower()}/impute/matrix_csr.npz"
        if os.path.exists(f): M = sp.load_npz(f); FR[t] = np.asarray(M.sum(1)).ravel()[oi] / M.shape[1]; NC[t] = M.shape[1]
        g = f"{U}/{t.lower()}/predict/pseudobulk_audit_bins.npz"
        if os.path.exists(g): z = np.load(g); v = np.full(len(reg), np.nan); v[z["bins_mm"]] = z["mean_score"]; CO[t] = np.nan_to_num(v[oi]); NC[t + "|cont"] = int(z["n_cells_used"])
        print(t, flush=True)
    # OmniATAC Tn5 5' ends per open bin
    A = pd.read_csv(f"{RG}/m2_abc/inputs/omniatac_merged.tn5.tagAlign.gz", sep="\t", header=None, usecols=[0, 1, 2, 5], names=["c", "s", "e", "st"]); pos = np.where(A.st.values == "+", A.s.values, A.e.values - 1); atac = np.zeros(len(oi))
    R = reg.iloc[oi].reset_index(drop=True)
    for c, ix in A.groupby("c", sort=False).indices.items():
        m = np.flatnonzero(R.c.values == c)
        if len(m): p = np.sort(pos[ix]); atac[m] = np.searchsorted(p, R.e.values[m]) - np.searchsorted(p, R.s.values[m])
    np.savez(f"{IM}/pseudobulk.npz", c=R.c.values, s=R.s.values, e=R.e.values, atac=atac, **{f"bin|{k}": v for k, v in FR.items()}, **{f"cont|{k}": v for k, v in CO.items()})
    json.dump(NC, open(f"{IM}/n_cells.json", "w")); print("open bins", len(oi), "bin TFs", len(FR), "cont TFs", len(CO))
else:
    Z = np.load(f"{IM}/pseudobulk.npz", allow_pickle=True); NC = json.load(open(f"{IM}/n_cells.json")); jobs = [(t, ty, v) for t in TFS for ty in ("bin", "cont") for v in ("V1imp", "V2imp")]; t, ty, v = jobs[int(sys.argv[2])]
    if f"{ty}|{t}" not in Z.files: print("no", ty, "for", t); sys.exit(0)
    O = f"{IM}/{v}_{ty}/{t}"; os.makedirs(O, exist_ok=True); n = NC[t] if ty == "bin" else NC[t + "|cont"]; tr = Z[f"{ty}|{t}"] * n
    others = [k for k in Z.files if k.startswith(ty + "|") and k != f"{ty}|{t}"]
    ct = np.mean([Z[k] for k in others], axis=0) * n if v == "V1imp" else Z["atac"] * (tr.sum() / max(Z["atac"].sum(), 1))
    B = pd.DataFrame(dict(c=Z["c"], s=Z["s"], e=Z["e"]))
    B.assign(v=tr).to_csv(f"{O}/treat.bdg", sep="\t", header=False, index=False, float_format="%.5f"); B.assign(v=ct).to_csv(f"{O}/ctrl.bdg", sep="\t", header=False, index=False, float_format="%.5f")
    subprocess.run(f"sort -k1,1 -k2,2n {O}/treat.bdg -o {O}/treat.bdg; sort -k1,1 -k2,2n {O}/ctrl.bdg -o {O}/ctrl.bdg; {MACS} bdgcmp -t {O}/treat.bdg -c {O}/ctrl.bdg -m qpois -o {O}/q.bdg && {MACS} bdgpeakcall -i {O}/q.bdg -c 1.301 -l 1000 -g 0 -o {O}/raw_peaks.narrowPeak", shell=True, check=True)
    P = pd.read_csv(f"{O}/raw_peaks.narrowPeak", sep="\t", header=None, skiprows=1) if os.path.getsize(f"{O}/raw_peaks.narrowPeak") > 0 else pd.DataFrame(columns=range(10))
    key = dict(zip(zip(B.c, B.s), range(len(B)))); fe = (tr + 1) / (ct + 1); rows = []
    for r in P.itertuples(index=False):
        sm = int(r[1]) + int(r[9]); bs = (sm // 1000) * 1000; i = key.get((r[0], bs)); ctr = bs + 500 if i is not None else sm
        rows.append([r[0], int(r[1]), int(r[2]), f"{t}_{v}_{ty}_{len(rows)}", int(r[4]), ".", float(fe[i]) if i is not None else np.nan, -1, float(r[8]), ctr - int(r[1])])
    pd.DataFrame(rows).to_csv(f"{O}/TF.{t}_peaks.narrowPeak", sep="\t", header=False, index=False)
    for f in ("treat.bdg", "ctrl.bdg", "q.bdg"): os.remove(f"{O}/{f}")
    print(t, ty, v, len(rows), "control from", len(others) if v == "V1imp" else "OmniATAC")
