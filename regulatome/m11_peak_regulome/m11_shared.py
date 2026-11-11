# m11_peak_regulome Amendment 6: six-TF entry-by-entry comparison on the shared 79-TF re-call (read only: complex_screen/pair_motif_composite/recall/). Diagnostic, benchmark already read.
# usage: m11_shared.py -> shared_stats.tsv, shared_deciles.tsv, shared_files.tsv, recall_shared/<TF>_aligned.tsv, <TF>_deciles.png
import os
import numpy as np, pandas as pd
from scipy.stats import rankdata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m11_peak_regulome"; SH = f"{X}/complex_screen/pair_motif_composite"; SR = f"{SH}/recall"; OD = f"{O}/out/recall_shared"; os.makedirs(OD, exist_ok=True)
LAB = "diagnostic, benchmark already read"; rng = np.random.default_rng(20261111); NB = 1000
TFS = ["NRF1", "CTCF", "ZNF143", "MAZ", "NFYA", "E4F1"]; SWAP = {"NRF1": "ZNF143", "ZNF143": "NRF1", "CTCF": "E4F1", "E4F1": "CTCF", "MAZ": "NFYA", "NFYA": "MAZ"}
R = f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed"
E = pd.read_csv(f"{R}/Neighborhoods/EnhancerList.txt", sep="\t", usecols=["chr", "start", "end"]); E["key"] = E.chr + ":" + E.start.astype(str) + "-" + E.end.astype(str); NE = len(E); ek = dict(zip(E.key, range(NE)))
CH = {c: (g.start.values, g.end.values, g.index.values) for c, g in E.sort_values(["chr", "start"]).groupby("chr")}
def elem_of(c, p):
    r = np.full(len(p), -1)
    for ch in np.unique(c):
        if ch not in CH: continue
        st, en, ix = CH[ch]; m = c == ch; k = np.searchsorted(st, p[m], side="right") - 1; ok = (k >= 0) & (p[m] < en[np.clip(k, 0, None)]); e_ = np.full(m.sum(), -1); e_[ok] = ix[k[ok]]; r[m] = e_
    return r
# shared 1 kb bins (index = row of bins_qc / ref.npz)
Q = pd.read_csv(f"{SH}/ref/bins_qc.tsv", sep="\t", usecols=["chrom", "s", "e"]); Q["s1"] = Q.s - 400; Q["e1"] = Q.e + 400; F = np.load(f"{SR}/ref.npz"); HOT = F["hot"]; assert len(Q) == len(HOT)
QB = {c: (G.s1.values[np.argsort(G.s1.values)], G.e1.values[np.argsort(G.s1.values)], G.index.values[np.argsort(G.s1.values)]) for c, G in Q.groupby("chrom")}
def bin_of(c, p):
    r = np.full(len(p), -1)
    for ch in np.unique(c):
        if ch not in QB: continue
        st, en, ix = QB[ch]; m = c == ch; k = np.searchsorted(st, p[m], side="right") - 1; ok = (k >= 0) & (p[m] < en[np.clip(k, 0, None)]); b = np.full(m.sum(), -1); b[ok] = ix[k[ok]]; r[m] = b
    return r
L = pd.read_csv(sorted(f"{R}/Predictions/{f}" for f in os.listdir(f"{R}/Predictions") if f.startswith("EnhancerPredictionsFull_threshold0.02") and f.endswith(".tsv"))[0], sep="\t", usecols=["chr", "start", "end", "TargetGene", "ABC.Score", "isSelfPromoter"])
L["e"] = (L.chr + ":" + L.start.astype(str) + "-" + L.end.astype(str)).map(ek).astype(int); L["self"] = L.isSelfPromoter.astype(str).str.lower().eq("true")
GLf = pd.read_csv(f"{R}/Neighborhoods/GeneList.txt", sep="\t", usecols=["name", "Ensembl_ID"]).drop_duplicates("name", keep="last"); genes = np.array(sorted(GLf.name.unique())); gi = pd.Series(np.arange(len(genes)), index=genes)
L = L[L.TargetGene.isin(gi.index)]; L["g"] = L.TargetGene.map(gi).values
emid = ((E.start + E.end) // 2).values; ebin = bin_of(E.chr.values.astype(str), emid)
def P_from_elem(sig):
    s = sig[L.e.values]; w = s * L["ABC.Score"].values; tot = np.bincount(L.g.values, weights=w, minlength=len(genes)); pr = np.bincount(L.g.values, weights=w * L.self.values, minlength=len(genes))
    return pd.DataFrame(dict(P=tot, P_promoter=pr, P_distal=tot - pr, n_linked_bound=np.bincount(L.g.values, weights=(s > 0).astype(float), minlength=len(genes)).astype(int)), index=genes)
RAW = {"raw_lab": lambda t: f"/vast/som/xujie_lab/TAGATG-293T/bigwig/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", "raw_tf": lambda t: f"{SR}/peaks_tf/TF.{t}_peaks.narrowPeak", "raw_atac": lambda t: f"{SR}/peaks_atac/TF.{t}_peaks.narrowPeak"}
IMP = {}
for f in ("imputed_scores.npz", "imputed_scores_binom.npz"):
    if os.path.exists(f"{SR}/{f}"):
        z = np.load(f"{SR}/{f}", allow_pickle=True); tl = [str(x).upper() for x in z["tfs"]]
        for k in z.files:
            if k != "tfs": IMP[f"imp_{k}"] = (tl, z[k])
FILES = []; PV = {}
for t in TFS:
    for v, fn in RAW.items():
        f = fn(t); ok = os.path.exists(f) and os.path.getsize(f) > 0; FILES.append(dict(tf=t, version=v, file=f, exists=ok, format="narrowPeak (full)"))
        if not ok: continue
        P = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 6, 9]); P.columns = ["c", "s", "sig", "off"]; sm = (P.s + P.off).values; c = P.c.values.astype(str); el = elem_of(c, sm); hot = np.zeros(len(P), bool); b = bin_of(c, sm); hot[b >= 0] = HOT[b[b >= 0]]
        for h in (False, True):
            keep = (el >= 0) & (~hot if h else True); sig = np.zeros(NE); np.maximum.at(sig, el[keep], P.sig.values[keep]); PV[(t, v, h)] = P_from_elem(sig)
        FILES[-1].update(n_peaks=len(P), frac_hot=float(hot.mean()))
    for v, (tl, M) in IMP.items():
        ok = t in tl; FILES.append(dict(tf=t, version=v, file="per-bin score (104,738 union-ATAC 1 kb bins)", exists=ok, format="bin-level score, not peaks"))
        if not ok: continue
        s = M[tl.index(t)].astype(float)
        for h in (False, True):
            sc = np.where(HOT, 0, s) if h else s; sig = np.zeros(NE); m = ebin >= 0; sig[m] = np.clip(sc[ebin[m]], 0, None); PV[(t, v, h)] = P_from_elem(sig)
pd.DataFrame(FILES).to_csv(f"{O}/shared_files.tsv", sep="\t", index=False)
# knockdown z
ens2sym = dict(zip(GLf.Ensembl_ID.astype(str).str.split(".").str[0], GLf.name))
d = np.load(f"{RG}/m4_validation/kd_response/out_v2/arm2_all_cell/kd_response.npz", allow_pickle=True); q = pd.read_csv(f"{RG}/m4_validation/kd_response/out_v2/arm2_all_cell/guide_qc.tsv", sep="\t")
u = d["units"].tolist(); ui = {x: k for k, x in enumerate(u)}; keep = d["ntc_detect"] >= .1; g = pd.Series(d["genes"][keep].astype(str)); Zm = d["z"][:, keep].astype(float)
ens = g.str.match(r"^ENSG\d+"); g2 = g.where(~ens, g.str.split(".").str[0].map(ens2sym).fillna(g)); prim = q[(q.kept == 1) & q.kind.isin(["all", "construct"])]
def zvec(t): return pd.Series(np.nanmean(Zm[[ui[x] for x in prim.unit[prim.tf == t]]], axis=0), index=g2.values).pipe(lambda s: s[~s.index.duplicated(keep="last")])
ATAC = pd.read_csv(f"{O}/out/atac_x_abc_per_gene.tsv.gz", sep="\t").set_index("gene").atac_x_abc
def sp(a, b): m = np.isfinite(a) & np.isfinite(b); return np.nan if m.sum() < 10 or np.std(a[m]) == 0 or np.std(b[m]) == 0 else float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1])
def stats_for(x, z):
    az = np.abs(z); pos = x > 0
    def f(ix):
        xx, zz, aa, pp = x[ix], z[ix], az[ix], pos[ix]
        return [sp(xx, zz), sp(xx, aa), sp(np.where(pp, xx, np.nan), zz), sp(np.where(pp, xx, np.nan), aa), (np.nanmean(aa[pp]) - np.nanmean(aa[~pp])) if pp.any() and (~pp).any() else np.nan]
    names = ["spearman_P_z", "spearman_P_absz", "spearman_P_z_Ppos", "spearman_P_absz_Ppos", "meanabsz_Ppos_minus_P0"]; obs = f(np.arange(len(x))); B = np.array([f(rng.integers(0, len(x), len(x))) for _ in range(NB)]); out = {"n_Ppos": int(pos.sum())}
    for k, nm in enumerate(names): out[nm] = obs[k]; out[nm + "_lo"] = float(np.nanpercentile(B[:, k], 2.5)); out[nm + "_hi"] = float(np.nanpercentile(B[:, k], 97.5))
    return out
def deciles(x, z):
    az = np.abs(z); pos = x > 0; groups = [("P=0", ~pos)]; rows = []
    if pos.sum() >= 10:
        dq = pd.qcut(pd.Series(x[pos]).rank(method="first"), 10, labels=False).values; ix = np.flatnonzero(pos)
        for k in range(10): m = np.zeros(len(x), bool); m[ix[dq == k]] = True; groups.append((f"D{k + 1}", m))
    for nm, m in groups:
        a, zz = az[m & np.isfinite(az)], z[m & np.isfinite(z)]
        if not len(a): continue
        ba = a[rng.integers(0, len(a), (NB, len(a)))].mean(1); bz = zz[rng.integers(0, len(zz), (NB, len(zz)))].mean(1)
        rows.append(dict(group=nm, n=int(m.sum()), mean_absz=float(a.mean()), absz_lo=float(np.percentile(ba, 2.5)), absz_hi=float(np.percentile(ba, 97.5)), mean_z=float(zz.mean()), z_lo=float(np.percentile(bz, 2.5)), z_hi=float(np.percentile(bz, 97.5))))
    return pd.DataFrame(rows)
ST = []; DE = []
for t in TFS:
    z = zvec(t); gl = [x for x in genes if x in z.index]; zz = z.reindex(gl).values.copy()
    if t in gl: zz[gl.index(t)] = np.nan
    cols = dict(gene=gl)
    for (tt_, v, h), pv in PV.items():
        if tt_ != t: continue
        k = v + ("_noHOT" if h else ""); pv = pv.reindex(gl); cols[f"P_{k}"] = pv.P.values
        if not h: cols[f"P_{k}_promoter"] = pv.P_promoter.values; cols[f"P_{k}_distal"] = pv.P_distal.values
    cols["atac_x_abc"] = ATAC.reindex(gl).fillna(0).values
    for (tt_, v, h), pv in PV.items():
        if tt_ == SWAP[t] and not h: cols[f"swapped_{SWAP[t]}_P_{v}"] = pv.P.reindex(gl).values
    cols["z"] = zz; al = pd.DataFrame(cols); al.to_csv(f"{OD}/{t}_aligned.tsv", sep="\t", index=False, float_format="%.5g")
    preds = {c: al[c].values for c in al.columns if (c.startswith("P_") and not c.endswith(("_promoter", "_distal"))) or c.startswith("swapped_")}; preds["ctrl_atac_only"] = al.atac_x_abc.values
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for nm, x in preds.items():
        r = dict(label=LAB, tf=t, predictor=nm, n_genes=len(gl)); r.update(stats_for(x, zz)); ST.append(r); D = deciles(x, zz); D.insert(0, "predictor", nm); D.insert(0, "tf", t); D.insert(0, "label", LAB); DE.append(D)
        if nm.startswith("P_") and not nm.endswith("_noHOT") or nm == "ctrl_atac_only":
            xs = np.arange(len(D))
            for k, col, lo, hi in ((0, "mean_absz", "absz_lo", "absz_hi"), (1, "mean_z", "z_lo", "z_hi")):
                ax[k].errorbar(xs, D[col], yerr=[D[col] - D[lo], D[hi] - D[col]], marker="o", capsize=2, label=nm, ms=3); ax[k].set_xticks(xs); ax[k].set_xticklabels(D.group, rotation=45, fontsize=7)
    ax[0].set_ylabel("mean |z| (Arm 2)"); ax[1].set_ylabel("mean z (Arm 2)"); ax[1].axhline(0, color="grey", lw=.5); ax[0].legend(fontsize=6); fig.suptitle(f"{t}: genes by decile of peak score (P = 0 first); shared re-call versions; {LAB}", fontsize=8); fig.tight_layout(); fig.savefig(f"{OD}/{t}_deciles.png", dpi=150); plt.close(fig)
pd.DataFrame(ST).to_csv(f"{O}/shared_stats.tsv", sep="\t", index=False, float_format="%.4f"); pd.concat(DE).to_csv(f"{O}/shared_deciles.tsv", sep="\t", index=False, float_format="%.4f"); print("done")
