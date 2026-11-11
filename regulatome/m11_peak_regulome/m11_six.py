# m11_peak_regulome Amendment 2: entry-wise comparison for six TFs (diagnostic, benchmark already read).
# usage: m11_six.py -> six_stats.tsv, six_deciles.tsv, six/<TF>_aligned.tsv, six/<TF>_deciles.png
import os
import numpy as np, pandas as pd
from scipy.stats import rankdata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m11_peak_regulome"; OUT = f"{O}/out"; SIX = f"{OUT}/six"; os.makedirs(SIX, exist_ok=True)
LAB = "diagnostic, benchmark already read"; SEED = 20261111; rng = np.random.default_rng(SEED); NB = 1000
TFS = ["NRF1", "CTCF", "ZNF143", "MAZ", "NFYA", "E4F1"]; SWAP = {"NRF1": "ZNF143", "ZNF143": "NRF1", "CTCF": "E4F1", "E4F1": "CTCF", "MAZ": "NFYA", "NFYA": "MAZ"}
GL = pd.read_csv(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/GeneList.txt", sep="\t", usecols=["name", "Ensembl_ID"]).drop_duplicates("name", keep="last"); ens2sym = dict(zip(GL.Ensembl_ID.astype(str).str.split(".").str[0], GL.name))
d = np.load(f"{RG}/m4_validation/kd_response/out_v2/arm2_all_cell/kd_response.npz", allow_pickle=True); q = pd.read_csv(f"{RG}/m4_validation/kd_response/out_v2/arm2_all_cell/guide_qc.tsv", sep="\t")
u = d["units"].tolist(); ui = {x: k for k, x in enumerate(u)}; keep = d["ntc_detect"] >= .1; g = pd.Series(d["genes"][keep].astype(str)); Zm = d["z"][:, keep].astype(float)
ens = g.str.match(r"^ENSG\d+"); g2 = g.where(~ens, g.str.split(".").str[0].map(ens2sym).fillna(g)); REC = int((ens & (g2 != g)).sum())
prim = q[(q.kept == 1) & q.kind.isin(["all", "construct"])]
def zvec(t): return pd.Series(np.nanmean(Zm[[ui[x] for x in prim.unit[prim.tf == t]]], axis=0), index=g2.values).pipe(lambda s: s[~s.index.duplicated(keep="last")])
ATAC = pd.read_csv(f"{OUT}/atac_x_abc_per_gene.tsv.gz", sep="\t").set_index("gene")
W = {(v, p): pd.read_csv(f"{OUT}/wide_{p}_{v}.tsv.gz", sep="\t", index_col=0).loc[TFS] for v in ("as_called_p0.01", "q0.05") for p in ("score", "promoter", "distal")}
NPK = {}
for chunk in pd.read_csv(f"{OUT}/long_as_called_p0.01.tsv.gz", sep="\t", usecols=["tf", "gene", "n_linked_with_peak"], chunksize=2_000_000):
    c = chunk[chunk.tf.isin(TFS)]
    for t, H in c.groupby("tf"): NPK.setdefault(t, []).append(H.set_index("gene").n_linked_with_peak)
NPK = {t: pd.concat(v) for t, v in NPK.items()}
def sp(a, b): m = np.isfinite(a) & np.isfinite(b); return np.nan if m.sum() < 10 or np.std(a[m]) == 0 or np.std(b[m]) == 0 else float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1])
def stats_for(x, z):
    az = np.abs(z); pos = x > 0; out = {}
    def f(ix):
        xx, zz, aa, pp = x[ix], z[ix], az[ix], pos[ix]
        return [sp(xx, zz), sp(xx, aa), sp(np.where(pp, xx, np.nan), zz), sp(np.where(pp, xx, np.nan), aa), (np.nanmean(aa[pp]) - np.nanmean(aa[~pp])) if pp.any() and (~pp).any() else np.nan]
    names = ["spearman_P_z", "spearman_P_absz", "spearman_P_z_Ppos", "spearman_P_absz_Ppos", "meanabsz_Ppos_minus_P0"]; obs = f(np.arange(len(x)))
    B = np.array([f(rng.integers(0, len(x), len(x))) for _ in range(NB)])
    for k, nm in enumerate(names): out[nm] = obs[k]; out[nm + "_lo"] = float(np.nanpercentile(B[:, k], 2.5)); out[nm + "_hi"] = float(np.nanpercentile(B[:, k], 97.5))
    out["n_Ppos"] = int(pos.sum()); return out
def deciles(x, z):
    az = np.abs(z); rows = []; pos = x > 0
    groups = [("P=0", ~pos)]
    if pos.sum() >= 10:
        dq = pd.qcut(pd.Series(x[pos]).rank(method="first"), 10, labels=False).values; ix = np.flatnonzero(pos)
        for k in range(10): m = np.zeros(len(x), bool); m[ix[dq == k]] = True; groups.append((f"D{k + 1}", m))
    for nm, m in groups:
        a, zz = az[m & np.isfinite(az)], z[m & np.isfinite(z)]
        if len(a) == 0: continue
        ba = a[rng.integers(0, len(a), (NB, len(a)))].mean(1); bz = zz[rng.integers(0, len(zz), (NB, len(zz)))].mean(1)
        rows.append(dict(group=nm, n=int(m.sum()), P_min=float(x[m].min()), P_max=float(x[m].max()), mean_absz=float(a.mean()), absz_lo=float(np.percentile(ba, 2.5)), absz_hi=float(np.percentile(ba, 97.5)), mean_z=float(zz.mean()), z_lo=float(np.percentile(bz, 2.5)), z_hi=float(np.percentile(bz, 97.5))))
    return pd.DataFrame(rows)
ST = []; DE = []
for t in TFS:
    z = zvec(t); genes = [x for x in W[("as_called_p0.01", "score")].columns if x in z.index]; zz = z.reindex(genes).values.copy()
    if t in genes: zz[genes.index(t)] = np.nan
    al = pd.DataFrame(dict(gene=genes, P_as_called=W[("as_called_p0.01", "score")].loc[t, genes].values, P_promoter=W[("as_called_p0.01", "promoter")].loc[t, genes].values, P_distal=W[("as_called_p0.01", "distal")].loc[t, genes].values,
                           P_q0_05=W[("q0.05", "score")].loc[t, genes].values, n_linked_elements_with_peak=NPK[t].reindex(genes).fillna(0).astype(int).values, atac_x_abc=ATAC.atac_x_abc.reindex(genes).fillna(0).values,
                           P_swapped=W[("as_called_p0.01", "score")].loc[SWAP[t], genes].values, z=zz))
    al.to_csv(f"{SIX}/{t}_aligned.tsv", sep="\t", index=False, float_format="%.5g")
    preds = {"P_as_called": al.P_as_called.values, "P_q0.05": al.P_q0_05.values, "P_promoter": al.P_promoter.values, "P_distal": al.P_distal.values, "ctrl_atac_only": al.atac_x_abc.values, f"ctrl_swapped_{SWAP[t]}": al.P_swapped.values}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for nm, x in preds.items():
        r = dict(label=LAB, tf=t, predictor=nm, n_genes=len(genes)); r.update(stats_for(x, zz)); ST.append(r)
        D = deciles(x, zz); D.insert(0, "predictor", nm); D.insert(0, "tf", t); D.insert(0, "label", LAB); DE.append(D)
        if nm in ("P_as_called", "ctrl_atac_only") or nm.startswith("ctrl_swapped"):
            xs = np.arange(len(D))
            for k, col, lo, hi in ((0, "mean_absz", "absz_lo", "absz_hi"), (1, "mean_z", "z_lo", "z_hi")):
                ax[k].errorbar(xs, D[col], yerr=[D[col] - D[lo], D[hi] - D[col]], marker="o", capsize=2, label=nm); ax[k].set_xticks(xs); ax[k].set_xticklabels(D.group, rotation=45, fontsize=7)
    ax[0].set_ylabel("mean |z| (Arm 2)"); ax[1].set_ylabel("mean z (Arm 2)"); ax[1].axhline(0, color="grey", lw=.5); ax[0].legend(fontsize=7); fig.suptitle(f"{t}: genes by decile of peak score (P = 0 group first); {LAB}", fontsize=9); fig.tight_layout(); fig.savefig(f"{SIX}/{t}_deciles.png", dpi=150); plt.close(fig)
pd.DataFrame(ST).to_csv(f"{O}/six_stats.tsv", sep="\t", index=False, float_format="%.4f"); pd.concat(DE).to_csv(f"{O}/six_deciles.tsv", sep="\t", index=False, float_format="%.4f")
open(f"{O}/six_join.txt", "w").write(f"Arm 2 knockdown genes >= 10% NTC detection: {len(g)}; Ensembl-ID-named: {int(ens.sum())}; recovered to a GeneList symbol: {REC}\n"); print("done")
