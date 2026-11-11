# m11_peak_regulome Amendment 3: evaluate control-called peaks for six TFs (diagnostic, benchmark already read).
# Amendment 4: variants V0 (full other-78 control), V1 (other-78 downsampled to 3x), V2 (OmniATAC 3x) and the lab q <= 0.05 set; HOT-flagged peaks; P with and without HOT-flagged peaks.
# usage: m11_recall_eval.py -> recall_peak_qc.tsv, recall_stats.tsv, recall_deciles.tsv, recall/six/<TF>_aligned.tsv, <TF>_deciles.png
import os, subprocess
import numpy as np, pandas as pd
from scipy.stats import rankdata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m11_peak_regulome"; RC = f"{O}/recall"; SIXD = f"{RC}/six"; os.makedirs(SIXD, exist_ok=True)
LAB = "diagnostic, benchmark already read"; SEED = 20261111; rng = np.random.default_rng(SEED); NB = 1000; QMIN = -np.log10(0.05)
FIMO = "/nas/longleaf/rhel9/apps/meme/5.5.7/bin/fimo"; BEDTOOLS = "/nas/longleaf/rhel9/apps/bedtools/2.31.1/bedtools2/bin/bedtools"; FA = f"{X}/downstream/cache/hg38.fa"
TFS = ["NRF1", "CTCF", "ZNF143", "MAZ", "NFYA", "E4F1"]; SWAP = {"NRF1": "ZNF143", "ZNF143": "NRF1", "CTCF": "E4F1", "E4F1": "CTCF", "MAZ": "NFYA", "NFYA": "MAZ"}
R = f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed"
E = pd.read_csv(f"{R}/Neighborhoods/EnhancerList.txt", sep="\t"); acol = [c for c in E.columns if c.startswith("ATAC.") and c.endswith(".RPM")][0]; E = E[["chr", "start", "end", acol]].rename(columns={acol: "atac"})
E["key"] = E.chr + ":" + E.start.astype(str) + "-" + E.end.astype(str); NE = len(E); ek = dict(zip(E.key, range(NE)))
CH = {c: (g.start.values, g.end.values, g.index.values) for c, g in E.sort_values(["chr", "start"]).groupby("chr")}
GLf = pd.read_csv(f"{R}/Neighborhoods/GeneList.txt", sep="\t", usecols=["chr", "name", "tss", "Ensembl_ID"]).dropna(subset=["tss"]).drop_duplicates("name", keep="last"); TSS = {c: np.sort(G.tss.values.astype(int)) for c, G in GLf.groupby("chr")}
UAdf = pd.read_csv(f"{X}/data/HEK293T_ATAC_union3.bed.gz", sep="\t", header=None, usecols=[0, 1, 2]); UAdf.columns = ["c", "s", "e"]
UA = {}
for c, G in UAdf.groupby("c"):
    iv = G[["s", "e"]].values; iv = iv[np.argsort(iv[:, 0])]; ms = []
    for s, e in iv:
        if ms and s <= ms[-1][1]: ms[-1][1] = max(ms[-1][1], e)
        else: ms.append([s, e])
    UA[c] = np.array(ms)
def in_iv(c, p):
    r = np.zeros(len(p), bool)
    for ch in np.unique(c):
        if ch not in UA: continue
        m = c == ch; iv = UA[ch]; k = np.searchsorted(iv[:, 0], p[m], side="right") - 1; ok = k >= 0; ok[ok] = p[m][ok] < iv[k[ok], 1]; r[m] = ok
    return r
def elem_of(c, p):
    r = np.full(len(p), -1)
    for ch in np.unique(c):
        if ch not in CH: continue
        st, en, ix = CH[ch]; m = c == ch; k = np.searchsorted(st, p[m], side="right") - 1; ok = (k >= 0) & (p[m] < en[np.clip(k, 0, None)]); e_ = np.full(m.sum(), -1); e_[ok] = ix[k[ok]]; r[m] = e_
    return r
def near(cA, pA, cB, pB, d):  # fraction of A within d bp of any B
    r = np.zeros(len(pA), bool)
    for ch in np.unique(cA):
        m = cA == ch; b = np.sort(pB[cB == ch])
        if not len(b): continue
        k = np.searchsorted(b, pA[m]); dd = np.minimum(np.abs(pA[m] - b[np.clip(k - 1, 0, len(b) - 1)]), np.abs(pA[m] - b[np.clip(k, 0, len(b) - 1)])); r[m] = dd <= d
    return r
def near_tss(c, p): return near(c, p, np.concatenate([np.full(len(v), k) for k, v in TSS.items()]), np.concatenate(list(TSS.values())), 1000)
def read_np(f):
    P = pd.read_csv(f, sep="\t", header=None, usecols=list(range(10))); P.columns = ["chr", "start", "end", "name", "score", "strand", "signal", "p", "q", "off"]; P["summit"] = P.start + P.off; return P
def motif_frac(t, c, p, tag):
    mf = f"{O}/../m10_promoter_regulome/work/g/motif_{t}.meme"
    if not os.path.exists(mf) or not len(p): return np.nan
    bed = f"{RC}/tmp_{t}_{tag}.bed"; fa = bed[:-4] + ".fa"
    pd.DataFrame(dict(c=c, s=np.maximum(p - 50, 0), e=p + 51, n=[f"s{i}" for i in range(len(p))])).to_csv(bed, sep="\t", header=False, index=False)
    subprocess.run(f"{BEDTOOLS} getfasta -fi {FA} -bed {bed} -nameOnly -fo {fa}", shell=True, check=True)
    out = subprocess.run(f"{FIMO} --thresh 1e-4 --text --bfile --uniform-- {mf} {fa} 2>/dev/null | awk -F'\\t' 'NR>1{{print $3}}' | sort -u | wc -l", shell=True, check=True, capture_output=True, text=True).stdout
    os.remove(bed); os.remove(fa); return int(out.strip()) / len(p)
# background: 10,000 positions uniform in union ATAC
lens = np.concatenate([iv[:, 1] - iv[:, 0] for iv in UA.values()]); chs = np.concatenate([np.full(len(iv), c) for c, iv in UA.items()]); sts = np.concatenate([iv[:, 0] for iv in UA.values()])
pick = rng.choice(len(lens), 10000, p=lens / lens.sum()); bg_c = chs[pick]; bg_p = sts[pick] + (rng.random(10000) * lens[pick]).astype(int)
HB = pd.read_csv(f"{RC}/hot_bins.bed", sep="\t", header=None, names=["c", "s", "e", "bin", "k"]); HOTIV = {c: G[["s", "e"]].values[np.argsort(G.s.values)] for c, G in HB.groupby("c")}
def hot(c, p):
    r = np.zeros(len(p), bool)
    for ch in np.unique(c):
        if ch not in HOTIV: continue
        m = c == ch; iv = HOTIV[ch]; k = np.searchsorted(iv[:, 0], p[m], side="right") - 1; ok = k >= 0; ok[ok] = p[m][ok] < iv[k[ok], 1]; r[m] = ok
    return r
VAR = {"V0_other78_full": lambda t: f"{RC}/{t}/TF.{t}_peaks.narrowPeak", "V1_other78_3x": lambda t: f"{RC}/V1/{t}/TF.{t}_peaks.narrowPeak", "V2_omniatac_3x": lambda t: f"{RC}/V2/{t}/TF.{t}_peaks.narrowPeak"}
QC = []; REC = {}
CT = pd.read_csv(f"{X}/downstream/tracks/bulk_chipseq/CTCF_peaks.bed", sep="\t", header=None); chip_c, chip_p = CT[0].values, (CT[1] + CT[9]).values
for t in TFS:
    Lb = read_np(f"/vast/som/xujie_lab/TAGATG-293T/bigwig/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"); Lb = Lb[Lb.q >= QMIN]; sets = {k: read_np(f(t)) for k, f in VAR.items()}; sets["lab_q0.05"] = Lb
    for nm, D in sets.items():
        D["hot"] = hot(D.chr.values.astype(str), D.summit.values); REC[(t, nm)] = D; c, p = D.chr.values.astype(str), D.summit.values
        rd = f"{RC}/{nm.split('_')[0]}/{t}/reads.tsv" if nm.startswith(("V1", "V2")) else None
        r = dict(label=LAB, tf=t, peak_set=nm, n_peaks=len(D), median_width=float((D.end - D.start).median()), frac_in_union_atac=float(in_iv(c, p).mean()), frac_summit_in_abc_element=float((elem_of(c, p) >= 0).mean()), frac_within_1kb_tss=float(near_tss(c, p).mean()),
                 frac_hot_flagged=float(D.hot.mean()), frac_own_motif_within_50bp=motif_frac(t, c, p, nm), frac_within_100bp_of_lab_q005=float(near(c, p, Lb.chr.values.astype(str), Lb.summit.values, 100).mean()))
        if rd and os.path.exists(rd): x = pd.read_csv(rd, sep="\t").iloc[0]; r.update(treatment_reads=int(x.treatment_reads), control_pool_reads=int(x.control_pool_reads), control_reads_kept=int(x.control_reads_kept))
        if t == "CTCF": r["frac_within_100bp_of_ENCODE_HEK293_ChIP_summit"] = float(near(c, p, chip_c, chip_p, 100).mean()); r["frac_ENCODE_HEK293_ChIP_summits_within_100bp_of_set"] = float(near(chip_c, chip_p, c, p, 100).mean())
        QC.append(r)
    QC.append(dict(label=LAB, tf=t, peak_set="background_10000_union_ATAC_positions", n_peaks=10000, frac_in_union_atac=1.0, frac_summit_in_abc_element=float((elem_of(bg_c, bg_p) >= 0).mean()), frac_within_1kb_tss=float(near_tss(bg_c, bg_p).mean()), frac_hot_flagged=float(hot(bg_c, bg_p).mean()), frac_own_motif_within_50bp=motif_frac(t, bg_c, bg_p, "bg")))
    print(t, flush=True)
pd.DataFrame(QC).to_csv(f"{O}/recall_peak_qc.tsv", sep="\t", index=False, float_format="%.4f")
# ---- P from re-called peaks, then the Amendment 2 comparison
L = pd.read_csv(sorted(f"{R}/Predictions/{f}" for f in os.listdir(f"{R}/Predictions") if f.startswith("EnhancerPredictionsFull_threshold0.02") and f.endswith(".tsv"))[0], sep="\t", usecols=["chr", "start", "end", "TargetGene", "ABC.Score", "isSelfPromoter"])
L["e"] = (L.chr + ":" + L.start.astype(str) + "-" + L.end.astype(str)).map(ek).astype(int); L["self"] = L.isSelfPromoter.astype(str).str.lower().eq("true"); genes = np.array(sorted(GLf.name.unique())); gi = pd.Series(np.arange(len(genes)), index=genes); L = L[L.TargetGene.isin(gi.index)]; L["g"] = L.TargetGene.map(gi).values
def pvec(D):
    el = elem_of(D.chr.values.astype(str), D.summit.values); sig = np.zeros(NE); np.maximum.at(sig, el[el >= 0], D.signal.values[el >= 0]); s = sig[L.e.values]; w = s * L["ABC.Score"].values
    tot = np.bincount(L.g.values, weights=w, minlength=len(genes)); pr = np.bincount(L.g.values, weights=w * L.self.values, minlength=len(genes)); nb = np.bincount(L.g.values, weights=(s > 0).astype(float), minlength=len(genes))
    return pd.DataFrame(dict(P=tot, P_promoter=pr, P_distal=tot - pr, n_linked_elements_with_peak=nb.astype(int)), index=genes)
PV = {(t, nm, h): pvec(D[~D.hot] if h else D) for (t, nm), D in REC.items() for h in (False, True)}
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
LABP = pd.read_csv(f"{O}/out/wide_score_as_called_p0.01.tsv.gz", sep="\t", index_col=0).loc[TFS]; LABQ = pd.read_csv(f"{O}/out/wide_score_q0.05.tsv.gz", sep="\t", index_col=0).loc[TFS]
ST = []; DE = []
for t in TFS:
    z = zvec(t); gl = [x for x in genes if x in z.index]; zz = z.reindex(gl).values.copy()
    if t in gl: zz[gl.index(t)] = np.nan
    cols = dict(gene=gl)
    for (tt_, nm, h), pv in PV.items():
        if tt_ != t: continue
        k = nm + ("_noHOT" if h else ""); pv = pv.reindex(gl); cols[f"P_{k}"] = pv.P.values
        if not h: cols[f"P_{k}_promoter"] = pv.P_promoter.values; cols[f"P_{k}_distal"] = pv.P_distal.values; cols[f"nlinkedpeak_{k}"] = pv.n_linked_elements_with_peak.values
    cols.update(P_lab_as_called=LABP.loc[t, gl].values, atac_x_abc=ATAC.reindex(gl).fillna(0).values, P_swapped_V1=PV[(SWAP[t], "V1_other78_3x", False)].P.reindex(gl).values, z=zz)
    al = pd.DataFrame(cols); al.to_csv(f"{SIXD}/{t}_aligned.tsv", sep="\t", index=False, float_format="%.5g")
    preds = {c: al[c].values for c in al.columns if c.startswith("P_") and not c.endswith(("_promoter", "_distal"))}; preds["ctrl_atac_only"] = al.atac_x_abc.values; preds[f"ctrl_swapped_{SWAP[t]}_V1"] = preds.pop("P_swapped_V1")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for nm, x in preds.items():
        r = dict(label=LAB, tf=t, predictor=nm, n_genes=len(gl)); r.update(stats_for(x, zz)); ST.append(r); D = deciles(x, zz); D.insert(0, "predictor", nm); D.insert(0, "tf", t); D.insert(0, "label", LAB); DE.append(D)
        if nm in ("P_V0_other78_full", "P_V1_other78_3x", "P_V2_omniatac_3x", "P_lab_q0.05", "ctrl_atac_only"):
            xs = np.arange(len(D))
            for k, col, lo, hi in ((0, "mean_absz", "absz_lo", "absz_hi"), (1, "mean_z", "z_lo", "z_hi")):
                ax[k].errorbar(xs, D[col], yerr=[D[col] - D[lo], D[hi] - D[col]], marker="o", capsize=2, label=nm); ax[k].set_xticks(xs); ax[k].set_xticklabels(D.group, rotation=45, fontsize=7)
    ax[0].set_ylabel("mean |z| (Arm 2)"); ax[1].set_ylabel("mean z (Arm 2)"); ax[1].axhline(0, color="grey", lw=.5); ax[0].legend(fontsize=6); fig.suptitle(f"{t}: genes by decile of peak score (P = 0 first); control-called vs lab peaks; {LAB}", fontsize=8); fig.tight_layout(); fig.savefig(f"{SIXD}/{t}_deciles.png", dpi=150); plt.close(fig)
pd.DataFrame(ST).to_csv(f"{O}/recall_stats.tsv", sep="\t", index=False, float_format="%.4f"); pd.concat(DE).to_csv(f"{O}/recall_deciles.tsv", sep="\t", index=False, float_format="%.4f"); print("done")
