# m22 Amendment 5: one overall correlation over all TFs, Regulome score vs Perturb-seq, canonical ABC vs simple ABC. Tables only. Diagnostic.
import json, os
import numpy as np, pandas as pd
from scipy.stats import rankdata
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m22_edge_table"; OUT = f"{O}/out"; V = "/vast/som/xujie_lab/TAGATG-293T/bigwig"; rng = np.random.default_rng(20261112); LAB = "diagnostic"
tfs79 = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]; ti = {t: i for i, t in enumerate(tfs79)}
Rz = np.load(f"{RG}/m9_abc_diagnosis/work/responses_arm2.npz", allow_pickle=True); UNIV = Rz["genes"].astype(str); gi = {g: k for k, g in enumerate(UNIV)}; NG = len(UNIV); KT = list(Rz["tfs"].astype(str))
PK = {}
for t in tfs79:
    f = f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"
    if not os.path.exists(f) or os.path.getsize(f) == 0: continue
    P = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 6, 9], names=["c", "s", "sig", "off"]); P = P[P.sig.rank(ascending=False, method="first") <= 5000]; PK[t] = (P.c.values.astype(str), (P.s + P.off).values, P.sig.values)
def build(linkfile, atac_col, el_file=None):
    cols = ["chr", "start", "end", "TargetGene", "ABC.Score"] + ([atac_col] if el_file is None else [])
    L = pd.read_csv(linkfile, sep="\t", usecols=cols); L = L[L.TargetGene.isin(gi)].reset_index(drop=True)
    E = L[["chr", "start", "end"]].drop_duplicates().reset_index(drop=True)
    if el_file is not None:
        EL = pd.read_csv(el_file, sep="\t"); ac = [c for c in EL.columns if c.startswith("ATAC.") and c.endswith(".RPM")][0]; E = E.merge(EL[["chr", "start", "end", ac]].rename(columns={ac: "atac"}), how="left")
    else: E = E.merge(L.groupby(["chr", "start", "end"], as_index=False)[atac_col].first().rename(columns={atac_col: "atac"}), how="left")
    E["e"] = np.arange(len(E)); L = L.merge(E[["chr", "start", "end", "e"]]); Le, Lg, La = L.e.values, L.TargetGene.map(gi).values, L["ABC.Score"].values.astype(float)
    CHI = {c: (g.start.values, g.end.values, g.e.values) for c, g in E.groupby("chr")}; PE = np.zeros((len(E), 79))
    for t, (c, p, s) in PK.items():
        for ch in np.unique(c):
            if ch not in CHI: continue
            st, en, ix = CHI[ch]; o = np.argsort(st); st, en, ix = st[o], en[o], ix[o]; m = c == ch; k = np.searchsorted(st, p[m], side="right") - 1; ok = (k >= 0) & (p[m] < en[np.maximum(k, 0)]); np.maximum.at(PE[:, ti[t]], ix[k[ok]], s[m][ok])
    S = np.vstack([np.bincount(Lg, weights=PE[Le, k] * La, minlength=NG) for k in range(79)]); A = np.bincount(Lg, weights=E.atac.fillna(0).values[Le] * La, minlength=NG)
    print(os.path.basename(os.path.dirname(os.path.dirname(linkfile))), "links", len(L), "elements", len(E), "edges", int((S > 0).sum()), flush=True); return S, A, len(L)
SETS = {"canonical ABC, Hi-C (current Regulome)": build(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Predictions/EnhancerPredictionsFull_threshold0.02_self_promoter.tsv", None, f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/EnhancerList.txt"),
        "simple ABC (m2_abc: ATAC only, power-law)": build(f"{RG}/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", "normalized_atac_enh")}
FL = pd.read_csv(f"{RG}/m6_improve/repro_v1/per_tf_floor_test.tsv", sep="\t").set_index("tf"); OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
T = [t for t in KT if t in ti]; Z = Rz["r1"].astype(float)[[KT.index(t) for t in T]].copy()
for i, t in enumerate(T):
    for o in OWN.get(t, [t]):
        if o in gi: Z[i, gi[o]] = np.nan
tert = np.array([str(FL.tertile.get(t, "na")) for t in T]); groups = {k: np.flatnonzero(tert == k) for k in np.unique(tert)}
def sp(a, b): m = np.isfinite(a) & np.isfinite(b); return float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1])
def pe(a, b): m = np.isfinite(a) & np.isfinite(b); return float(np.corrcoef(a[m], b[m])[0, 1])
def rowstd(M): mu = np.nanmean(M, 1, keepdims=True); sd = np.nanstd(M, 1, keepdims=True); sd[sd == 0] = np.nan; return (M - mu) / sd
def rowpct(M):
    out = np.full(M.shape, np.nan)
    for i in range(len(M)):
        m = np.isfinite(M[i]); out[i, m] = rankdata(M[i, m]) / m.sum() if m.sum() else np.nan
    return out
ROWS = []
for nm, (S79, AT, nl) in SETS.items():
    S = np.vstack([S79[ti[t]] for t in T]); S = np.where(np.isfinite(Z), S, np.nan); A = np.where(np.isfinite(Z), np.tile(AT, (len(T), 1)), np.nan)
    for target, Y in (("|z|", np.abs(Z)), ("signed z", Z)):
        for way, f in (("1 raw pooled", lambda M: M), ("2 row-standardised", rowstd), ("3 within-TF percentile", rowpct)):
            Sx, Yx = f(S), f(Y); obs = sp(Sx.ravel(), Yx.ravel()); null = []
            for _ in range(500):
                pr = np.arange(len(T))
                for k, ix in groups.items(): pr[ix] = rng.permutation(ix)
                null.append(sp(f(np.where(np.isfinite(Z), S[pr], np.nan)).ravel(), Yx.ravel()))
            bt = [sp(Sx[b].ravel(), Yx[b].ravel()) for b in (rng.integers(0, len(T), len(T)) for _ in range(2000))]
            ROWS.append(dict(label=LAB, abc_model=nm, n_links=nl, target=target, way=way, n_tfs=len(T), n_entries=int((np.isfinite(Sx) & np.isfinite(Yx)).sum()), frac_entries_with_edge=float(np.nanmean(S > 0)),
                             spearman=obs, tf_boot_lo=np.percentile(bt, 2.5), tf_boot_hi=np.percentile(bt, 97.5), pearson=pe(Sx.ravel(), Yx.ravel()), swapped_tf_mean=float(np.mean(null)), swapped_tf_95pct=float(np.percentile(null, 95)),
                             swapped_p=(1 + int(np.sum(np.array(null) >= obs))) / 501, atac_only_spearman=sp(f(A).ravel(), Yx.ravel())))
            print(nm, target, way, round(obs, 4), flush=True)
R = pd.DataFrame(ROWS); R.to_csv(f"{OUT}/overall_correlation_simple_vs_canonical_abc.tsv", sep="\t", index=False, float_format="%.4g"); print(R.drop(columns="label").round(4).to_string(index=False)); print("done")
