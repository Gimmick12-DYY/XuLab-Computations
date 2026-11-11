# m11_peak_regulome sections 1-3 (SPEC.md): peak inventory, peak-only TF x gene tables, NRF1/CTCF worked examples. Diagnostic / development, benchmark already read.
# usage: m11_build.py inventory | build | examples
import glob, gzip, os, sys
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m11_peak_regulome"; OUT = f"{O}/out"; os.makedirs(OUT, exist_ok=True); LAB = "diagnostic / development, benchmark already read"
V = "/vast/som/xujie_lab/TAGATG-293T"; tfs79 = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]; QMIN = -np.log10(0.05)
R = f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed"
E = pd.read_csv(f"{R}/Neighborhoods/EnhancerList.txt", sep="\t"); acol = [c for c in E.columns if c.startswith("ATAC.") and c.endswith(".RPM")][0]; E = E[["chr", "start", "end", acol]].rename(columns={acol: "atac"})
E["key"] = E.chr + ":" + E.start.astype(str) + "-" + E.end.astype(str); NE = len(E); ek = dict(zip(E.key, range(NE)))
GLf = pd.read_csv(f"{R}/Neighborhoods/GeneList.txt", sep="\t", usecols=["chr", "name", "tss"]).dropna().drop_duplicates("name", keep="last")
TSS = {c: np.sort(G.tss.values.astype(int)) for c, G in GLf.groupby("chr")}
CH = {c: (g.start.values, g.end.values, g.index.values) for c, g in E.sort_values(["chr", "start"]).groupby("chr")}
def merged(df):
    out = {}
    for c, G in df.groupby(df.columns[0]):
        iv = G.iloc[:, 1:3].values; iv = iv[np.argsort(iv[:, 0])]; ms = []
        for s, e in iv:
            if ms and s <= ms[-1][1]: ms[-1][1] = max(ms[-1][1], e)
            else: ms.append([s, e])
        out[c] = np.array(ms)
    return out
UA = merged(pd.read_csv(f"{X}/data/HEK293T_ATAC_union3.bed.gz", sep="\t", header=None, usecols=[0, 1, 2]))
def in_iv(IV, c, p):
    r = np.zeros(len(p), bool)
    for ch in np.unique(c):
        if ch not in IV: continue
        m = c == ch; iv = IV[ch]; k = np.searchsorted(iv[:, 0], p[m], side="right") - 1; ok = k >= 0; ok[ok] = p[m][ok] < iv[k[ok], 1]; r[m] = ok
    return r
def elem_of(c, p):
    r = np.full(len(p), -1)
    for ch in np.unique(c):
        if ch not in CH: continue
        st, en, ix = CH[ch]; m = c == ch; k = np.searchsorted(st, p[m], side="right") - 1; ok = (k >= 0) & (p[m] < en[np.clip(k, 0, None)]); e_ = np.full(m.sum(), -1); e_[ok] = ix[k[ok]]; r[m] = e_
    return r
def near_tss(c, p):
    r = np.zeros(len(p), bool)
    for ch in np.unique(c):
        t = TSS.get(ch)
        if t is None: continue
        m = c == ch; k = np.searchsorted(t, p[m]); d = np.minimum(np.abs(p[m] - t[np.clip(k - 1, 0, len(t) - 1)]), np.abs(p[m] - t[np.clip(k, 0, len(t) - 1)])); r[m] = d <= 1000
    return r
def read_np(f):
    if os.path.getsize(f) == 0: return pd.DataFrame(columns=["chr", "start", "end", "name", "score", "strand", "signal", "p", "q", "off", "summit"]).astype({"start": int, "end": int, "q": float, "signal": float, "summit": int})
    P = pd.read_csv(f, sep="\t", header=None, usecols=list(range(10))); P.columns = ["chr", "start", "end", "name", "score", "strand", "signal", "p", "q", "off"]; P["summit"] = P.start + P.off; return P
mode = sys.argv[1]
if mode == "inventory":
    rows = []
    def add(setname, tf, path, P, pos, qcol=None, caller=""):
        c = P.chr.values.astype(str); r = dict(label=LAB, set=setname, tf=tf, path=path, caller=caller, n_as_called=len(P), n_q05=int((P[qcol] >= QMIN).sum()) if qcol else np.nan, median_width=float((P.end - P.start).median()) if len(P) else np.nan)
        if len(P): r.update(frac_within_1kb_tss=float(near_tss(c, pos).mean()), frac_in_canonical_element=float((elem_of(c, pos) >= 0).mean()), frac_in_union_atac=float(in_iv(UA, c, pos).mean()))
        r["flag"] = "few (<100)" if len(P) < 100 else ("many (>50,000)" if len(P) > 50000 else ""); rows.append(r)
    mac = "MACS2 2.2.9.1 callpeak -f BED -g hs -p 0.01 --nomodel --shift -75 --extsize 150 --SPMR --keep-dup all --call-summits, no control (from .xls header)"
    for t in tfs79:
        f = f"{V}/bigwig/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"
        if os.path.exists(f): P = read_np(f); add("a_lab_macs2_as_called_p0.01", t, f, P, P.summit.values, "q", mac)
        f = f"{V}/bigwig/peaks/TF.{t}_peaks.q005.narrowPeak"
        if os.path.exists(f): P = read_np(f); add("b_lab_q005", t, f, P, P.summit.values, "q", "subset of (a) at q <= 0.05")
        f = f"{V}/TFevent/{t}_sig_bins_fdr0.2.bed"
        if os.path.exists(f) and os.path.getsize(f) > 0: P = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2]); P.columns = ["chr", "start", "end"]; add("c_TFevent_bins_fdr0.2", t, f, P, ((P.start + P.end) // 2).values, None, "1 kb bin permutation test, FDR 0.2")
        for setname, pat, cl in (("d_imputed_peaks", f"{X}/motif_analysis/motif_imputed/{t.lower()}/{t.lower()}.imputed_peaks.bed", "MACS3 q < 0.05 on imputed pseudobulk (call_imputed_peaks.py)"), ("e_pmat_peaks", f"{X}/motif_analysis/motif_pmat/{t.lower()}/{t.lower()}.pmat_peaks.bed", "pmat peaks (motif_analysis)")):
            if os.path.exists(pat) and os.path.getsize(pat) > 0:
                P = pd.read_csv(pat, sep="\t", header=None); P = P.rename(columns={0: "chr", 1: "start", 2: "end"}); pos = ((P.start + P.end) // 2).values
                if P.shape[1] >= 10 and pd.api.types.is_numeric_dtype(P[9]): pos = (P.start + P[9]).values
                add(setname, t, pat, P, pos, None, cl)
    extra = sorted(set(os.listdir(f"{V}/bigwig/peaks")) - {f"TF.{t}_peaks.q005.narrowPeak" for t in tfs79})
    I = pd.DataFrame(rows); I.to_csv(f"{O}/inventory_per_tf.tsv", sep="\t", index=False, float_format="%.4g")
    S = I.groupby("set").agg(n_tfs=("tf", "count"), median_n=("n_as_called", "median"), min_n=("n_as_called", "min"), max_n=("n_as_called", "max"), median_width=("median_width", "median"), frac_tss=("frac_within_1kb_tss", "median"), frac_canon=("frac_in_canonical_element", "median"), frac_atac=("frac_in_union_atac", "median")).reset_index()
    S.to_csv(f"{O}/inventory_summary.tsv", sep="\t", index=False, float_format="%.4g"); print(S.to_string(index=False)); print("extra files in bigwig/peaks:", extra)
elif mode == "build":
    L = pd.read_csv(sorted(f"{R}/Predictions/{f}" for f in os.listdir(f"{R}/Predictions") if f.startswith("EnhancerPredictionsFull_threshold0.02") and f.endswith(".tsv"))[0], sep="\t", usecols=["chr", "start", "end", "TargetGene", "ABC.Score", "isSelfPromoter"])
    L["e"] = (L.chr + ":" + L.start.astype(str) + "-" + L.end.astype(str)).map(ek).astype(int); L["self"] = L.isSelfPromoter.astype(str).str.lower().eq("true")
    genes = np.array(sorted(GLf.name.unique())); gi = pd.Series(np.arange(len(genes)), index=genes); L = L[L.TargetGene.isin(gi.index)].copy(); L["g"] = L.TargetGene.map(gi).values; NG = len(genes)
    nlink = np.bincount(L.g.values, minlength=NG); W = {}; SUM = []; DROP = []
    for ver in ("as_called_p0.01", "q0.05"):
        longf = gzip.open(f"{OUT}/long_{ver}.tsv.gz", "wt"); first = True; Wm = np.zeros((len(tfs79), NG)); Wp = np.zeros((len(tfs79), NG)); Wd = np.zeros((len(tfs79), NG))
        for i, t in enumerate(tfs79):
            P = read_np(f"{V}/bigwig/macs2/TF.{t}/TF.{t}_peaks.narrowPeak")
            if ver == "q0.05": P = P[P.q >= QMIN]
            c = P.chr.values.astype(str); el = elem_of(c, P.summit.values); sig = np.zeros(NE); qq = np.zeros(NE); np.maximum.at(sig, el[el >= 0], P.signal.values[el >= 0]); np.maximum.at(qq, el[el >= 0], P.q.values[el >= 0])
            dr = el < 0; DROP.append(dict(label=LAB, version=ver, tf=t, n_peaks=len(P), n_summit_in_no_element=int(dr.sum()), frac_dropped=float(dr.mean()) if len(P) else np.nan, n_dropped_within_1kb_tss=int(near_tss(c[dr], P.summit.values[dr]).sum()) if dr.any() else 0))
            s = sig[L.e.values]; w = s * L["ABC.Score"].values; hp = s > 0
            tot = np.bincount(L.g.values, weights=w, minlength=NG); pr = np.bincount(L.g.values, weights=w * L.self.values, minlength=NG); di = tot - pr
            nb = np.bincount(L.g.values, weights=hp.astype(float), minlength=NG).astype(int); prom = np.bincount(L.g.values, weights=(hp & L.self.values).astype(float), minlength=NG) > 0
            Wm[i], Wp[i], Wd[i] = tot, pr, di
            pd.DataFrame(dict(tf=t, gene=genes, n_linked_elements=nlink, n_linked_with_peak=nb, promoter_peak=prom, score_signal_x_abc=tot, score_promoter=pr, score_distal=di, any_peak=nb > 0)).to_csv(longf, sep="\t", index=False, header=first, float_format="%.5g"); first = False
            SUM.append(dict(label=LAB, version=ver, tf=t, n_peaks=len(P), genes_nonzero=int((tot > 0).sum()), share_of_genelist=float((tot > 0).mean()), genes_promoter_peak=int(prom.sum()), median_score_nonzero=float(np.median(tot[tot > 0])) if (tot > 0).any() else np.nan))
        longf.close()
        for nm, M in (("score", Wm), ("promoter", Wp), ("distal", Wd)): pd.DataFrame(M, index=tfs79, columns=genes).to_csv(f"{OUT}/wide_{nm}_{ver}.tsv.gz", sep="\t", float_format="%.5g")
        print(ver, flush=True)
    pd.DataFrame(SUM).to_csv(f"{O}/per_tf_summary.tsv", sep="\t", index=False, float_format="%.4g"); pd.DataFrame(DROP).to_csv(f"{O}/dropped_peaks.tsv", sep="\t", index=False, float_format="%.4g")
    # ATAC-only matrix row (identical for every TF): per gene, sum over links of element ATAC.RPM x ABC.Score
    A = np.bincount(L.g.values, weights=E.atac.values[L.e.values] * L["ABC.Score"].values, minlength=NG); pd.DataFrame(dict(gene=genes, atac_x_abc=A, n_linked_elements=nlink)).to_csv(f"{OUT}/atac_x_abc_per_gene.tsv.gz", sep="\t", index=False, float_format="%.5g")
elif mode == "examples":
    L = pd.read_csv(sorted(f"{R}/Predictions/{f}" for f in os.listdir(f"{R}/Predictions") if f.startswith("EnhancerPredictionsFull_threshold0.02") and f.endswith(".tsv"))[0], sep="\t", usecols=["chr", "start", "end", "TargetGene", "ABC.Score", "isSelfPromoter"])
    L["e"] = (L.chr + ":" + L.start.astype(str) + "-" + L.end.astype(str)).map(ek).astype(int); L["self"] = L.isSelfPromoter.astype(str).str.lower().eq("true"); L = L[L.TargetGene.isin(set(GLf.name))]
    D = f"{OUT}/worked_example"; os.makedirs(D, exist_ok=True)
    for t in ("NRF1", "CTCF"):
        for ver in ("as_called_p0.01", "q0.05"):
            P = read_np(f"{V}/bigwig/macs2/TF.{t}/TF.{t}_peaks.narrowPeak")
            if ver == "q0.05": P = P[P.q >= QMIN]
            P["e"] = elem_of(P.chr.values.astype(str), P.summit.values); P = P[P.e >= 0].sort_values("signal", ascending=False).drop_duplicates("e")  # strongest peak per element
            X_ = L.merge(P[["e", "chr", "start", "end", "summit", "signal", "q"]].rename(columns={"chr": "peak_chr", "start": "peak_start", "end": "peak_end", "summit": "peak_summit", "signal": "peak_signalValue", "q": "peak_minus_log10_q"}), on="e", how="left")
            X_["link_score"] = X_.peak_signalValue.fillna(0) * X_["ABC.Score"]
            X_[["chr", "start", "end", "TargetGene", "ABC.Score", "self", "peak_chr", "peak_start", "peak_end", "peak_summit", "peak_signalValue", "peak_minus_log10_q", "link_score"]].rename(columns={"chr": "element_chr", "start": "element_start", "end": "element_end", "self": "is_self_promoter"}).to_csv(f"{D}/{t}_{ver}_links.tsv", sep="\t", index=False, float_format="%.5g")
            G = X_.groupby("TargetGene").agg(n_linked_elements=("e", "size"), n_linked_with_peak=("peak_signalValue", lambda v: int(v.notna().sum())), score_signal_x_abc=("link_score", "sum"))
            G["score_promoter"] = X_[X_.self].groupby("TargetGene").link_score.sum().reindex(G.index).fillna(0); G["score_distal"] = G.score_signal_x_abc - G.score_promoter
            G["promoter_peak"] = X_[X_.self & X_.peak_signalValue.notna()].TargetGene.drop_duplicates().pipe(lambda s: G.index.isin(s)); G.index.name = "gene"; G.to_csv(f"{D}/{t}_{ver}_genes.tsv", sep="\t", float_format="%.5g")
    print("examples done")
