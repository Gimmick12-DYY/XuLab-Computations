# m35 (SPEC.md): threshold sweep, ours vs bulk ChIP through the identical path. Descriptive.
import sys, gzip
import numpy as np, pandas as pd
RG_ = "/work/users/d/y/dyy12/XuLab/regulatome"
sys.argv = ["m15.py", "raw"]; src = open(f"{RG_}/m15_gene_to_tf/m15.py").read(); exec(src[:src.index("# ---- binding percentiles")])  # TF, UG, Z1, Z2, ZA
X35 = "/work/users/d/y/dyy12/XuLab"; R35 = f"{X35}/regulatome"; O35 = f"{R35}/m35_threshold_sweep/out"; LAB = "descriptive"
TFL = list(TF); gi = {g: k for k, g in enumerate(UG)}; ti = {t: i for i, t in enumerate(TFL)}; NG = len(UG); OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
for i, t in enumerate(TFL):
    for o in OWN.get(t, [t]):
        if o in gi: ZA[i, gi[o]] = Z1[i, gi[o]] = Z2[i, gi[o]] = np.nan
def resp(i, th):
    z, z1, z2 = ZA[i], Z1[i], Z2[i]; ok = np.isfinite(z); a = np.abs(np.nan_to_num(z))
    if th.startswith("abs"): return ok & (a >= int(th[3:]))
    if th.startswith("top"): m = np.zeros(NG, bool); m[np.argsort(-np.where(ok, a, -1))[:int(th[3:])]] = True; return m & ok
    if th.startswith("rep"): c = int(th[3:]); return ok & np.isfinite(z1) & np.isfinite(z2) & (np.abs(np.nan_to_num(z1)) >= c) & (np.abs(np.nan_to_num(z2)) >= c) & (np.sign(z1) == np.sign(z2))
    if th.startswith("down"): return ok & (np.nan_to_num(z) <= -int(th[4:]))
TH = ["abs2", "abs3", "abs4", "abs5", "top50", "top100", "top200", "rep2", "rep3", "down2", "down3", "down4"]
THL = {"abs2": "|z| >= 2", "abs3": "|z| >= 3", "abs4": "|z| >= 4", "abs5": "|z| >= 5", "top50": "top 50 by |z|", "top100": "top 100 by |z|", "top200": "top 200 by |z|", "rep2": "both halves |z| >= 2", "rep3": "both halves |z| >= 3", "down2": "down z <= -2", "down3": "down z <= -3", "down4": "down z <= -4"}
R = pd.read_csv(f"{R35}/m34_fig5_panels_v3/out/regulome_v1.tsv.gz", sep="\t", usecols=["tf", "gene"]); LISTS = []
for t, g in R.groupby("tf"):
    if t in ti: m = np.zeros(NG, bool); m[[gi[x] for x in g.gene if x in gi]] = True; LISTS.append(("our HITAG (HEK293T)", t, "HEK293T (ours)", m))
# bulk through the identical path (as m33)
NB = f"{R35}/m2c_abc_canonical/results/HEK293T_hicfixed"; EL = pd.read_csv(f"{NB}/Neighborhoods/EnhancerList.txt", sep="\t", usecols=["chr", "start", "end"]); ek = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(EL.chr, EL.start, EL.end))}
L = pd.read_csv(f"{NB}/Predictions/EnhancerPredictionsFull_threshold0.02_self_promoter.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene"]); L = L[L.TargetGene.isin(gi)]
Le = (L.chr + ":" + L.start.astype(str) + "-" + L.end.astype(str)).map(ek).values.astype(int); Lg = L.TargetGene.map(gi).values
CHI = {}
for ch, G in EL.groupby("chr"): o = np.argsort(G.start.values); CHI[ch] = (G.start.values[o], G.end.values[o], G.index.values[o])
def elem_of(c, p):
    r = np.full(len(p), -1)
    for ch in np.unique(c):
        if ch not in CHI: continue
        st, en, ix = CHI[ch]; m = c == ch; k = np.searchsorted(st, p[m], side="right") - 1; ok = (k >= 0) & (p[m] < en[np.maximum(k, 0)]); rr = np.full(m.sum(), -1); rr[ok] = ix[k[ok]]; r[m] = rr
    return r
def bulk_list(f):
    P = pd.read_csv(f, sep="\t", header=None, compression="gzip" if f.endswith(".gz") else None); P = P.iloc[:, [0, 1, 2, 6] + ([9] if P.shape[1] >= 10 else [])]; P.columns = ["c", "s", "e", "sig"] + (["off"] if P.shape[1] == 5 else [])
    if "off" not in P: P["off"] = (P.e - P.s) // 2
    P.loc[P.off < 0, "off"] = (P.e - P.s) // 2; P = P[P.sig.rank(ascending=False, method="first") <= 5000]
    e = elem_of(P.c.astype(str).values, (P.s + P.off).values); v = np.zeros(len(EL), bool); v[e[e >= 0]] = True; m = np.zeros(NG, bool); m[Lg[v[Le]]] = True; return m
D33 = f"{R35}/m33_abc_pipeline_audit"
CH = [("CTCF", "HEK293T GSE103651", f"{X35}/data/CTCF_HEK293T_GSE103651_peaks.bed", True), ("CTCF", "HEK293 ENCODE", f"{X35}/data/CTCF_HEK293_peaks.bed", False), ("MAZ", "HEK293", f"{X35}/data/MAZ_HEK293_peaks.bed", True),
      ("ZIC2", "HEK293", f"{X35}/data/ZIC2_HEK293_peaks.bed", True), ("ZNF282", "HEK293", f"{X35}/data/ZNF282_HEK293_peaks.bed", True), ("ZNF777", "HEK293", f"{X35}/data/ZNF777_HEK293_peaks.bed", True),
      ("ZBTB7A", "HEK293", f"{X35}/data/ZBTB7A_HEK293_peaks.bed", True), ("SETDB1", "HEK293", f"{X35}/data/SETDB1_HEK293_peaks.bed", True), ("NFYA", "K562", f"{X35}/data/NFYA_K562_peaks.bed", True),
      ("NRF1", "K562 ENCFF259YUE", f"{D33}/encode/ENCFF259YUE.bed.gz", True), ("NRF1", "HepG2 ENCFF348EDE", f"{D33}/encode/ENCFF348EDE.bed.gz", False), ("E4F1", "K562 ENCFF683WRK", f"{D33}/encode/ENCFF683WRK.bed.gz", True)]
PRIM = {}
for t, line, f, prim in CH:
    if t in ti: LISTS.append(("bulk ChIP", t, line, bulk_list(f))); PRIM[(t, line)] = prim
BT = sorted({t for t, *_ in CH})
ROWS = []
for src, t, line, m in LISTS:
    i = ti[t]; ok = np.isfinite(ZA[i]); N = int(ok.sum()); ref = 100 * (m & ok).sum() / N
    for th in TH:
        r = resp(i, th); n = int(r.sum()); k = int((r & m).sum()); rec = 100 * k / n if n else np.nan
        ROWS.append(dict(label=LAB, threshold=th, threshold_label=THL[th], source=src, TF=t, reference_cell_line=line, primary_reference=PRIM.get((t, line), True), bulk_tf=t in BT, n_responders=n, n_shared=k,
                         pct_recovered=rec, reference_pct=ref, fold=rec / ref if n and ref else np.nan, enough_responders=n >= 10))
T = pd.DataFrame(ROWS); T.to_csv(f"{O35}/per_tf_threshold_sweep.tsv", sep="\t", index=False, float_format="%.3f")
def pool(g, name):
    e = g[g.enough_responders]; ws = (g.n_responders * g.reference_pct).sum() / max(g.n_responders.sum(), 1); hi = e[e.fold > 1.2]
    return dict(label=LAB, set=name, threshold=g.threshold.iloc[0], threshold_label=g.threshold_label.iloc[0], n_tfs=len(g), n_tfs_ge10_responders=len(e), total_responders=int(g.n_responders.sum()), total_recovered=int(g.n_shared.sum()),
                pooled_pct_recovered=100 * g.n_shared.sum() / max(g.n_responders.sum(), 1), pooled_reference_pct=ws, pooled_fold=(100 * g.n_shared.sum() / max(g.n_responders.sum(), 1)) / ws if ws else np.nan,
                median_fold_tfs_ge10=float(e.fold.median()) if len(e) else np.nan, n_tfs_fold_gt_1_2=len(hi), tfs_fold_gt_1_2=",".join(hi.TF + ("" if (hi.source == "our HITAG (HEK293T)").all() else "")) if len(hi) else "")
P = []
for th in TH:
    g = T[T.threshold == th]
    P.append(pool(g[g.source == "our HITAG (HEK293T)"], "ours, all 75 TFs"))
    P.append(pool(g[(g.source == "our HITAG (HEK293T)") & g.bulk_tf], "ours, 10 TFs with bulk ChIP"))
    b = g[(g.source == "bulk ChIP") & g.primary_reference]; p = pool(b, "bulk ChIP, same 10 TFs (one reference each)"); p["tfs_fold_gt_1_2"] = ",".join(b[b.enough_responders & (b.fold > 1.2)].TF); P.append(p)
P = pd.DataFrame(P); P.to_csv(f"{O35}/pooled_threshold_sweep.tsv", sep="\t", index=False, float_format="%.3f")
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 80)
print(P.drop(columns=["label", "threshold"]).round(2).to_string(index=False))
FOC = ["NRF1", "E4F1", "CTCF", "NFYA", "MAZ", "DNMT1", "NME2"]
F = T[T.TF.isin(FOC)].copy(); F["name"] = np.where(F.source.str.startswith("our"), F.TF + " ours", F.TF + " bulk " + F.reference_cell_line)
print(F.pivot_table(index="name", columns="threshold", values="fold", aggfunc="first")[TH].round(2).to_string())
print(F.pivot_table(index="name", columns="threshold", values="n_responders", aggfunc="first")[TH].to_string())
print("done")
