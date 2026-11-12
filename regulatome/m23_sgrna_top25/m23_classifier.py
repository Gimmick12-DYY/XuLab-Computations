# m23: the m16 strong-phenotype classifier (same settings) rerun with knockdown cells restricted to a selection (top25_own or random25_d0); NTC cells unchanged.
# usage: m23_classifier.py <selection column>  -> out/classifier_<selection>/
# m16_strong_phenotype part A (SPEC.md): reproduce the X-Atlas/Orion binary classifier on our streamed HEK293T cells, then apply the authors' strong-phenotype criteria.
# usage: m16_classifier.py -> out/classifier_per_gene.tsv, out/ntc_negative_control.tsv, out/strong_phenotype_per_tf.tsv, out/criteria_counts.tsv
import glob, os, zlib
import numpy as np, pandas as pd, scipy.sparse as sp, pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
X_ = "/work/users/d/y/dyy12/XuLab"; RG = f"{X_}/regulatome"; import sys; SELCOL = sys.argv[1]; O = f"{RG}/m16_strong_phenotype"; OUT = f"{RG}/m23_sgrna_top25/out/classifier_{SELCOL}"; os.makedirs(OUT, exist_ok=True); A = f"{X_}/data/x_atlas_orion"; LAB = "diagnostic, benchmark already read"
SEED = 20261111; rng = np.random.default_rng(SEED); NHVG = 2000; NPERM = 200; MINC = 100; NJ = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}
files = sorted(glob.glob(f"{A}/panel_ntc_cells/*.parquet")); gm = pd.read_parquet(f"{A}/hf_sample/gene_metadata.parquet"); G = int(gm.gene_token_id.max()) + 1; sym = np.array(["."] * G, dtype=object); sym[gm.gene_token_id.values] = gm.gene_name.values
# pass 1: metadata
meta = []
for fi, f in enumerate(files):
    t = pq.read_table(f, columns=["cell_barcode", "sample", "gene_target", "pass_guide_filter"]).to_pandas(); t["file"] = fi; t["row"] = np.arange(len(t)); meta.append(t)
M = pd.concat(meta, ignore_index=True); M = M[M.pass_guide_filter == 1].reset_index(drop=True)
SEL = pd.read_csv(f"{RG}/m23_sgrna_top25/out/matrices/cell_selection.tsv.gz", sep="\t", usecols=["cell_barcode", "gem", SELCOL]); keepset = set(zip(SEL.gem[SEL[SELCOL]], SEL.cell_barcode[SEL[SELCOL]]))
isntc = M.gene_target == "Non-Targeting"; M = M[isntc | pd.Series([(f_, b_) in keepset for f_, b_ in zip(M.file, M.cell_barcode)])].reset_index(drop=True); print("cells after selection", len(M), flush=True); ntc = M.gene_target == "Non-Targeting"
panel = pd.read_csv(f"{RG}/phase1/panel_tf_xatlas_status.tsv", sep="\t"); genes_kd = sorted(set(M.gene_target[~ntc]) & set(panel.gene_symbol))
print("cells", len(M), "ntc", int(ntc.sum()), "kd genes", len(genes_kd), flush=True)
# NTC selection: GEM-matched (same sample) 1:1 per knocked-down gene; 20,000 for HVG; negative-control sets
NI = M.index[ntc].values; by_s = {s: g.index.values for s, g in M[ntc].groupby("sample")}; sel = {}; need = set()
for gname in genes_kd:
    P = M.index[(M.gene_target == gname).values].values; m = []
    for s, n in M.loc[P, "sample"].value_counts().items():
        pool = by_s.get(s, NI); m += list(rng.choice(pool, n, replace=len(pool) < n))
    sel[gname] = (P, np.array(m)); need |= set(P) | set(m)
hvg_cells = rng.choice(NI, 20000, replace=False); need |= set(hvg_cells); negsets = []
for k in range(20):
    a = rng.choice(NI, 600, replace=False); negsets.append((a[:300], a[300:])); need |= set(a)
need = np.array(sorted(need)); pos = {c: i for i, c in enumerate(need)}
# pass 2: expression for the needed cells -> CSR (cells x genes), log1p CP10k
rows, cols, vals = [], [], []; tot = np.zeros(len(need))
for fi, f in enumerate(files):
    sub = M.loc[need[M.loc[need, "file"].values == fi]]
    if not len(sub): continue
    t = pq.read_table(f, columns=["gene_token_id", "gene_expression", "total_counts"]).take(sub.row.values)
    ids = t.column("gene_token_id").to_pylist(); ex = t.column("gene_expression").to_pylist(); tc = t.column("total_counts").to_numpy()
    for k, c in enumerate(sub.index.values):
        i = pos[c]; rows.append(np.full(len(ids[k]), i)); cols.append(np.asarray(ids[k])); vals.append(np.asarray(ex[k])); tot[i] = tc[k]
    print("file", fi, flush=True)
C = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(need), G))
L = sp.diags(1e4 / np.maximum(tot, 1)) @ C; L = L.tocsr(); L.data = np.log1p(L.data)
# HVG (seurat_v3-like on raw counts of 20,000 NTC cells): standardised variance from a binned mean-variance trend
H = C[[pos[c] for c in hvg_cells]]; mu = np.asarray(H.mean(0)).ravel(); var = np.asarray(H.multiply(H).mean(0)).ravel() - mu ** 2; ok = mu > 0
lm, lv = np.log10(mu[ok]), np.log10(np.maximum(var[ok], 1e-12)); b = pd.qcut(lm, 50, labels=False, duplicates="drop"); trend = pd.Series(lv).groupby(b).transform("mean").values
score = np.full(G, -np.inf); score[np.flatnonzero(ok)] = lv - trend; HV = np.argsort(-score)[:NHVG + 10]
tok = {g: i for i, g in enumerate(sym)}
def run(gname, P, N, ablate):
    feats = np.array([j for j in HV if sym[j] not in ablate])[:NHVG]; Xc = L[[pos[c] for c in np.concatenate([P, N])]][:, feats].toarray(); y = np.r_[np.ones(len(P)), np.zeros(len(N))]
    r = np.random.default_rng(zlib.crc32(gname.encode()) + SEED); idx = r.permutation(len(y)); te = np.zeros(len(y), bool)
    for cl in (0, 1): ii = idx[y[idx] == cl]; te[ii[:int(round(len(ii) * .3))]] = True
    tr = ~te; mu_, sd_ = Xc[tr].mean(0), Xc[tr].std(0); sd_[sd_ == 0] = 1; Z = (Xc - mu_) / sd_
    def fit_auc(ytr):
        m = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=2000).fit(Z[tr], ytr); return roc_auc_score(y[te], m.decision_function(Z[te]))
    obs = fit_auc(y[tr]); null = np.array([fit_auc(r.permutation(y[tr])) for _ in range(NPERM)])
    tgt = [tok[g] for g in ablate if g in tok]; ntc_train_nonzero = bool(np.asarray(C[[pos[c] for c in np.concatenate([P, N])[tr][y[tr] == 0]]][:, tgt].sum()).ravel().sum() > 0) if tgt else None
    return dict(name=gname, n_cells=len(P), auc=float(obs), null_mean=float(null.mean()), null_p97_5=float(np.percentile(null, 97.5)), p=float((1 + (null >= obs).sum()) / (NPERM + 1)), target_nonzero_in_ntc_train=ntc_train_nonzero)
jobs = [(g, *sel[g], [g]) for g in genes_kd if len(sel[g][0]) >= MINC]
R = Parallel(n_jobs=NJ)(delayed(run)(*j) for j in jobs); R = pd.DataFrame(R)
from scipy.stats import false_discovery_control
R["q"] = false_discovery_control(R.p.values) if len(R) else []
small = [dict(name=g, n_cells=len(sel[g][0]), auc=np.nan, p=np.nan, q=np.nan, note=f"not assessable (< {MINC} cells)") for g in genes_kd if len(sel[g][0]) < MINC]
R = pd.concat([R, pd.DataFrame(small)], ignore_index=True); R.insert(0, "label", LAB); R.to_csv(f"{OUT}/classifier_per_gene.tsv", sep="\t", index=False, float_format="%.4g")
NEG = Parallel(n_jobs=NJ)(delayed(run)(f"NTCset{k}", a, b, []) for k, (a, b) in enumerate(negsets)); NEG = pd.DataFrame(NEG); NEG["q"] = false_discovery_control(NEG.p.values); NEG.to_csv(f"{OUT}/ntc_negative_control.tsv", sep="\t", index=False, float_format="%.4g")
print("negative control: NTC sets with p < 0.05:", int((NEG.p < .05).sum()), "of", len(NEG), "; q < 0.05:", int((NEG.q < .05).sum()), flush=True)
# authors' four criteria per knocked-down gene, then panel TF
K = panel.set_index("gene_symbol"); R["knockdown_pct"] = R.name.map(K.knockdown_pct); R["panel_tf"] = R.name.map(lambda g: ALIAS.get(g, g))
R["c1_ge25_cells"] = R.n_cells >= 25; R["c2_ge30pct_kd"] = R.knockdown_pct >= 30; R["c3_classifier_q05"] = R.q < .05; R["c4_target_nonzero_ntc_train"] = R.target_nonzero_in_ntc_train.fillna(False).astype(bool)
R["strong_phenotype"] = R.c1_ge25_cells & R.c2_ge30pct_kd & R.c3_classifier_q05 & R.c4_target_nonzero_ntc_train
R.to_csv(f"{OUT}/classifier_per_gene.tsv", sep="\t", index=False, float_format="%.4g")
FL = pd.read_csv(f"{RG}/m6_improve/repro_v1/per_tf_floor_test.tsv", sep="\t").set_index("tf"); FT = pd.read_csv(f"{RG}/m7_step2/splits/tf_folds.tsv", sep="\t").set_index("TF")
TEST = set(pd.read_csv(f"{RG}/m10_promoter_regulome/F1_testable.tsv", sep="\t").query("arm == 'arm2'").tf)
T = R.groupby("panel_tf").agg(genes=("name", ",".join), n_cells=("n_cells", "sum"), knockdown_pct=("knockdown_pct", "max"), auc=("auc", "max"), q=("q", "min"), c1=("c1_ge25_cells", "any"), c2=("c2_ge30pct_kd", "any"), c3=("c3_classifier_q05", "any"), c4=("c4_target_nonzero_ntc_train", "any"), strong_phenotype=("strong_phenotype", "any")).reset_index()
tfs79 = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]; T = pd.DataFrame(dict(panel_tf=tfs79)).merge(T, on="panel_tf", how="left")
T["kd_ceiling"] = T.panel_tf.map(FT.ceiling); T["benchmark_testable_m10"] = T.panel_tf.isin(TEST); T["depth_tertile"] = T.panel_tf.map(FL.tertile); T["hitag_cells"] = T.panel_tf.map(FL.cells); T["claim_tier_d_ge_0.04"] = T.panel_tf.map(FL.d) >= .04
T["status"] = np.where(T.genes.isna(), "no knockdown unit in streamed data", np.where(T.auc.isna(), "not assessable (< 100 cells)", np.where(T.strong_phenotype.fillna(False).astype(bool), "strong phenotype", "fails")))
T.insert(0, "label", LAB); T.to_csv(f"{OUT}/strong_phenotype_per_tf.tsv", sep="\t", index=False, float_format="%.4g")
S = T[T.genes.notna()]; ST = S.strong_phenotype.fillna(False).astype(bool)
CT = [("TFs with a knockdown in the streamed cells", len(S)), ("c1 >= 25 cells", int(S.c1.fillna(False).sum())), ("c2 >= 30% knockdown", int(S.c2.fillna(False).sum())), ("assessable (>= 100 cells)", int(S.auc.notna().sum())), ("c3 classifier q < 0.05", int(S.c3.fillna(False).sum())),
      ("c4 target non-zero in NTC training", int(S.c4.fillna(False).sum())), ("strong phenotype (all four)", int(ST.sum())), ("strong & ceiling >= 0.2", int((ST & (S.kd_ceiling >= .2)).sum())), ("ceiling >= 0.2", int((S.kd_ceiling >= .2).sum())), ("strong & m10 testable", int((ST & S.benchmark_testable_m10).sum())), ("m10 testable", int(S.benchmark_testable_m10.sum()))]
pd.DataFrame(CT, columns=["criterion", "n_tfs"]).to_csv(f"{OUT}/criteria_counts.tsv", sep="\t", index=False); print(pd.DataFrame(CT).to_string(index=False)); print("strong:", ",".join(S.panel_tf[ST]))
