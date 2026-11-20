# m51: re-stream the m23 inputs; per GEM NTC log1p(CP10k) mean / variance (all NTC cells), a random sample of up to 100 NTC cells per GEM and all m23 top-25% knockdown cells (log1p CP10k rows).
import glob, os
import numpy as np, pandas as pd, pyarrow.parquet as pq, scipy.sparse as sp
A = "/work/users/d/y/dyy12/XuLab/data/x_atlas_orion"; M23 = "/work/users/d/y/dyy12/XuLab/regulatome/m23_sgrna_top25/out/matrices"; OUT = "/work/users/d/y/dyy12/XuLab/regulatome/m51_edistance/out"; os.makedirs(OUT, exist_ok=True)
gm = pd.read_parquet(f"{A}/hf_sample/gene_metadata.parquet"); G = int(gm.gene_token_id.max()) + 1; sym = np.array(["."] * G, dtype=object); sym[gm.gene_token_id.values] = gm.gene_name.values
fs = sorted(glob.glob(f"{A}/panel_ntc_cells/*.parquet")); rng = np.random.default_rng(20261120)
CS = pd.read_csv(f"{M23}/cell_selection.tsv.gz", sep="\t"); CS = CS[CS.top25_own]; sel = {(int(g), b): (t, gt) for g, b, t, gt in zip(CS.gem, CS.cell_barcode, CS.gene_target, CS.guide_target)}
Ca = pd.read_csv(f"{A}/HEK293T_filtered_guide_calls_per_cell.csv.gz", usecols=["cell_barcode", "num_umis"]); sgu = pd.Series(Ca.num_umis.astype(str).str.split("|").map(lambda v: sum(float(x) for x in v)).values, index=Ca.cell_barcode.values); del Ca
NM, NV, NN, ND, NC_, NCo, KD, KDo, NLIN = [], [], [], [], [], [], [], [], []
for gi, f in enumerate(fs):
    t = pq.read_table(f, columns=["gene_token_id", "gene_expression", "cell_barcode", "gene_target", "guide_target", "total_counts", "pass_guide_filter"]); o = t.select(["cell_barcode", "gene_target", "guide_target", "total_counts", "pass_guide_filter"]).to_pandas()
    a = t["gene_token_id"].combine_chunks(); b = t["gene_expression"].combine_chunks(); ptr = a.offsets.to_numpy(); j = a.values.to_numpy()[ptr[0]:ptr[-1]]; x = b.values.to_numpy().astype(np.float32)[ptr[0]:ptr[-1]]; ptr = ptr - ptr[0]
    X_ = sp.csr_matrix((x, j.astype(np.int32), ptr.astype(np.int64)), shape=(len(o), G)); sg = sgu.reindex(o.cell_barcode.values).values; ok = (o.pass_guide_filter.values == 1) & np.isfinite(sg); ntc = ok & (o.gene_target.values == "Non-Targeting")
    C_ = (sp.diags(1e4 / o.total_counts.values.astype(np.float32)) @ X_).tocsr(); L_ = C_.copy(); L_.data = np.log1p(L_.data)
    Ln = L_[ntc]; m = np.asarray(Ln.mean(0)).ravel(); m2 = np.asarray(Ln.multiply(Ln).mean(0)).ravel(); n = Ln.shape[0]
    NM.append(m.astype(np.float32)); NV.append((np.maximum(m2 - m ** 2, 0) * n / max(n - 1, 1)).astype(np.float32)); NN.append(n); ND.append(np.asarray((Ln > 0).sum(0)).ravel()); NLIN.append(np.asarray(C_[ntc].mean(0)).ravel().astype(np.float32))
    ix = np.flatnonzero(ntc); pick = np.sort(rng.choice(ix, min(100, len(ix)), replace=False)); NC_.append(L_[pick]); NCo.append(pd.DataFrame({"gem": gi, "cell_barcode": o.cell_barcode.values[pick]}))
    rows, lab = [], []
    for r in np.flatnonzero(ok & ~ntc):
        k = sel.get((gi, o.cell_barcode.iat[r]))
        if k is not None: rows.append(r); lab.append(k)
    if rows: KD.append(L_[rows]); KDo.append(pd.DataFrame({"gem": gi, "cell_barcode": o.cell_barcode.values[rows], "gene_target": [k[0] for k in lab], "guide_target": [k[1] for k in lab]}))
    print(os.path.basename(f), n, len(rows), flush=True)
sp.save_npz(f"{OUT}/ntc_sample.npz", sp.vstack(NC_).tocsr()); pd.concat(NCo, ignore_index=True).to_csv(f"{OUT}/ntc_sample_cells.tsv.gz", sep="\t", index=False)
sp.save_npz(f"{OUT}/kd_cells.npz", sp.vstack(KD).tocsr()); pd.concat(KDo, ignore_index=True).to_csv(f"{OUT}/kd_cells.tsv.gz", sep="\t", index=False)
np.savez_compressed(f"{OUT}/ntc_gem_stats.npz", mean=np.vstack(NM), var=np.vstack(NV), n=np.array(NN), detected=np.vstack(ND), mean_cp10k=np.vstack(NLIN), genes=sym.astype(str)); print("done", sum(NN), flush=True)
