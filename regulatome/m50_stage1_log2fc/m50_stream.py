# m50: re-stream the m23 inputs (as m49) for the top-25% cells and save mean CP10k per knockdown unit and its GEM-weighted NTC reference, plus the log1p difference check.
import glob, os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq, scipy.sparse as sp
A = "/work/users/d/y/dyy12/XuLab/data/x_atlas_orion"; M23 = "/work/users/d/y/dyy12/XuLab/regulatome/m23_sgrna_top25/out/matrices"; OUT = "/work/users/d/y/dyy12/XuLab/regulatome/m50_stage1_log2fc/out"; os.makedirs(OUT, exist_ok=True)
gm = pd.read_parquet(f"{A}/hf_sample/gene_metadata.parquet"); G = int(gm.gene_token_id.max()) + 1; sym = np.array(["."] * G, dtype=object); sym[gm.gene_token_id.values] = gm.gene_name.values
gs = pd.read_csv("/work/users/d/y/dyy12/XuLab/regulatome/phase1/guide_stats.tsv", sep="\t"); dual = set(gs[gs.n_guide_pairs == 2].gene_symbol)
fs = sorted(glob.glob(f"{A}/panel_ntc_cells/*.parquet"))
CS = pd.read_csv(f"{M23}/cell_selection.tsv.gz", sep="\t"); CS = CS[CS.top25_own]; sel = {(int(g), b): (t, gt) for g, b, t, gt in zip(CS.gem, CS.cell_barcode, CS.gene_target, CS.guide_target)}
Q = pd.read_csv(f"{M23}/top25_own/guide_qc.tsv", sep="\t"); Q = Q[(Q.kept == 1) & Q.kind.isin(["all", "construct"])]
# unit key: (gene_target, guide_target or '.') as in m23 (dual TFs: one unit per guide_target, constructs numbered by sorted guide_target)
Ca = pd.read_csv(f"{A}/HEK293T_filtered_guide_calls_per_cell.csv.gz", usecols=["cell_barcode", "num_umis"]); sgu = pd.Series(Ca.num_umis.astype(str).str.split("|").map(lambda v: sum(float(x) for x in v)).values, index=Ca.cell_barcode.values); del Ca
NL, NG_, NN, Det, SL, SG, CNT, UN = [], [], [], [], {}, {}, {}, None
for gi, f in enumerate(fs):
    t = pq.read_table(f, columns=["gene_token_id", "gene_expression", "cell_barcode", "gene_target", "guide_target", "total_counts", "pass_guide_filter"]); o = t.select(["cell_barcode", "gene_target", "guide_target", "total_counts", "pass_guide_filter"]).to_pandas()
    a = t["gene_token_id"].combine_chunks(); b = t["gene_expression"].combine_chunks(); ptr = a.offsets.to_numpy(); j = a.values.to_numpy()[ptr[0]:ptr[-1]]; x = b.values.to_numpy().astype(np.float32)[ptr[0]:ptr[-1]]; ptr = ptr - ptr[0]
    X_ = sp.csr_matrix((x, j.astype(np.int32), ptr.astype(np.int64)), shape=(len(o), G)); sg = sgu.reindex(o.cell_barcode.values).values; ok = (o.pass_guide_filter.values == 1) & np.isfinite(sg); ntc = ok & (o.gene_target.values == "Non-Targeting")
    C_ = (sp.diags(1e4 / o.total_counts.values.astype(np.float32)) @ X_).tocsr(); L_ = C_.copy(); L_.data = np.log1p(L_.data)
    NL.append(np.asarray(L_[ntc].mean(0)).ravel()); NG_.append(np.asarray(C_[ntc].mean(0)).ravel()); NN.append(int(ntc.sum())); Det.append(np.asarray((L_[ntc] > 0).sum(0)).ravel())
    keep = ok & ~ntc
    for r in np.flatnonzero(keep):
        k = sel.get((gi, o.cell_barcode.iat[r]))
        if k is None: continue
        u = (k[0], k[1] if k[0] in dual else ".")
        if u not in SL: SL[u] = np.zeros(G); SG[u] = np.zeros(G); CNT[u] = np.zeros(len(fs))
        SL[u] += L_[r].toarray().ravel(); SG[u] += C_[r].toarray().ravel(); CNT[u][gi] += 1
    print(os.path.basename(f), flush=True)
NL, NG_, NN = np.vstack(NL), np.vstack(NG_), np.array(NN, float); det = np.sum(Det, 0) / NN.sum(); cg = np.flatnonzero((det >= .01) & (sym != "."))
d = np.load(f"{M23}/top25_own/kd_response.npz", allow_pickle=True); assert (d["genes"].astype(str) == sym[cg].astype(str)).all()
units = d["units"].astype(str); ui = {u: i for i, u in enumerate(units)}; rows, LOGD, LOGB = [], [], []
for g in sorted({k[0] for k in SL}):
    ks = sorted(k for k in SL if k[0] == g)
    for c, k in enumerate(ks):
        name = f"{g}|construct{c + 1}" if g in dual else f"{g}|all"
        if name not in ui: continue
        ng = CNT[k]; n = ng.sum(); dl = SL[k] / n - (ng[:, None] * NL).sum(0) / n; kb = SG[k] / n; rb = (ng[:, None] * NG_).sum(0) / n
        LOGD.append(kb[cg]); LOGB.append(rb[cg]); rows.append((name, int(n), float(np.nanmax(np.abs(dl[cg] - d["logfc"][ui[name]])))))
R = pd.DataFrame(rows, columns=["unit", "cells", "max abs diff vs stored logfc"]); print(R.to_string(index=False)); assert R["max abs diff vs stored logfc"].max() < 1e-3
np.savez_compressed(f"{OUT}/unit_mean_cp10k.npz", units=R.unit.values.astype(str), genes=sym[cg].astype(str), kd_cp10k=np.vstack(LOGD), ntc_cp10k=np.vstack(LOGB), cells=R.cells.values); print("done")
