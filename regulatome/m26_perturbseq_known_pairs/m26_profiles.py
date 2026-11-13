# m26 Part 1-2, definition (ii) inputs: per perturbation mean log1p CP10k (all cells, and two random halves), all-NTC mean and sd, NTC raw-count mean/var for HVG, own-target CP10k.
# Reads panel_ntc_cells and cohesin_cells (same batch names). Output: work/profiles_cells.npz. Diagnostic.
import glob, os
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, scipy.sparse as sp
A = "/work/users/d/y/dyy12/XuLab/data/x_atlas_orion"; O = "/work/users/d/y/dyy12/XuLab/regulatome/m26_perturbseq_known_pairs"; os.makedirs(f"{O}/work", exist_ok=True); rng = np.random.default_rng(20261113)
gm = pd.read_parquet(f"{A}/hf_sample/gene_metadata.parquet"); G = int(gm.gene_token_id.max()) + 1; sym = np.array(["."] * G, dtype=object); sym[gm.gene_token_id.values] = gm.gene_name.values
cols = ["gene_token_id", "gene_expression", "gene_target", "total_counts", "pass_guide_filter"]
S, S1, S2, N, N1, N2, OWN, OWNN = {}, {}, {}, {}, {}, {}, {}, {}
ntc_s = np.zeros(G); ntc_ss = np.zeros(G); ntc_n = 0; raw_s = np.zeros(G); raw_ss = np.zeros(G); ntc_cp = np.zeros(G)
tok = dict(zip(gm.gene_name, gm.gene_token_id))
for f in sorted(glob.glob(f"{A}/panel_ntc_cells/*.parquet")):
    t = pq.read_table(f, columns=cols); cf = f.replace("panel_ntc_cells", "cohesin_cells")
    if os.path.exists(cf): t = pa.concat_tables([t, pq.read_table(cf, columns=cols).cast(t.schema)])
    o = t.select(["gene_target", "total_counts", "pass_guide_filter"]).to_pandas(); ok = o.pass_guide_filter.values == 1
    a = t["gene_token_id"].combine_chunks(); b = t["gene_expression"].combine_chunks(); ptr = a.offsets.to_numpy(); j = a.values.to_numpy()[ptr[0]:ptr[-1]]; x = b.values.to_numpy().astype(np.float64)[ptr[0]:ptr[-1]]; ptr = ptr - ptr[0]
    C = sp.csr_matrix((x, j.astype(np.int64), ptr), shape=(len(o), G)); tc = np.maximum(o.total_counts.values.astype(float), 1); CP = sp.diags(1e4 / tc) @ C; Lg = CP.copy(); Lg.data = np.log1p(Lg.data)
    nt = ok & (o.gene_target.values == "Non-Targeting")
    if nt.any():
        Ln = Lg[nt]; ntc_s += np.asarray(Ln.sum(0)).ravel(); ntc_ss += np.asarray(Ln.multiply(Ln).sum(0)).ravel(); ntc_n += int(nt.sum()); Cn = C[nt]; raw_s += np.asarray(Cn.sum(0)).ravel(); raw_ss += np.asarray(Cn.multiply(Cn).sum(0)).ravel(); ntc_cp += np.asarray(CP[nt].sum(0)).ravel()
    for g in np.unique(o.gene_target.values[ok & ~nt]):
        ix = np.flatnonzero(ok & (o.gene_target.values == g)); h = rng.random(len(ix)) < .5
        for D_, Nd, sel in ((S, N, ix), (S1, N1, ix[h]), (S2, N2, ix[~h])):
            if len(sel): D_[g] = D_.get(g, 0) + np.asarray(Lg[sel].sum(0)).ravel(); Nd[g] = Nd.get(g, 0) + len(sel)
        if g in tok: OWN[g] = OWN.get(g, 0) + float(CP[ix][:, tok[g]].sum()); OWNN[g] = OWNN.get(g, 0) + len(ix)
    print(os.path.basename(f), len(o), flush=True)
P = sorted(S); mu = ntc_s / ntc_n; sd = np.sqrt(np.maximum(ntc_ss / ntc_n - mu ** 2, 0))
np.savez_compressed(f"{O}/work/profiles_cells.npz", perts=np.array(P), genes=sym, mean_all=np.vstack([S[p] / N[p] for p in P]), mean_h1=np.vstack([S1.get(p, np.zeros(G)) / max(N1.get(p, 0), 1) for p in P]), mean_h2=np.vstack([S2.get(p, np.zeros(G)) / max(N2.get(p, 0), 1) for p in P]),
                    n=np.array([N[p] for p in P]), n1=np.array([N1.get(p, 0) for p in P]), n2=np.array([N2.get(p, 0) for p in P]), ntc_mean=mu, ntc_sd=sd, ntc_n=ntc_n, raw_mean=raw_s / ntc_n, raw_var=raw_ss / ntc_n - (raw_s / ntc_n) ** 2,
                    ntc_cp10k=ntc_cp / ntc_n, own_cp10k=np.array([OWN.get(p, np.nan) / max(OWNN.get(p, 1), 1) for p in P]))
print("done", len(P), ntc_n)
