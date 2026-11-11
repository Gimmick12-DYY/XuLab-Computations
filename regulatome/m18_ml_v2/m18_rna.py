# m18_ml_v2 block R (SPEC Amendment 1.3 / 2): NTC pseudo-bulk in two random halves (seed 20261112): mean log1p CP10k and detection per gene.
import glob
import numpy as np, pandas as pd, pyarrow.parquet as pq
X = "/work/users/d/y/dyy12/XuLab"; A = f"{X}/data/x_atlas_orion"; O = f"{X}/regulatome/m18_ml_v2/features"
import os; os.makedirs(O, exist_ok=True)
gm = pd.read_parquet(f"{A}/hf_sample/gene_metadata.parquet"); G = int(gm.gene_token_id.max()) + 1; sym = np.array(["."] * G, dtype=object); sym[gm.gene_token_id.values] = gm.gene_name.values
rng = np.random.default_rng(20261112); S = {h: np.zeros(G) for h in (0, 1)}; D = {h: np.zeros(G) for h in (0, 1)}; N = {0: 0, 1: 0}
for f in sorted(glob.glob(f"{A}/panel_ntc_cells/*.parquet")):
    pf = pq.ParquetFile(f)
    for b in pf.iter_batches(batch_size=20000, columns=["gene_target", "pass_guide_filter", "gene_token_id", "gene_expression", "total_counts"]):
        t = b.to_pydict(); keep = [i for i in range(len(t["gene_target"])) if t["gene_target"][i] == "Non-Targeting" and t["pass_guide_filter"][i] == 1]
        half = rng.integers(0, 2, len(t["gene_target"]))
        for i in keep:
            h = int(half[i]); ids = np.asarray(t["gene_token_id"][i]); ex = np.asarray(t["gene_expression"][i], float); tc = max(float(t["total_counts"][i]), 1.0)
            np.add.at(S[h], ids, np.log1p(ex * 1e4 / tc)); np.add.at(D[h], ids, ex > 0); N[h] += 1
    print(f, N, flush=True)
R = pd.DataFrame({"gene": sym, **{f"mean_log1p_cp10k_h{h}": S[h] / N[h] for h in (0, 1)}, **{f"detect_h{h}": D[h] / N[h] for h in (0, 1)}})
R = R[R.gene != "."].groupby("gene", as_index=False).max(); R.to_csv(f"{O}/rna_ntc_halves.tsv.gz", sep="\t", index=False, float_format="%.6g"); print("cells per half", N)
