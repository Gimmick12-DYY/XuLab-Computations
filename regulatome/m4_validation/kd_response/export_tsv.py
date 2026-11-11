# Export the out_v2 knockdown response matrices (npz) as plain TSV for viewing outside Python. usage: export_tsv.py -> export_tsv/
# <arm>_unit_z: per-unit z (units x genes); <arm>_unit_doublecentred: the npz "response" (z minus each gene's mean over kept primary units, then minus each unit's mean);
# <arm>_tf_z: raw per-TF z = mean of the TF's kept primary units ("all" or "construct"), no centring (as m7_step2 load_kd before panel collapsing);
# <arm>_genes: NTC detection rate, NTC mean log1p(CP10k), mean primary-unit z per gene; <arm>_units_guide_qc: unit table.
import os
import numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__))); os.makedirs("export_tsv", exist_ok=True)
for a in ["arm2_all_cell", "arm1_high_knockdown"]:
    d = np.load(f"out_v2/{a}/kd_response.npz", allow_pickle=True); q = pd.read_csv(f"out_v2/{a}/guide_qc.tsv", sep="\t"); g = d["genes"].astype(str); u = d["units"].astype(str)
    pd.DataFrame(d["z"], index=u, columns=g).to_csv(f"export_tsv/{a}_unit_z.tsv.gz", sep="\t", float_format="%.4g")
    pd.DataFrame(d["response"], index=u, columns=g).to_csv(f"export_tsv/{a}_unit_doublecentred.tsv.gz", sep="\t", float_format="%.4g")
    prim = q[(q.kept == 1) & q.kind.isin(["all", "construct"])]; ui = {x: i for i, x in enumerate(u)}; tfs = sorted(prim.tf.unique())
    P = np.vstack([np.nanmean(d["z"][[ui[x] for x in prim.unit[prim.tf == t]]], axis=0) for t in tfs])
    pd.DataFrame(P, index=tfs, columns=g).to_csv(f"export_tsv/{a}_tf_z.tsv.gz", sep="\t", float_format="%.4g")
    pd.DataFrame(dict(gene=g, ntc_detect=d["ntc_detect"], ntc_mean_log1pCP10k=d["ntc_mean"], gene_mean_primary_z=d["gene_mean_primary"])).to_csv(f"export_tsv/{a}_genes.tsv", sep="\t", index=False, float_format="%.5g")
    q.to_csv(f"export_tsv/{a}_units_guide_qc.tsv", sep="\t", index=False); print(a, P.shape, len(u))
