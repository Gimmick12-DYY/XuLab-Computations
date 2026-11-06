# Propagation sources with the ATAC-only top 1,000 genes (ABC x log1p ATAC CPM) removed. Writes sources_atacfiltered/<source>.tsv and
# sources_atacfiltered_summary.tsv. The source and direct lists are left as they are.
import glob, os
import pandas as pd
O = "/work/users/d/y/dyy12/XuLab/regulatome/r2_targets"; os.makedirs(f"{O}/sources_atacfiltered", exist_ok=True)
T = pd.read_csv(f"{O}/per_tf/RBBP4.tsv.gz", sep="\t"); atac = set(T.nlargest(1000, "S_atac").gene)
for t in ("NFYA", "ZNF777"): assert set(pd.read_csv(f"{O}/per_tf/{t}.tsv.gz", sep="\t").nlargest(1000, "S_atac").gene) == atac
rows = []
for kind, d in (("TF", "sources"), ("module", "top1000_modules")):
    for f in sorted(glob.glob(f"{O}/{d}/*.tsv")):
        n = os.path.basename(f)[:-4]; S = pd.read_csv(f, sep="\t"); K = S[~S.gene.isin(atac)]
        K.to_csv(f"{O}/sources_atacfiltered/{n}.tsv", sep="\t", index=False, float_format="%.5g")
        rows.append(dict(source=n, kind=kind, genes_before=len(S), atac_overlap_removed=len(S) - len(K), genes_after=len(K)))
R = pd.DataFrame(rows); R.to_csv(f"{O}/sources_atacfiltered_summary.tsv", sep="\t", index=False); print(R.to_string(index=False))
