# Panel TF list -> target gene symbols used in X-Atlas/Orion
import pandas as pd, sys
meta = pd.read_csv(sys.argv[1])
tfs = sorted(set(meta["TF"].dropna()) - {"TF"})
# non-HGNC / renamed panel names
ALIAS = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
rows = []
for tf in tfs:
    for g in ALIAS.get(tf, [tf]):
        rows.append((tf, g, "alias" if tf in ALIAS else "same"))
pd.DataFrame(rows, columns=["panel_tf", "gene_symbol", "name_mapping"]).to_csv(sys.argv[2], sep="\t", index=False)
print(len(tfs), "panel TFs ->", len(rows), "gene symbols")
