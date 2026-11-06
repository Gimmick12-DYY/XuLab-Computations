# Stream HF HEK293T parquet batches: keep panel-TF and NTC cells, extract panel-gene counts.
# Each batch is downloaded to a temp file, processed, then deleted.
import sys, os, subprocess, json
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc

panel_tsv, gene_meta, file_list, task, tmpdir, outdir = sys.argv[1:7]
task = int(task)
genes = pd.read_csv(panel_tsv, sep="\t").gene_symbol.tolist()
gm = pd.read_parquet(gene_meta)
tok = dict(zip(gm.gene_name, gm.gene_token_id))
tok_ids = np.array([tok[g] for g in genes])
keep = set(genes) | {"Non-Targeting"}
files = [l.split("\t")[0] for l in open(file_list) if int(l.split("\t")[2]) == task]
base = "https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion/resolve/main/"
os.makedirs(tmpdir, exist_ok=True)
for f in files:
    out = os.path.join(outdir, os.path.basename(f).replace(".parquet", ".tsv.gz"))
    if os.path.exists(out):
        continue
    tmp = os.path.join(tmpdir, os.path.basename(f))
    subprocess.run(["curl", "-sSL", "--retry", "5", "-o", tmp, base + f], check=True)
    pf = pq.ParquetFile(tmp)
    recs = []
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg)
        t = t.filter(pc.is_in(t["gene_target"], value_set=__import__("pyarrow").array(sorted(keep))))
        if t.num_rows == 0:
            continue
        ids = t["gene_token_id"].to_pylist(); vals = t["gene_expression"].to_pylist()
        m = np.zeros((t.num_rows, len(genes)), dtype=np.float32)
        for i, (gi, gv) in enumerate(zip(ids, vals)):
            d = dict(zip(gi, gv))
            m[i] = [d.get(x, 0.0) for x in tok_ids]
        df = t.select(["cell_barcode", "sample", "gene_target", "guide_target", "total_counts", "n_genes_by_counts", "pass_guide_filter"]).to_pandas()
        recs.append(pd.concat([df, pd.DataFrame(m, columns=genes)], axis=1))
    res = pd.concat(recs) if recs else pd.DataFrame()
    res.to_csv(out + ".part", sep="\t", index=False)
    os.rename(out + ".part", out)
    os.remove(tmp)
    print(f, pf.metadata.num_rows, len(res), flush=True)
