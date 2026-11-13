# m26 (copy of m4 stream_b1.py):  stream the HF HEK293T parquet batches; keep all columns for cells targeting the 81 panel genes and Non-Targeting cells. Each batch goes to a temp file that is deleted after processing.
import os, subprocess, sys
import pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
panel_tsv, file_list, task, tmpdir, outdir = sys.argv[1:6]; task = int(task)
keep = sorted(set(pd.read_csv(panel_tsv, sep="\t").gene_symbol) | {"__none__"}); assert len(keep) == 10, len(keep)  # m26: nine cohesin/CTCF-related genes; Non-Targeting cells are NOT kept again (already in panel_ntc_cells)
base = "https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion/resolve/main/"; os.makedirs(tmpdir, exist_ok=True)
for l in open(file_list):
    f, size, tk = l.rstrip("\n").split("\t")
    if int(tk) != task: continue
    out = os.path.join(outdir, os.path.basename(f))
    if os.path.exists(out): continue
    tmp = os.path.join(tmpdir, os.path.basename(f)); subprocess.run(["curl", "-sSLf", "--retry", "5", "-o", tmp, base + f], check=True); assert os.path.getsize(tmp) == int(size), (f, os.path.getsize(tmp), size)
    pf = pq.ParquetFile(tmp); parts = []
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg); parts.append(t.filter(pc.is_in(t["gene_target"], value_set=pa.array(keep))))
    res = pa.concat_tables(parts); pq.write_table(res, out + ".part", compression="zstd"); os.rename(out + ".part", out); n = pf.metadata.num_rows; os.remove(tmp); print(f, n, res.num_rows, os.path.getsize(out), flush=True)
