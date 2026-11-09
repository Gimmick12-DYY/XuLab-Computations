# Manifest v8 freeze: gene layer = binding-only over ABC links (B6 list e, b6_lists.py), top 1,000 per TF, all 79 TFs.
# Copies list e from m4_validation/b6_core/lists (verified against manifest_b6) to r6_v8/lists/<TF>.tsv (gene, rank). usage: build_v8.py
import os, subprocess
import pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/regulatome/r6_v8"; B = f"{X}/regulatome/m4_validation/b6_core"
assert not os.path.exists(f"{O}/manifest_v8.sha256"), "v8 frozen"
v = subprocess.run("sha256sum -c --quiet manifest_b6.sha256", shell=True, cwd=B, capture_output=True, text=True); assert v.returncode == 0, v.stdout + v.stderr
os.makedirs(f"{O}/lists", exist_ok=True); n = []
for f in sorted(os.listdir(f"{B}/lists")):
    T = pd.read_csv(f"{B}/lists/{f}", sep="\t"); e = T[T.list == "e_binding_only"].gene.tolist(); assert len(e) == 1000, (f, len(e))
    pd.DataFrame(dict(gene=e, rank=range(1, len(e) + 1))).to_csv(f"{O}/lists/{f}", sep="\t", index=False); n.append(f[:-4])
assert len(n) == 79, len(n); print("TFs", len(n))
