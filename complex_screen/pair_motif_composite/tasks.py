# Task files for run_pair.sbatch. python tasks.py gate -> tasks_gate.tsv, tasks_perm.tsv ; python tasks.py all -> tasks_all.tsv (testable candidate pairs)
import sys, pandas as pd
import os
O = os.environ.get("BASE", "/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite")
SPAMO = {"CTCF-STAG2": "MA0139.2", "MAZ-ZNF143": "MA1522.2,MA0088.2", "NFYA-ZNF644": "MA0060.4"}
T = pd.read_csv(f"{O}/pool_sizes.tsv", sep="\t"); T = T[T.testable]
gate = T.category.isin(["gate", "negative"]) | T.category.str.startswith("random_matched")
sel = T[gate] if sys.argv[1] == "gate" else T[~gate]
rows = []
for p in sel.pair:
    for bg in ["vsA", "vsB", "vsInput"]:
        for L in ["known", "8", "10", "12"]: rows.append(f"{p}\thomer\t{bg},{L}")
    rows.append(f"{p}\tsea\t{SPAMO.get(p, '')}"); rows.append(f"{p}\tstreme\t")
open(f"{O}/tasks_{sys.argv[1]}.tsv", "w").write("\n".join(rows) + "\n"); print(len(rows), "tasks")
if sys.argv[1] == "gate":
    pr = [f"{p}\tperm\t{','.join(f'{k:02d}' for k in range(c, c + 5))}" for p in T[T.category.isin(['gate', 'negative'])].pair for c in range(0, 50, 5)]
    open(f"{O}/tasks_perm.tsv", "w").write("\n".join(pr) + "\n"); print(len(pr), "perm tasks")
