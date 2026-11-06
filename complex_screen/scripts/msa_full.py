# One unpaired ColabFold (UniRef+env) MSA per full-length protein, reused for every pair and trimmed construct
import json, sys, os, glob, time
from pathlib import Path
from boltz.data.msa.mmseqs2 import run_mmseqs2

seq_dir, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
for f in sorted(glob.glob(f"{seq_dir}/uniprot/*.json")):
    g = Path(f).stem
    dst = f"{out}/{g}.a3m"
    if os.path.exists(dst): continue
    s = json.load(open(f))["sequence"]["value"]
    t0 = time.time()
    for i in range(4):
        try:
            a3m = run_mmseqs2(s, f"{out}/tmp_{g}", use_env=True, use_pairing=False)
            break
        except Exception as e:
            print(g, "retry", i, e, flush=True); time.sleep(30 * (i + 1))
    else:
        print(g, "FAILED", flush=True); continue
    a3m = a3m[0] if isinstance(a3m, list) else a3m
    open(dst, "w").write(a3m)
    print(g, len(s), "aa", a3m.count(">"), "seqs", f"{time.time() - t0:.0f}s", flush=True)
