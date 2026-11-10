# Fallback (iii): distal-only version of a per-TF pool set. Keeps foreground windows with no GENCODE v46 basic TSS within 1 kb and
# background bins not flagged promoter (ref/promoter_bins.txt); the promoter flag was a matching stratum, so the pools stay matched.
# Usage: python distal_pools.py <src base> <dst base>   e.g. per_tf_raw per_tf_raw_distal
import os, sys, subprocess
O = os.path.dirname(os.path.abspath(__file__)); SRC, DST = [os.path.join(O, a) for a in sys.argv[1:3]]
prom = set(f"b{x}" for x in open(f"{O}/ref/promoter_bins.txt").read().split())
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True, executable="/bin/bash"); assert r.returncode == 0, r.stderr; return r.stdout
rows, kt, dt = [], [], []
for t in sorted(os.listdir(f"{SRC}/pools")):
    r = [t]
    for pt in ["disc", "val"]:
        d = f"{DST}/pools/{t}/{pt}"; os.makedirs(d, exist_ok=True); fg = f"{SRC}/pools/{t}/{pt}/fg.bed"
        near = set(sh(f"module load bedtools >/dev/null 2>&1; bedtools window -u -w 1000 -a {fg} -b {O}/ref/tss.bed | cut -f4").split()) if os.path.getsize(fg) else set()
        F = [l for l in open(fg) if l.split("\t")[3] not in near and l.split("\t")[3] not in prom]
        B = [l for l in open(f"{SRC}/pools/{t}/{pt}/bg.bed") if l.split("\t")[3] not in prom]
        n = min(len(F), len(B)); open(f"{d}/fg.bed", "w").writelines(F[:n] if len(F) <= n else F[:n]); open(f"{d}/bg.bed", "w").writelines(B[:n]); r.append(n)
    rows.append(r)
    if r[1] >= 300: kt += [f"{t}\tknown", f"{t}\tval_known"]; dt += [f"{t}\t8", f"{t}\t10", f"{t}\t12"]
open(f"{DST}/tf_pools.tsv", "w").write("tf\tdisc_n\tval_n\n" + "\n".join("\t".join(map(str, r)) for r in rows) + "\n"); open(f"{DST}/tasks_tf.tsv", "w").write("\n".join(kt + dt) + "\n")
print(len(kt) // 2, "TFs with >= 300 distal discovery windows"); print("\n".join("\t".join(map(str, r)) for r in rows))
