"""5' ends of every TF's reads inside windows, plus each TF's total reads (read-only on xujie_lab)."""
import gzip, sys
from pathlib import Path
FR = Path("/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment"); OUT = sys.argv[1]
W = [(w.split(":")[0], *map(int, w.split(":")[1].split("-"))) for w in sys.argv[2:]]
with open(OUT, "w") as o:
    for f in sorted(FR.glob("TF.*.bed.gz")):
        tf = f.name[3:-7]; n = 0
        with gzip.open(f, "rt") as fh:
            for ln in fh:
                n += 1; x = ln.split("\t", 6)
                p = int(x[1]) if x[5][0] == "+" else int(x[2]) - 1
                for c, a, b in W:
                    if x[0] == c and a <= p < b: o.write(f"{tf}\t{c}\t{p}\n")
        o.write(f"{tf}\tTOTAL\t{n}\n"); print(tf, n, flush=True)
