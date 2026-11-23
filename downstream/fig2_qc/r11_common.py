"""Round 11 shared paths and sample table (SPEC.md 1)."""
import sys
from pathlib import Path
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/downstream/fig2_qc")
from fig2_common import BAM, OUT, ROOT  # noqa: E402,F401

O11 = OUT / "round11_chipdeseq2"; O11.mkdir(parents=True, exist_ok=True)
R = ROOT / "data/chip_reps"
SAMPLES = [  # (ChIP, rep, bam)
    ("CTCF", 1, BAM["CTCF"]), ("CTCF", 2, R / "ENCFF474ZKB.bam"),
    ("MAZ", 1, BAM["MAZ"]), ("MAZ", 2, R / "ENCFF298ZTU.bam"),
    ("ZBTB7A", 1, BAM["ZBTB7A"]), ("ZBTB7A", 2, R / "ENCFF421CTJ.bam"),
    ("ZIC2", 1, BAM["ZIC2"]), ("ZIC2", 2, R / "ENCFF042EMQ.bam"),
    ("ZNF282", 1, R / "ZNF282_SRR5197098.markdup.bam"), ("ZNF282", 2, R / "ZNF282_SRR5197099.markdup.bam"),
    ("ZNF777", 1, BAM["ZNF777"]), ("ZNF777", 2, R / "ENCFF859NFR.bam"),
]
