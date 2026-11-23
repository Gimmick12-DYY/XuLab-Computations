"""ETS2 coordinated binding network, whole genome (style of the original RBBP4 network): every ETS2 network with >= MIN regions as its own
force-directed cluster (grey = surrounding pairs), each higher-order coordinated binding module in its own colour, hinge regions shared by
>= 2 modules outlined in black, the 24 largest networks labelled with the genes of the two best-connected promoter regions
(RefSeq Select TSS within +-1 kb), chromosome span and region count.
Size cutoff MIN = 30 regions (1,184 networks of >= 10 regions do not fit legibly on one canvas); stated in the summary line.
ETS2 set = cobinding/work/ourrun_k50/ETS2 (Cicero k = 50) and cobinding/results/ourrun_k50/ETS2/cliques.tsv. Drawing rules: _network.py.
Inputs (source_data/): ets2_peak_edges.tsv.gz, ets2_modules.tsv, refseq_select_tss.tsv.gz. Output: ets2_network_genome.{pdf,svg,png} (24 in canvas)."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _network  # noqa: E402

MIN = int(os.environ.get("ETS2_GENOME_MIN", 30))


def main(src, out):
    src = Path(src)
    _network.draw_all(src / "ets2_peak_edges.tsv.gz", src / "ets2_modules.tsv", out, "ets2_network_genome", min_comp=MIN,
                      tss_path=src / "refseq_select_tss.tsv.gz", tf="ETS2")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
