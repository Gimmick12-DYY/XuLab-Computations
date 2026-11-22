"""RBBP4 coordinated binding network (original colour code, no Hi-C loop overlay; PI request 2026-11-21).

Every RBBP4 component with >= 10 regions on one 24 x 24 in canvas (large original; scale it down when placing); the 24 largest
are labelled by chromosome span and number of regions. Drawing rules: _network.py.
Inputs (source_data/): rbbp4_peak_edges.tsv.gz, rbbp4_modules.tsv.  Output: rbbp4_network.{pdf,svg,png}, rbbp4_network_components.tsv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _network  # noqa: E402


def main(src, out):
    _network.draw_all(Path(src) / "rbbp4_peak_edges.tsv.gz", Path(src) / "rbbp4_modules.tsv", out, "rbbp4_network")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
