"""ETS2 coordinated binding network, overview plus zoom (183 x 92 mm, print size).

Left: every ETS2 component with >= 10 regions, packed, in grey (darker grey = edges inside higher-order coordinated binding
modules); the box marks the zoomed region. Right: the boxed region in colour (one colour per module, black outline = hinge
region shared by >= 2 modules, light grey = other pairs), each component labelled by its chromosome span. The box is chosen
automatically (_network.pick_zoom_box): 8-12 whole components of 40-250 regions, most module regions per area.
ETS2 set = the one used by the other Fig 3 panels: cobinding/work/ourrun_k50/ETS2 (Cicero k = 50) and
cobinding/results/ourrun_k50/ETS2/cliques.tsv. Drawing rules: _network.py.
Inputs (source_data/): ets2_peak_edges.tsv.gz, ets2_modules.tsv.  Output: ets2_network.{pdf,svg,png}, ets2_network_components.tsv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _network  # noqa: E402


def main(src, out):
    _network.draw_zoom(Path(src) / "ets2_peak_edges.tsv.gz", Path(src) / "ets2_modules.tsv", out, "ets2_network", tf="ETS2")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
