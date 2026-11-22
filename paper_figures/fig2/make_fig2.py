"""Build every Figure 2 panel from source_data/ into output/ (one entry point).

  python make_fig2.py            # all panels
  python make_fig2.py a          # only panels whose script name starts with these prefixes

Source data are written by downstream/fig2_qc/r21_fig2_source.py (corr_matrix_6x6.tsv, heatmaps_<TF>.npz,
heatmap_counts.tsv). Each panel script has main(source_dir, out_dir) and writes .pdf (editable TrueType Arial), .svg
(live text) and a 300 dpi .png preview through common/style.py.
"""
import importlib.util
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC, OUT = HERE / "source_data", HERE / "output"


def main(prefixes):
    scripts = sorted(p for p in (HERE / "panels").glob("*.py") if not p.name.startswith(("_", ".")))
    if prefixes:
        scripts = [p for p in scripts if any(p.stem.startswith(x) for x in prefixes)]
    OUT.mkdir(exist_ok=True)
    for p in scripts:
        t = time.time()
        spec = importlib.util.spec_from_file_location(p.stem, p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main(SRC, OUT)
        print(f"[{p.stem}] done in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
