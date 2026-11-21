"""Build every Figure 3 panel from source_data/ into output/ (one entry point).

  python make_figure3.py                 # all panels
  python make_figure3.py b1 g            # only panels whose script name starts with these prefixes

Each panel script in panels/ has main(source_dir, out_dir) and writes <name>.pdf (editable text, TrueType Arial embedded),
<name>.svg (live <text>) and a 300 dpi <name>.png preview. See README.md for the panel -> script -> input table.
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
