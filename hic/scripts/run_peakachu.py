#!/usr/bin/env python3
"""Peakachu CLI with sklearn>=1.3 compatibility for pretrained models."""
from __future__ import annotations

import runpy
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn_rf_compat import patch  # noqa: E402


def main() -> None:
    patch()
    exe = shutil.which("peakachu")
    if not exe:
        sys.exit("peakachu not on PATH")
    # Keep user args; run the installed CLI in this process so the patch applies.
    sys.argv[0] = exe
    runpy.run_path(exe, run_name="__main__")


if __name__ == "__main__":
    main()
