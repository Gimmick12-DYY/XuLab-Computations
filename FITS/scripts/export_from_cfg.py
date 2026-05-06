#!/usr/bin/env python
from __future__ import annotations
import subprocess,sys
from pathlib import Path
import yaml
from _cfg import base_parser, load_config, resolve_paths

def main():
    ap=base_parser("")
    args=ap.parse_args()
    cfg=resolve_paths(load_config(args.config), args.work_dir)
    inp=cfg["paths"]["input_rds"]
    out=str(Path(cfg["paths"]["work_dir"]) / "mm")
    cmd=["Rscript","scripts/01_export_rds_to_mm.R","--input",inp,"--outdir",out]
    return subprocess.run(cmd).returncode

if __name__=="__main__":
    raise SystemExit(main())
