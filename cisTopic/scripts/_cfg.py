"""Small shared helper: load YAML config + allow CLI overrides."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml


def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def base_parser(description: str) -> argparse.ArgumentParser:
    """Return an ArgumentParser pre-populated with --config and --work-dir."""
    p = argparse.ArgumentParser(description=description)
    default_cfg = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    p.add_argument("--config", type=str, default=str(default_cfg),
                   help="Path to YAML config (default: configs/default.yaml)")
    p.add_argument("--work-dir", type=str, default=None,
                   help="Override paths.work_dir from the config.")
    return p


def resolve_paths(cfg: dict, work_dir_override: str | None = None) -> dict:
    if work_dir_override:
        cfg.setdefault("paths", {})["work_dir"] = work_dir_override
    wd = Path(cfg["paths"]["work_dir"]).expanduser().resolve()
    wd.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["work_dir"] = str(wd)

    subdirs = {
        "mm":        wd / "mm",
        "obj":       wd / "obj",
        "models":    wd / "models",
        "select":    wd / "select",
        "impute":    wd / "impute",
        "downstream":wd / "downstream",
        "logs":      wd / "logs",
    }
    for d in subdirs.values():
        d.mkdir(parents=True, exist_ok=True)
    cfg["paths"].update({k: str(v) for k, v in subdirs.items()})
    return cfg
