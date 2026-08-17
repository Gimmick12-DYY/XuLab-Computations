"""Shared config helper for the unified pipeline."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CFG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overlay onto a deep copy of base (dicts only)."""
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """Load YAML config, deep-merging onto default.yaml when path differs.

    Per-TF configs are full copies historically; merging lets new default keys
    (e.g. open_chromatin_bed) apply without editing every TF yaml.
    """
    path = Path(path)
    with path.open() as fh:
        cfg = yaml.safe_load(fh) or {}
    if path.resolve() == DEFAULT_CFG.resolve():
        return cfg
    if DEFAULT_CFG.is_file():
        with DEFAULT_CFG.open() as fh:
            base = yaml.safe_load(fh) or {}
        return _deep_merge(base, cfg)
    return cfg


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=str, default=str(DEFAULT_CFG))
    p.add_argument("--work-dir", type=str, default=None)
    return p


def resolve_paths(cfg: dict, work_dir_override: str | None = None) -> dict:
    if work_dir_override:
        cfg.setdefault("paths", {})["work_dir"] = work_dir_override
    wd = Path(cfg["paths"]["work_dir"]).expanduser().resolve()
    wd.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["work_dir"] = str(wd)

    subdirs = {
        "mm":      wd / "mm",
        "seqs":    wd / "seqs",
        "model":   wd / "model",
        "predict": wd / "predict",
        "impute":  wd / "impute",
        "logs":    wd / "logs",
    }
    for d in subdirs.values():
        d.mkdir(parents=True, exist_ok=True)
    cfg["paths"].update({k: str(v) for k, v in subdirs.items()})
    return cfg
