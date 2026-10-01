from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_path: str | Path, base_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path).resolve()
    repo_root = path.parent.parent if path.parent.name == "configs" else path.parent
    base_file = Path(base_path) if base_path else repo_root / "configs" / "base.yaml"

    with base_file.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if path != base_file.resolve():
        with path.open(encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, override)

    cfg["_config_path"] = str(path)
    cfg["_repo_root"] = str(repo_root)

    if env_root := os.environ.get("CIFAR10_DATA_ROOT"):
        cfg["data_root"] = env_root
    if "mlflow" not in cfg or not isinstance(cfg["mlflow"], dict):
        cfg["mlflow"] = {}
    if env_mlflow := os.environ.get("MLFLOW_TRACKING_URI"):
        cfg["mlflow"]["tracking_uri"] = env_mlflow

    cfg["data_root"] = str((repo_root / cfg["data_root"]).resolve()) if not Path(cfg["data_root"]).is_absolute() else cfg["data_root"]
    cfg["torchvision_data_root"] = str(
        (repo_root / cfg["torchvision_data_root"]).resolve()
        if not Path(cfg["torchvision_data_root"]).is_absolute()
        else cfg["torchvision_data_root"]
    )
    cfg["checkpoint_dir"] = str(
        (repo_root / cfg["checkpoint_dir"]).resolve()
        if not Path(cfg["checkpoint_dir"]).is_absolute()
        else cfg["checkpoint_dir"]
    )
    return cfg
