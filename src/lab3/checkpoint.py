from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    cfg: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def append_history(history_path: Path, record: dict[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    if history_path.is_file():
        with history_path.open(encoding="utf-8") as f:
            history = json.load(f)
    history.append(record)
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
