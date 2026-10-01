from __future__ import annotations

from typing import Any


def print_section(title: str) -> None:
    line = "-" * 40
    print(line)
    print(title)
    print(line)


def print_architecture(architecture: str) -> None:
    print_section("Architecture")
    print(architecture)


def print_hyperparameters(cfg: dict[str, Any], num_params: int) -> None:
    print_section("Hyperparameters")
    print(f"Learning Rate: {cfg['lr']}")
    print(f"Batch size: {cfg['batch_size']}")
    print(f"Max train epoch: {cfg['epochs']}")
    print(f"Optimizer: {cfg.get('optimizer', 'adamw')}")
    print(f"Weight decay: {cfg.get('weight_decay', 0)}")
    print(f"Trainable parameters: {num_params}")
    print(f"Aug table id: {cfg.get('aug_table_id')}")
    sch = cfg.get("scheduler") or {}
    if sch.get("name") and str(sch.get("name")).lower() not in {"", "none"}:
        print(f"LR scheduler: {sch.get('name')}")
    es = cfg.get("early_stopping", {})
    if es:
        print(
            f"Early stopping: monitor={es.get('monitor')}, "
            f"patience={es.get('patience')}, min_delta={es.get('min_delta')}"
        )


def print_epoch_line(epoch: int, train_acc: float, test_acc: float, elapsed_sec: float) -> None:
    print(
        f"Epoch #{epoch}: train_accuracy = {train_acc:.4f} ; "
        f"test_accuracy = {test_acc:.4f} ; time = {elapsed_sec:.2f}s"
    )
