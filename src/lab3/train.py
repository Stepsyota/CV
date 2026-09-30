from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lab3.augment import augment_callable, build_augment_pipeline
from lab3.checkpoint import append_history, save_checkpoint
from lab3.config import load_config
from lab3.data import build_dataloaders
from lab3.logging_fmt import print_architecture, print_epoch_line, print_hyperparameters, print_section
from lab3.metrics import evaluate
from lab3.mlflow_utils import log_config_params, log_epoch_metrics, setup_mlflow
from lab3.model import build_model, count_trainable_parameters


def _build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(cfg.get("optimizer", "adamw")).lower()
    lr = float(cfg["lr"])
    wd = float(cfg.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    if name in {"adam", "adamw"}:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_x, batch_y in tqdm(loader, desc="train", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_y.size(0)
        correct += (logits.argmax(dim=1) == batch_y).sum().item()
        total += batch_y.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def _save_curves(history_path: Path, out_path: Path) -> None:
    if not history_path.is_file():
        return
    with history_path.open(encoding="utf-8") as f:
        history = json.load(f)
    epochs = [h["epoch"] for h in history]
    train_acc = [h["train_accuracy"] for h in history]
    test_acc = [h["test_accuracy"] for h in history]
    val_acc = [h["val_accuracy"] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="train")
    plt.plot(epochs, val_acc, label="val")
    plt.plot(epochs, test_acc, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def run_training(cfg: dict[str, Any], use_mlflow: bool = True, download_test: bool = True) -> dict[str, Any]:
    torch.manual_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    aug_id = int(cfg["aug_table_id"])
    pipeline = build_augment_pipeline(aug_id, cfg.get("augment"))
    augment_train = augment_callable(pipeline)

    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, augment_train=augment_train, download_test=download_test
    )

    model = build_model(cfg).to(device)
    num_params = count_trainable_parameters(model)
    optimizer = _build_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss()

    run_name = cfg.get("run_name", f"aug_table_{aug_id}")
    run_dir = Path(cfg["checkpoint_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.json"
    best_ckpt_path = run_dir / "best.pt"
    last_ckpt_path = run_dir / "last.pt"
    curves_path = run_dir / "accuracy_curves.png"

    print_architecture(model.architecture_string())
    print_hyperparameters(cfg, num_params)
    print_section("Train & Test process")

    es_cfg = cfg.get("early_stopping", {})
    monitor = es_cfg.get("monitor", "val_accuracy")
    patience = int(es_cfg.get("patience", 10))
    min_delta = float(es_cfg.get("min_delta", 0.0))

    best_test_acc = 0.0
    best_val_acc = 0.0
    epochs_without_improve = 0

    if use_mlflow:
        setup_mlflow(cfg)
        mlflow.start_run(run_name=run_name)
        log_config_params(cfg, num_params)

    try:
        for epoch in range(1, int(cfg["epochs"]) + 1):
            t0 = time.perf_counter()
            train_loss, train_acc = _train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, device)
            test_loss, test_acc = evaluate(model, test_loader, device)
            elapsed = time.perf_counter() - t0

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                save_checkpoint(
                    best_ckpt_path,
                    model,
                    optimizer,
                    epoch,
                    {"test_accuracy": test_acc, "val_accuracy": val_acc},
                    cfg,
                )

            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            save_checkpoint(
                last_ckpt_path,
                model,
                optimizer,
                epoch,
                {"test_accuracy": test_acc, "val_accuracy": val_acc},
                cfg,
            )

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "time_sec": elapsed,
            }
            append_history(history_path, record)
            print_epoch_line(epoch, train_acc, test_acc, elapsed)

            if use_mlflow:
                log_epoch_metrics(
                    epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, best_test_acc
                )

            if epochs_without_improve >= patience:
                print(f"Early stopping on epoch {epoch} (monitor={monitor}).")
                break

        _save_curves(history_path, curves_path)
        summary = {
            "run_name": run_name,
            "best_test_accuracy": best_test_acc,
            "best_checkpoint": str(best_ckpt_path),
            "history_path": str(history_path),
            "curves_path": str(curves_path),
        }

        with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print_section("Result")
        print(f"Best test accuracy: {best_test_acc:.4f}")
        print(f"Checkpoint: {best_ckpt_path}")

        if use_mlflow:
            mlflow.log_metric("best_test_accuracy_final", best_test_acc)
            mlflow.log_artifact(str(best_ckpt_path))
            mlflow.log_artifact(str(history_path))
            mlflow.log_artifact(str(curves_path))
            if Path(cfg["_config_path"]).is_file():
                mlflow.log_artifact(cfg["_config_path"])

        return summary
    finally:
        if use_mlflow and mlflow.active_run() is not None:
            mlflow.end_run()


def train_from_config_path(config_path: str | Path, use_mlflow: bool = True, download_test: bool = True) -> dict[str, Any]:
    cfg = load_config(config_path)
    return run_training(cfg, use_mlflow=use_mlflow, download_test=download_test)
