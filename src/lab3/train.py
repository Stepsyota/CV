from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import torch
import torch.nn as nn
from torch.distributions import Beta
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from lab3.augment import augment_callable, build_augment_pipeline
from lab3.checkpoint import append_history, load_checkpoint, save_checkpoint
from lab3.config import load_config
from lab3.data import build_dataloaders
from lab3.logging_fmt import print_architecture, print_epoch_line, print_hyperparameters, print_section
from lab3.metrics import evaluate
from lab3.mlflow_utils import log_config_params, log_epoch_metrics, setup_mlflow
from lab3.model import build_model, count_trainable_parameters


def _build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: dict[str, Any]
) -> torch.optim.lr_scheduler.LRScheduler | None:
    sch_cfg = cfg.get("scheduler") or {}
    name = str(sch_cfg.get("name", "none")).lower()
    if name in {"", "none"}:
        return None
    if name == "cosine":
        total_epochs = int(cfg["epochs"])
        warmup_epochs = int(sch_cfg.get("warmup_epochs", 0))
        eta_min = float(sch_cfg.get("eta_min", 1e-6))
        t_max = int(sch_cfg.get("T_max", max(total_epochs - warmup_epochs, 1)))
        if warmup_epochs > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=float(sch_cfg.get("warmup_start_factor", 0.1)),
                total_iters=warmup_epochs,
            )
            cosine = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
            return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(sch_cfg.get("factor", 0.5)),
            patience=int(sch_cfg.get("patience", 5)),
            min_lr=float(sch_cfg.get("min_lr", 1e-6)),
        )
    raise ValueError(f"Unknown scheduler: {name}")


def _build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(cfg.get("optimizer", "adamw")).lower()
    lr = float(cfg["lr"])
    wd = float(cfg.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    if name in {"adam", "adamw"}:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")


def _mixup_batch(
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return batch_x, batch_y, batch_y, 1.0
    lam = float(Beta(alpha, alpha).sample().item())
    lam = max(lam, 1.0 - lam)
    index = torch.randperm(batch_x.size(0), device=batch_x.device)
    mixed_x = lam * batch_x + (1.0 - lam) * batch_x[index]
    return mixed_x, batch_y, batch_y[index], lam


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mixup_alpha: float = 0.0,
    grad_clip_norm: float | None = None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_x, batch_y in tqdm(loader, desc="train", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_x, y_a, y_b, lam = _mixup_batch(batch_x, batch_y, mixup_alpha)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        if lam < 1.0:
            loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
        else:
            loss = criterion(logits, batch_y)
        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
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
    scheduler = _build_scheduler(optimizer, cfg)
    train_cfg = cfg.get("training") or {}
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    mixup_alpha = float(train_cfg.get("mixup_alpha", 0.0))
    grad_clip_norm = train_cfg.get("grad_clip_norm")
    grad_clip = float(grad_clip_norm) if grad_clip_norm is not None else None

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

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

    mlflow_uri: str | None = None
    if use_mlflow:
        mlflow_uri = setup_mlflow(cfg)
        mlflow.start_run(run_name=run_name)
        log_config_params(cfg, num_params)
        print(f"MLflow: {mlflow_uri}  experiment={cfg['mlflow']['experiment_name']}  run={run_name}")

    try:
        for epoch in range(1, int(cfg["epochs"]) + 1):
            t0 = time.perf_counter()
            train_loss, train_acc = _train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                mixup_alpha=mixup_alpha,
                grad_clip_norm=grad_clip,
            )
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
                current_lr = float(optimizer.param_groups[0]["lr"])
                log_epoch_metrics(
                    epoch,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                    test_loss,
                    test_acc,
                    best_test_acc,
                    learning_rate=current_lr,
                )

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric = val_acc if monitor == "val_accuracy" else test_acc
                    scheduler.step(metric)
                else:
                    scheduler.step()

            if epochs_without_improve >= patience:
                print(f"Early stopping on epoch {epoch} (monitor={monitor}).")
                break

        if best_ckpt_path.is_file():
            load_checkpoint(best_ckpt_path, model)
            _, best_test_acc = evaluate(model, test_loader, device)

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
            mlflow.log_artifact(str(best_ckpt_path), artifact_path="checkpoints")
            mlflow.log_artifact(str(history_path), artifact_path="checkpoints")
            mlflow.log_artifact(str(curves_path), artifact_path="plots")
            if Path(cfg["_config_path"]).is_file():
                mlflow.log_artifact(cfg["_config_path"], artifact_path="config")
            if mlflow_uri:
                print(f"MLflow run saved. Open UI: scripts/mlflow_server.ps1 → http://127.0.0.1:5000")

        return summary
    finally:
        if use_mlflow and mlflow.active_run() is not None:
            mlflow.end_run()


def train_from_config_path(config_path: str | Path, use_mlflow: bool = True, download_test: bool = True) -> dict[str, Any]:
    cfg = load_config(config_path)
    return run_training(cfg, use_mlflow=use_mlflow, download_test=download_test)
