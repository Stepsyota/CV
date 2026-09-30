from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow


def setup_mlflow(cfg: dict[str, Any]) -> None:
    uri = cfg["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])


def log_config_params(cfg: dict[str, Any], num_params: int) -> None:
    mlflow.log_param("aug_table_id", cfg.get("aug_table_id"))
    mlflow.log_param("run_name", cfg.get("run_name"))
    mlflow.log_param("lr", cfg["lr"])
    mlflow.log_param("batch_size", cfg["batch_size"])
    mlflow.log_param("epochs", cfg["epochs"])
    mlflow.log_param("optimizer", cfg.get("optimizer", "adamw"))
    mlflow.log_param("weight_decay", cfg.get("weight_decay", 0))
    mlflow.log_param("num_params", num_params)
    mlflow.log_param("hidden_dims", str(cfg["model"]["hidden_dims"]))
    mlflow.log_param("activation", cfg["model"].get("activation", "relu"))
    mlflow.log_param("dropout", cfg["model"].get("dropout", 0))


def log_epoch_metrics(
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    test_loss: float,
    test_acc: float,
    best_test_acc: float,
) -> None:
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("train_accuracy", train_acc, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_metric("val_accuracy", val_acc, step=epoch)
    mlflow.log_metric("test_loss", test_loss, step=epoch)
    mlflow.log_metric("test_accuracy", test_acc, step=epoch)
    mlflow.log_metric("best_test_accuracy", best_test_acc, step=epoch)


def log_artifacts(paths: list[Path]) -> None:
    for path in paths:
        if path.is_file():
            mlflow.log_artifact(str(path))
        elif path.is_dir():
            mlflow.log_artifacts(str(path))
