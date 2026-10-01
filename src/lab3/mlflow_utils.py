from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import yaml

from lab3.augment import describe_augment_variant


def resolve_mlruns_dir(cfg: dict[str, Any]) -> Path:
    repo_root = Path(cfg["_repo_root"])
    mlruns_dir = repo_root / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    return mlruns_dir


def resolve_tracking_uri(cfg: dict[str, Any]) -> str:
    """Единый каталог ./mlruns в корне репо; UI — scripts/mlflow_server.ps1."""
    mlruns_dir = resolve_mlruns_dir(cfg)
    raw = str((cfg.get("mlflow") or {}).get("tracking_uri", "file")).strip()
    lowered = raw.lower()
    if lowered in {"", "file", "local", "auto"}:
        return mlruns_dir.as_uri()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        return raw.rstrip("/")
    if lowered.startswith("file:"):
        return raw
    return mlruns_dir.as_uri()


def setup_mlflow(cfg: dict[str, Any]) -> str:
    uri = resolve_tracking_uri(cfg)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    return uri


def _safe_params(params: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in params.items():
        if value is None:
            continue
        out[key] = str(value)
    return out


def log_config_params(cfg: dict[str, Any], num_params: int) -> None:
    model_cfg = cfg.get("model") or {}
    train_cfg = cfg.get("training") or {}
    sch_cfg = cfg.get("scheduler") or {}
    es_cfg = cfg.get("early_stopping") or {}

    params = _safe_params(
        {
            "aug_table_id": cfg.get("aug_table_id"),
            "run_name": cfg.get("run_name"),
            "data_source": cfg.get("data_source"),
            "lr": cfg.get("lr"),
            "batch_size": cfg.get("batch_size"),
            "epochs": cfg.get("epochs"),
            "optimizer": cfg.get("optimizer", "adamw"),
            "weight_decay": cfg.get("weight_decay", 0),
            "num_params": num_params,
            "hidden_dims": model_cfg.get("hidden_dims"),
            "activation": model_cfg.get("activation", "relu"),
            "dropout": model_cfg.get("dropout", 0),
            "label_smoothing": train_cfg.get("label_smoothing"),
            "mixup_alpha": train_cfg.get("mixup_alpha"),
            "grad_clip_norm": train_cfg.get("grad_clip_norm"),
            "scheduler": sch_cfg.get("name"),
            "warmup_epochs": sch_cfg.get("warmup_epochs"),
            "early_stopping_patience": es_cfg.get("patience"),
            "early_stopping_monitor": es_cfg.get("monitor"),
            "seed": cfg.get("seed"),
        }
    )
    mlflow.log_params(params)

    aug_id = cfg.get("aug_table_id")
    if aug_id is not None:
        mlflow.set_tags(
            {
                "run_name": str(cfg.get("run_name", "")),
                "aug_table_id": str(aug_id),
                "augmentation": describe_augment_variant(int(aug_id)),
            }
        )

    repo_root = Path(cfg["_repo_root"])
    merged_path = repo_root / "artifacts" / "mlflow_merged_config.yaml"
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    dump_cfg = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    merged_path.write_text(yaml.safe_dump(dump_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    mlflow.log_artifact(str(merged_path), artifact_path="config")


def log_epoch_metrics(
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    test_loss: float,
    test_acc: float,
    best_test_acc: float,
    learning_rate: float | None = None,
) -> None:
    metrics: dict[str, float] = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "best_test_accuracy": best_test_acc,
    }
    if learning_rate is not None:
        metrics["learning_rate"] = learning_rate
    mlflow.log_metrics(metrics, step=epoch)


def log_artifacts(paths: list[Path]) -> None:
    for path in paths:
        if path.is_file():
            mlflow.log_artifact(str(path))
        elif path.is_dir():
            mlflow.log_artifacts(str(path))
