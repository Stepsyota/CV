from __future__ import annotations

import argparse
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 MLP for lab 3")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/aug_variant_2.yaml"),
        help="Path to experiment config (merged with configs/base.yaml)",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    parser.add_argument(
        "--download-test",
        action="store_true",
        help="Download official CIFAR-10 test split via torchvision if missing",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument(
        "--data-source",
        choices=("kaggle", "torchvision"),
        default=None,
        help="kaggle = PNG+CSV; torchvision = скачать train+test в Colab/ноут (~170 MB), без Kaggle на Drive",
    )
    args = parser.parse_args()
    from lab3.config import load_config

    cfg = load_config(args.config)
    if args.data_source is not None:
        cfg["data_source"] = args.data_source
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    from lab3.train import run_training

    run_training(
        cfg,
        use_mlflow=not args.no_mlflow,
        download_test=args.download_test,
    )


if __name__ == "__main__":
    main()
