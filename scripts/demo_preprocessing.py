from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from lab3.augment import build_augment_pipeline, describe_augment_variant
from lab3.config import load_config
from lab3.data import load_kaggle_train_index, resolve_train_image_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo preprocessing for table variants 2, 6, 7")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/preprocess_demo"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo = Path(cfg["_repo_root"])
    df = load_kaggle_train_index(cfg["data_root"])
    image_dir = resolve_train_image_dir(cfg["data_root"])
    sample_ids = df["id"].astype(int).head(args.num_samples).tolist()

    variant_configs = [
        (2, repo / "configs" / "aug_variant_2.yaml"),
        (6, repo / "configs" / "aug_variant_6.yaml"),
        (7, repo / "configs" / "aug_variant_7.yaml"),
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for aug_id, variant_cfg_path in variant_configs:
        vcfg = load_config(variant_cfg_path)
        pipeline = build_augment_pipeline(aug_id, vcfg.get("augment"))
        variant_dir = args.out_dir / f"variant_{aug_id}"
        variant_dir.mkdir(parents=True, exist_ok=True)

        for image_id in sample_ids:
            path = image_dir / f"{image_id}.png"
            image = np.array(Image.open(path).convert("RGB"))
            augmented = pipeline(image=image)["image"]

            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            axes[0].imshow(image)
            axes[0].set_title("Original")
            axes[0].axis("off")
            axes[1].imshow(augmented)
            axes[1].set_title(describe_augment_variant(aug_id))
            axes[1].axis("off")
            fig.suptitle(f"Table #{aug_id}: {describe_augment_variant(aug_id)}")
            fig.tight_layout()
            out_path = variant_dir / f"sample_{image_id}.png"
            fig.savefig(out_path, dpi=120)
            plt.close(fig)

    print(f"Saved demos to {args.out_dir}")


if __name__ == "__main__":
    main()
