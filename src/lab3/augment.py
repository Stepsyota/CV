from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np


def build_augment_pipeline(aug_table_id: int, params: dict[str, Any] | None = None) -> A.BasicTransform:
    params = params or {}
    if aug_table_id == 2:
        crop_h = int(params.get("crop_height", 32))
        crop_w = int(params.get("crop_width", 32))
        pad = int(params.get("pad_pixels", 4))
        padded = crop_h + 2 * pad
        return A.Compose(
            [
                A.PadIfNeeded(
                    min_height=padded,
                    min_width=padded,
                    border_mode=int(params.get("pad_border_mode", 4)),
                    p=1.0,
                ),
                A.RandomCrop(height=crop_h, width=crop_w, p=1.0),
                A.HorizontalFlip(p=float(params.get("horizontal_flip_p", 0.5))),
            ]
        )
    if aug_table_id == 6:
        return A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=float(params.get("brightness_limit", 0.2)),
                    contrast_limit=float(params.get("contrast_limit", 0.2)),
                    p=float(params.get("brightness_contrast_p", 0.8)),
                ),
            ]
        )
    if aug_table_id == 7:
        return A.Compose(
            [
                A.GaussNoise(
                    std_range=tuple(params.get("noise_std_range", (0.05, 0.15))),
                    p=float(params.get("noise_p", 0.5)),
                ),
                A.Blur(
                    blur_limit=int(params.get("blur_limit", 3)),
                    p=float(params.get("blur_p", 0.3)),
                ),
            ]
        )
    raise ValueError(f"Неподдерживаемый aug_table_id={aug_table_id}. Ожидаются 2, 6, 7.")


def augment_callable(pipeline: A.BasicTransform):
    def _apply(image: np.ndarray) -> np.ndarray:
        out = pipeline(image=image)
        return out["image"]

    return _apply


def describe_augment_variant(aug_table_id: int) -> str:
    mapping = {
        2: "Pad + RandomCrop + HorizontalFlip",
        6: "RandomBrightnessContrast",
        7: "GaussNoise + Blur",
    }
    return mapping.get(aug_table_id, f"variant_{aug_table_id}")
