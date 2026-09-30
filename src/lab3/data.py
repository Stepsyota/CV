from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as tv_datasets

from lab3.constants import LABEL_TO_IDX, NUM_CLASSES


def resolve_train_image_dir(data_root: str | Path) -> Path:
    root = Path(data_root)
    nested = root / "train" / "train"
    flat = root / "train"
    if nested.is_dir() and any(nested.glob("*.png")):
        return nested
    if flat.is_dir() and any(flat.glob("*.png")):
        return flat
    raise FileNotFoundError(
        f"Не найдены PNG в {nested} или {flat}. Проверьте распаковку train.7z."
    )


def load_kaggle_train_index(data_root: str | Path) -> pd.DataFrame:
    root = Path(data_root)
    labels_path = root / "trainLabels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Нет файла {labels_path}")
    df = pd.read_csv(labels_path)
    if "id" not in df.columns or "label" not in df.columns:
        raise ValueError("trainLabels.csv должен содержать колонки id, label")
    df["target"] = df["label"].map(LABEL_TO_IDX)
    if df["target"].isna().any():
        bad = df.loc[df["target"].isna(), "label"].unique()[:5]
        raise ValueError(f"Неизвестные классы в CSV: {bad}")
    return df


class KaggleCifarTrainDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        frame: pd.DataFrame,
        transform: Callable[[np.ndarray], torch.Tensor] | None = None,
        augment: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.image_dir = image_dir
        self.ids = frame["id"].astype(int).tolist()
        self.targets = frame["target"].astype(int).tolist()
        self.transform = transform
        self.augment = augment

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_id = self.ids[index]
        path = self.image_dir / f"{image_id}.png"
        image = np.array(Image.open(path).convert("RGB"))
        if self.augment is not None:
            image = self.augment(image)
        if self.transform is None:
            raise ValueError("transform is required")
        tensor = self.transform(image)
        return tensor, self.targets[index]


class TorchvisionCifarTrainSubset(Dataset):
    """Подмножество официального CIFAR-10 train (50k) с тем же split, что и Kaggle train/val."""

    def __init__(
        self,
        root: str | Path,
        indices: list[int] | np.ndarray,
        transform: Callable[[np.ndarray], torch.Tensor],
        augment: Callable[[np.ndarray], np.ndarray] | None = None,
        download: bool = True,
    ) -> None:
        self._inner = tv_datasets.CIFAR10(
            root=str(root),
            train=True,
            download=download,
        )
        self.indices = [int(i) for i in indices]
        self.transform = transform
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        idx = self.indices[index]
        image, target = self._inner[idx]
        image_np = np.array(image)
        if self.augment is not None:
            image_np = self.augment(image_np)
        return self.transform(image_np), int(target)


class TorchvisionCifarTestDataset(Dataset):
    """Официальный test split CIFAR-10 (10k, с метками) — Kaggle test без labels."""

    def __init__(
        self,
        root: str | Path,
        transform: Callable[[np.ndarray], torch.Tensor],
        download: bool = True,
    ) -> None:
        self._inner = tv_datasets.CIFAR10(
            root=str(root),
            train=False,
            download=download,
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, target = self._inner[index]
        image_np = np.array(image)
        return self.transform(image_np), int(target)


def make_tensor_transform(mean: list[float], std: list[float], flatten: bool = True):
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)

    def _transform(image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - mean_t) / std_t
        if flatten:
            return tensor.reshape(-1)
        return tensor

    return _transform


def build_dataloaders(
    cfg: dict,
    augment_train: Callable[[np.ndarray], np.ndarray] | None,
    download_test: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_source = str(cfg.get("data_source", "kaggle")).lower()
    if data_source == "torchvision":
        return _build_dataloaders_torchvision(cfg, augment_train, download=download_test)
    if data_source != "kaggle":
        raise ValueError(f"Unknown data_source: {data_source!r} (use kaggle or torchvision)")

    seed = int(cfg["seed"])
    batch_size = int(cfg["batch_size"])
    val_size = int(cfg["val_size"])

    mean = cfg["normalize"]["mean"]
    std = cfg["normalize"]["std"]
    transform = make_tensor_transform(mean, std, flatten=True)

    df = load_kaggle_train_index(cfg["data_root"])
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=df["target"],
    )

    image_dir = resolve_train_image_dir(cfg["data_root"])
    train_ds = KaggleCifarTrainDataset(image_dir, train_df, transform=transform, augment=augment_train)
    val_ds = KaggleCifarTrainDataset(image_dir, val_df, transform=transform, augment=None)

    test_ds = TorchvisionCifarTestDataset(
        cfg["torchvision_data_root"],
        transform=transform,
        download=download_test,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": False,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def _build_dataloaders_torchvision(
    cfg: dict,
    augment_train: Callable[[np.ndarray], np.ndarray] | None,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    seed = int(cfg["seed"])
    batch_size = int(cfg["batch_size"])
    val_size = int(cfg["val_size"])
    root = cfg["torchvision_data_root"]

    mean = cfg["normalize"]["mean"]
    std = cfg["normalize"]["std"]
    transform = make_tensor_transform(mean, std, flatten=True)

    probe = tv_datasets.CIFAR10(root=str(root), train=True, download=download)
    targets = np.array(probe.targets)
    indices = np.arange(len(probe))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        random_state=seed,
        stratify=targets,
    )

    train_ds = TorchvisionCifarTrainSubset(
        root, train_idx, transform=transform, augment=augment_train, download=download
    )
    val_ds = TorchvisionCifarTrainSubset(root, val_idx, transform=transform, augment=None, download=False)
    test_ds = TorchvisionCifarTestDataset(root, transform=transform, download=download)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": False,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
