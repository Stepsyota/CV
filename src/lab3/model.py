from __future__ import annotations

from typing import Protocol, Sequence

import torch
import torch.nn as nn

from lab3.constants import INPUT_DIM, NUM_CLASSES


class ArchitectureModel(Protocol):
    def architecture_string(self) -> str: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: Sequence[int] = (512, 256, 128),
        num_classes: int = NUM_CLASSES,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
        self._activation_name = activation
        self._hidden_dims = list(hidden_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def architecture_string(self) -> str:
        parts: list[str] = [str(INPUT_DIM)]
        for dim in self._hidden_dims:
            parts.append(f"{dim} ({self._activation_name})")
        parts.append(str(NUM_CLASSES))
        return "MLP: " + " – ".join(parts)


class MixerBlock(nn.Module):
    """Token-mixing и channel-mixing MLP (Tolstikhin et al.), только Linear + norm + residual."""

    def __init__(
        self,
        num_patches: int,
        dim: int,
        token_mlp_dim: int,
        channel_mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mix = nn.Sequential(
            nn.Linear(num_patches, token_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_mlp_dim, num_patches),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mix = nn.Sequential(
            nn.Linear(dim, channel_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_mlp_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = y.transpose(1, 2)
        y = self.token_mix(y)
        y = y.transpose(1, 2)
        x = x + y
        y = self.norm2(x)
        y = self.channel_mix(y)
        return x + y


class MLPMixer(nn.Module):
    """
    MLP-Mixer для CIFAR-10 (32×32).
    Вход: flatten 3072 (как в data.py) или тензор B×3×32×32.
    Свёрток нет; mixing — только nn.Linear.
    """

    def __init__(
        self,
        image_size: int = 32,
        channels: int = 3,
        patch_size: int = 4,
        dim: int = 128,
        depth: int = 12,
        token_mlp_dim: int = 64,
        channel_mlp_dim: int = 256,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.channels = channels
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        self.patch_dim = channels * patch_size * patch_size
        self.dim = dim
        self.depth = depth

        self.patch_embed = nn.Linear(self.patch_dim, dim)
        self.blocks = nn.ModuleList(
            [
                MixerBlock(self.num_patches, dim, token_mlp_dim, channel_mlp_dim, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(x.size(0), self.channels, self.image_size, self.image_size)
        p = self.patch_size
        gh, gw = self.grid, self.grid
        x = x.reshape(x.size(0), self.channels, gh, p, gw, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        return x.reshape(x.size(0), self.num_patches, self.patch_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_patches(x)
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)

    def architecture_string(self) -> str:
        return (
            f"MLP-Mixer: patches {self.grid}×{self.grid} (P={self.patch_size}), "
            f"dim={self.dim}, depth={self.depth} → {NUM_CLASSES}"
        )


def init_linear_weights(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def _build_mlp(model_cfg: dict) -> MLP:
    return MLP(
        hidden_dims=model_cfg["hidden_dims"],
        activation=model_cfg.get("activation", "relu"),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )


def _build_mlp_mixer(model_cfg: dict) -> MLPMixer:
    return MLPMixer(
        patch_size=int(model_cfg.get("patch_size", 4)),
        dim=int(model_cfg.get("dim", 128)),
        depth=int(model_cfg.get("depth", 12)),
        token_mlp_dim=int(model_cfg.get("token_mlp_dim", 64)),
        channel_mlp_dim=int(model_cfg.get("channel_mlp_dim", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )


def build_model(cfg: dict) -> nn.Module:
    model_cfg = cfg["model"]
    model_type = str(model_cfg.get("type", "mlp_mixer")).lower()
    if model_type in {"mlp", "fc"}:
        model = _build_mlp(model_cfg)
    elif model_type in {"mlp_mixer", "mixer", "mlp-mixer"}:
        model = _build_mlp_mixer(model_cfg)
    else:
        raise ValueError(f"Unknown model.type: {model_type!r} (use mlp or mlp_mixer)")

    max_params = int(cfg.get("max_params", 1_000_000))
    n_params = count_trainable_parameters(model)
    if n_params > max_params:
        raise ValueError(f"Модель имеет {n_params} параметров, лимит {max_params}")
    if model_cfg.get("init_weights", True):
        init_linear_weights(model)
    return model
