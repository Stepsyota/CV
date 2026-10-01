from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from lab3.constants import INPUT_DIM, NUM_CLASSES


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
        return " – ".join(parts)


def init_linear_weights(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def build_model(cfg: dict) -> MLP:
    model_cfg = cfg["model"]
    model = MLP(
        hidden_dims=model_cfg["hidden_dims"],
        activation=model_cfg.get("activation", "relu"),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    max_params = int(cfg.get("max_params", 1_000_000))
    n_params = count_trainable_parameters(model)
    if n_params > max_params:
        raise ValueError(f"Модель имеет {n_params} параметров, лимит {max_params}")
    if model_cfg.get("init_weights", True):
        init_linear_weights(model)
    return model
