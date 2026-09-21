import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn


class NumericalMLP(nn.Module):
    """
    Configurable Multi-Layer Perceptron model for numerical prediction (regression & classification)
    built with Apple MLX.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: Optional[List[int]] = None,
        out_features: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.out_features = out_features
        self.activation_name = activation
        self.dropout_rate = dropout

        activation_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
        }
        act_cls = activation_map.get(activation.lower(), nn.ReLU)

        layers = []
        curr_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            curr_dim = h_dim

        layers.append(nn.Linear(curr_dim, out_features))
        self.network = nn.Sequential(*layers)

    def __call__(self, x: mx.array) -> mx.array:
        return self.network(x)

    def predict(self, x: Union[mx.array, List[List[float]]]) -> mx.array:
        """Runs forward evaluation without training dropout."""
        if not isinstance(x, mx.array):
            x = mx.array(x, dtype=mx.float32)
        if x.ndim == 1:
            x = x[None, :]
        return self(x)

    def save_model(
        self,
        save_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Saves MLX model weights and model metadata to disk.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save weights in safetensors format
        weights_file = save_path.with_suffix(".safetensors")
        self.save_weights(str(weights_file))

        # Save model configuration and metadata
        config_file = save_path.with_suffix(".json")
        info = {
            "in_features": self.in_features,
            "hidden_dims": self.hidden_dims,
            "out_features": self.out_features,
            "activation": self.activation_name,
            "dropout": self.dropout_rate,
            "weights_file": weights_file.name,
            "metadata": metadata or {},
        }
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

    @classmethod
    def load_model(cls, model_path: Union[str, Path]) -> "Tuple[NumericalMLP, Dict[str, Any]]":
        """
        Loads MLX model and metadata from disk.
        """
        model_path = Path(model_path)
        if model_path.suffix == ".safetensors":
            config_file = model_path.with_suffix(".json")
            weights_file = model_path
        elif model_path.suffix == ".json":
            config_file = model_path
            weights_file = model_path.with_suffix(".safetensors")
        else:
            config_file = model_path.with_suffix(".json")
            weights_file = model_path.with_suffix(".safetensors")

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        if not weights_file.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_file}")

        with open(config_file, "r", encoding="utf-8") as f:
            info = json.load(f)

        model = cls(
            in_features=info["in_features"],
            hidden_dims=info["hidden_dims"],
            out_features=info["out_features"],
            activation=info.get("activation", "relu"),
            dropout=info.get("dropout", 0.0),
        )
        model.load_weights(str(weights_file))
        return model, info.get("metadata", {})
