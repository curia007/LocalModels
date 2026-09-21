"""
Local MLX Model Package for Remote Numerical Data
"""

try:
    from .data_loader import fetch_remote_numerical_data, RemoteDatasetCatalog
    from .model import NumericalMLP
    from .trainer import MLXTrainer
except (ImportError, ValueError):
    from data_loader import fetch_remote_numerical_data, RemoteDatasetCatalog
    from model import NumericalMLP
    from trainer import MLXTrainer

__all__ = [
    "fetch_remote_numerical_data",
    "RemoteDatasetCatalog",
    "NumericalMLP",
    "MLXTrainer",
]
