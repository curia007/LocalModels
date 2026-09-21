import io
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
import pandas as pd


@dataclass
class DatasetConfig:
    name: str
    url: str
    target_col: str
    delimiter: str = ","
    description: str = ""
    task_type: str = "regression"


REMOTE_DATASETS: Dict[str, DatasetConfig] = {
    "boston_housing": DatasetConfig(
        name="boston_housing",
        url="https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
        target_col="medv",
        delimiter=",",
        description="Boston Housing Prices dataset (numerical regression)",
        task_type="regression",
    ),
    "wine_quality_red": DatasetConfig(
        name="wine_quality_red",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        target_col="quality",
        delimiter=";",
        description="Wine Quality (Red) physicochemical properties dataset",
        task_type="regression",
    ),
    "california_housing": DatasetConfig(
        name="california_housing",
        url="https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv",
        target_col="median_house_value",
        delimiter=",",
        description="California Housing prices based on 1990 census",
        task_type="regression",
    ),
    "diabetes": DatasetConfig(
        name="diabetes",
        url="https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
        target_col="Outcome",
        delimiter=",",
        description="Pima Indians Diabetes numerical dataset",
        task_type="classification",
    ),
}


class RemoteDatasetCatalog:
    """Catalog helper to list and retrieve remote numerical datasets."""

    @classmethod
    def list_datasets(cls) -> List[DatasetConfig]:
        return list(REMOTE_DATASETS.values())

    @classmethod
    def get(cls, name: str) -> Optional[DatasetConfig]:
        return REMOTE_DATASETS.get(name.lower())


@dataclass
class PreprocessedData:
    X_train: mx.array
    y_train: mx.array
    X_val: mx.array
    y_val: mx.array
    X_test: mx.array
    y_test: mx.array
    feature_names: List[str]
    target_name: str
    feature_means: List[float]
    feature_stds: List[float]
    target_mean: float
    target_std: float
    task_type: str


def fetch_remote_numerical_data(
    source: Union[str, DatasetConfig],
    target_col: Optional[str] = None,
    delimiter: Optional[str] = None,
    task_type: str = "regression",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    normalize: bool = True,
) -> PreprocessedData:
    """
    Fetches a remote numerical dataset via URL or catalog name,
    cleans missing values, standardizes numerical features,
    and returns MLX array splits ready for training.
    """
    if isinstance(source, DatasetConfig):
        url = source.url
        target = target_col or source.target_col
        sep = delimiter or source.delimiter
        task_type = source.task_type
    elif source in REMOTE_DATASETS:
        config = REMOTE_DATASETS[source]
        url = config.url
        target = target_col or config.target_col
        sep = delimiter or config.delimiter
        task_type = config.task_type
    else:
        url = source
        if not target_col:
            raise ValueError("target_col must be provided when passing a custom remote URL.")
        target = target_col
        sep = delimiter or ","

    # Fetch remote data over HTTP/HTTPS
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        raw_bytes = response.read()

    df = pd.read_csv(io.BytesIO(raw_bytes), sep=sep)

    # Validate target column
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in remote dataset. Available columns: {list(df.columns)}")

    # Extract target
    y_raw = df[target]
    X_raw = df.drop(columns=[target])

    # Convert non-numeric categorical columns via one-hot encoding if any
    X_raw = pd.get_dummies(X_raw, drop_first=True)

    # Drop any rows with NaN / infinite values
    combined = pd.concat([X_raw, y_raw], axis=1).dropna()
    X_clean = combined.drop(columns=[target])
    y_clean = combined[target]

    feature_names = list(X_clean.columns)
    target_name = target

    X_np = X_clean.to_numpy(dtype=np.float32)
    y_np = y_clean.to_numpy(dtype=np.float32)

    if y_np.ndim == 1:
        y_np = y_np[:, np.newaxis]

    # Shuffle and split
    np.random.seed(seed)
    num_samples = len(X_np)
    indices = np.random.permutation(num_samples)

    test_count = int(num_samples * test_ratio)
    val_count = int(num_samples * val_ratio)
    train_count = num_samples - test_count - val_count

    train_idx = indices[:train_count]
    val_idx = indices[train_count : train_count + val_count]
    test_idx = indices[train_count + val_count :]

    X_train_raw = X_np[train_idx]
    y_train_raw = y_np[train_idx]

    X_val_raw = X_np[val_idx]
    y_val_raw = y_np[val_idx]

    X_test_raw = X_np[test_idx]
    y_test_raw = y_np[test_idx]

    # Feature standardization
    if normalize:
        feature_means = np.mean(X_train_raw, axis=0)
        feature_stds = np.std(X_train_raw, axis=0)
        # Avoid division by zero for constant features
        feature_stds = np.where(feature_stds == 0, 1.0, feature_stds)

        X_train_proc = (X_train_raw - feature_means) / feature_stds
        X_val_proc = (X_val_raw - feature_means) / feature_stds
        X_test_proc = (X_test_raw - feature_means) / feature_stds

        if task_type == "regression":
            target_mean = float(np.mean(y_train_raw))
            target_std = float(np.std(y_train_raw))
            if target_std == 0:
                target_std = 1.0
            y_train_proc = (y_train_raw - target_mean) / target_std
            y_val_proc = (y_val_raw - target_mean) / target_std
            y_test_proc = (y_test_raw - target_mean) / target_std
        else:
            target_mean = 0.0
            target_std = 1.0
            y_train_proc = y_train_raw
            y_val_proc = y_val_raw
            y_test_proc = y_test_raw
    else:
        feature_means = np.zeros(X_np.shape[1], dtype=np.float32)
        feature_stds = np.ones(X_np.shape[1], dtype=np.float32)
        target_mean = 0.0
        target_std = 1.0
        X_train_proc = X_train_raw
        X_val_proc = X_val_raw
        X_test_proc = X_test_raw
        y_train_proc = y_train_raw
        y_val_proc = y_val_raw
        y_test_proc = y_test_raw

    return PreprocessedData(
        X_train=mx.array(X_train_proc),
        y_train=mx.array(y_train_proc),
        X_val=mx.array(X_val_proc),
        y_val=mx.array(y_val_proc),
        X_test=mx.array(X_test_proc),
        y_test=mx.array(y_test_proc),
        feature_names=feature_names,
        target_name=target_name,
        feature_means=feature_means.tolist(),
        feature_stds=feature_stds.tolist(),
        target_mean=target_mean,
        target_std=target_std,
        task_type=task_type,
    )
