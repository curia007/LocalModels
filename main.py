import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from src.data_loader import RemoteDatasetCatalog, fetch_remote_numerical_data
from src.model import NumericalMLP
from src.trainer import MLXTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create and train a local model via Apple MLX from remote numerical data."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="boston_housing",
        help="Catalog dataset name (e.g., 'boston_housing', 'california_housing', 'wine_quality_red', 'diabetes') or remote URL.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=None,
        help="Target column name (required if using custom remote URL).",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=None,
        help="Delimiter for CSV parsing (default: ',' or catalog default).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for AdamW optimizer (default: 0.001).",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[64, 32],
        help="Hidden layer dimensions (default: 64 32).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./saved_models/local_mlx_model",
        help="Destination path for saving local MLX model weights and config.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available remote numerical datasets and exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_datasets:
        print("Available remote numerical datasets:")
        for cfg in RemoteDatasetCatalog.list_datasets():
            print(f"  - {cfg.name:<20} : {cfg.description}")
            print(f"    URL: {cfg.url}")
            print(f"    Target: {cfg.target_col} | Type: {cfg.task_type}")
        return

    print("=" * 60)
    print(" Local MLX Model Creation Pipeline (Apple Silicon / MLX)")
    print("=" * 60)
    print(f"MLX Core Version: {mx.__version__}")
    print(f"Selected Remote Dataset: {args.dataset}")

    # 1. Fetch and preprocess numerical data remotely
    print("\n[Step 1] Fetching numerical data remotely and preprocessing...")
    data = fetch_remote_numerical_data(
        source=args.dataset,
        target_col=args.target_col,
        delimiter=args.delimiter,
    )

    in_features = data.X_train.shape[1]
    out_features = data.y_train.shape[1]
    print(f"Dataset summary:")
    print(f"  - Features ({in_features}): {data.feature_names}")
    print(f"  - Target ({out_features}): {data.target_name} ({data.task_type})")
    print(f"  - Train samples: {data.X_train.shape[0]}")
    print(f"  - Val samples:   {data.X_val.shape[0]}")
    print(f"  - Test samples:  {data.X_test.shape[0]}")

    # 2. Instantiate Local MLX Model
    print("\n[Step 2] Initializing local Neural Network via MLX...")
    loss_name = "bce" if data.task_type == "classification" else "mse"
    model = NumericalMLP(
        in_features=in_features,
        hidden_dims=args.hidden_dims,
        out_features=out_features,
        activation="relu",
    )
    print(f"Model architecture: NumericalMLP(in={in_features}, hidden={args.hidden_dims}, out={out_features})")

    # 3. Train Model Locally using MLX
    print("\n[Step 3] Training local model with MLX AdamW optimizer...")
    trainer = MLXTrainer(
        model=model,
        lr=args.lr,
        loss_name=loss_name,
    )
    history = trainer.fit(
        data=data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True,
    )

    # 4. Save Local Model
    print(f"\n[Step 4] Saving local model to '{args.save_path}'...")
    metadata = {
        "dataset_name": args.dataset,
        "feature_names": data.feature_names,
        "target_name": data.target_name,
        "task_type": data.task_type,
        "feature_means": data.feature_means,
        "feature_stds": data.feature_stds,
        "target_mean": data.target_mean,
        "target_std": data.target_std,
        "test_metrics": history.get("test_metrics", {}),
    }
    model.save_model(args.save_path, metadata=metadata)
    print(f"  ✓ Saved weights to: {args.save_path}.safetensors")
    print(f"  ✓ Saved configuration & metadata to: {args.save_path}.json")

    # 5. Reload Local Model and Run Verification / Inference
    print("\n[Step 5] Loading saved local model and running inference test...")
    loaded_model, loaded_meta = NumericalMLP.load_model(args.save_path)

    sample_count = min(3, data.X_test.shape[0])
    sample_X = data.X_test[:sample_count]
    sample_y = data.y_test[:sample_count]

    raw_preds = loaded_model.predict(sample_X)
    raw_preds_np = np.array(raw_preds)
    actual_y_np = np.array(sample_y)

    print("\nSample Predictions on Test Set:")
    for i in range(sample_count):
        if loaded_meta.get("task_type") == "regression":
            t_mean = loaded_meta.get("target_mean", 0.0)
            t_std = loaded_meta.get("target_std", 1.0)
            unscaled_pred = raw_preds_np[i, 0] * t_std + t_mean
            unscaled_actual = actual_y_np[i, 0] * t_std + t_mean
            print(f"  Sample #{i+1}: Predicted = {unscaled_pred:.2f} | Actual = {unscaled_actual:.2f}")
        else:
            prob = 1.0 / (1.0 + np.exp(-raw_preds_np[i, 0]))
            pred_class = int(prob >= 0.5)
            print(f"  Sample #{i+1}: Predicted Prob = {prob:.4f} (Class {pred_class}) | Actual = {int(actual_y_np[i, 0])}")

    print("\n" + "=" * 60)
    print(" Local MLX Model created and verified successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
