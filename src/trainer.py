import math
from typing import Any, Callable, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

try:
    from .data_loader import PreprocessedData
    from .model import NumericalMLP
except (ImportError, ValueError):
    from data_loader import PreprocessedData
    from model import NumericalMLP


class MLXTrainer:
    """
    Trainer for MLX numerical models supporting regression and binary/multiclass classification.
    """

    def __init__(
        self,
        model: NumericalMLP,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_name: str = "mse",
    ):
        self.model = model
        self.lr = lr
        self.optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
        self.loss_name = loss_name.lower()

    def _loss_fn(self, model: NumericalMLP, x: mx.array, y: mx.array) -> mx.array:
        preds = model(x)
        if self.loss_name == "mse":
            return mx.mean((preds - y) ** 2)
        elif self.loss_name == "mae":
            return mx.mean(mx.abs(preds - y))
        elif self.loss_name == "huber":
            delta = 1.0
            diff = mx.abs(preds - y)
            is_small = diff <= delta
            small_loss = 0.5 * (diff**2)
            large_loss = delta * (diff - 0.5 * delta)
            return mx.mean(mx.where(is_small, small_loss, large_loss))
        elif self.loss_name in ("bce", "binary_cross_entropy"):
            probs = mx.sigmoid(preds)
            eps = 1e-7
            bce = - (y * mx.log(probs + eps) + (1.0 - y) * mx.log(1.0 - probs + eps))
            return mx.mean(bce)
        else:
            return mx.mean((preds - y) ** 2)

    def train_epoch(
        self,
        X: mx.array,
        y: mx.array,
        batch_size: int = 32,
    ) -> float:
        """Trains the model for one epoch over the dataset."""
        num_samples = X.shape[0]
        indices = np.random.permutation(num_samples)
        total_loss = 0.0
        num_batches = 0

        loss_and_grad_fn = nn.value_and_grad(self.model, self._loss_fn)

        for start_idx in range(0, num_samples, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            x_batch = X[mx.array(batch_idx)]
            y_batch = y[mx.array(batch_idx)]

            loss, grads = loss_and_grad_fn(self.model, x_batch, y_batch)
            self.optimizer.update(self.model, grads)
            mx.eval(self.model.parameters(), self.optimizer.state)

            total_loss += float(loss.item())
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def evaluate(
        self,
        X: mx.array,
        y: mx.array,
        task_type: str = "regression",
    ) -> Dict[str, float]:
        """Evaluates model performance metrics on given dataset."""
        preds = self.model(X)
        mx.eval(preds)

        loss = float(self._loss_fn(self.model, X, y).item())
        preds_np = np.array(preds)
        y_np = np.array(y)

        metrics = {"loss": loss}

        if task_type == "regression":
            mse = float(np.mean((preds_np - y_np) ** 2))
            rmse = float(math.sqrt(mse))
            mae = float(np.mean(np.abs(preds_np - y_np)))
            ss_tot = float(np.sum((y_np - np.mean(y_np)) ** 2))
            ss_res = float(np.sum((y_np - preds_np) ** 2))
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            metrics.update({"mse": mse, "rmse": rmse, "mae": mae, "r2": r2})
        else:
            if preds_np.shape[1] == 1:
                # Binary classification
                probs = 1.0 / (1.0 + np.exp(-preds_np))
                pred_labels = (probs >= 0.5).astype(np.float32)
                accuracy = float(np.mean(pred_labels == y_np))
            else:
                pred_labels = np.argmax(preds_np, axis=1, keepdims=True)
                accuracy = float(np.mean(pred_labels == y_np))
            metrics["accuracy"] = accuracy

        return metrics

    def fit(
        self,
        data: PreprocessedData,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 15,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs complete training and validation loops with early stopping.
        """
        history = {"train_loss": [], "val_loss": [], "val_metrics": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_weights = None

        if verbose:
            print(f"--- Starting local MLX training for {epochs} epochs ---")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(data.X_train, data.y_train, batch_size=batch_size)
            val_metrics = self.evaluate(data.X_val, data.y_val, task_type=data.task_type)
            val_loss = val_metrics["loss"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_metrics"].append(val_metrics)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs):
                metric_str = ", ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
                print(f"Epoch {epoch:03d}/{epochs} | Train Loss: {train_loss:.4f} | Val: {metric_str}")

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch}")
                    break

        test_metrics = self.evaluate(data.X_test, data.y_test, task_type=data.task_type)
        if verbose:
            print(f"--- Final Test Evaluation: ---")
            for k, v in test_metrics.items():
                print(f"  {k}: {v:.4f}")

        history["test_metrics"] = test_metrics
        return history
