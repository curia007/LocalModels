import os
import shutil
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np

from src.data_loader import (
    DatasetConfig,
    PreprocessedData,
    RemoteDatasetCatalog,
    fetch_remote_numerical_data,
)
from src.model import NumericalMLP
from src.trainer import MLXTrainer


class TestLocalMLXModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_catalog(self):
        datasets = RemoteDatasetCatalog.list_datasets()
        self.assertGreater(len(datasets), 0)
        boston = RemoteDatasetCatalog.get("boston_housing")
        self.assertIsNotNone(boston)
        self.assertEqual(boston.target_col, "medv")

    def test_fetch_remote_data(self):
        # Fetch small remote dataset
        data = fetch_remote_numerical_data("boston_housing")
        self.assertIsInstance(data, PreprocessedData)
        self.assertIsInstance(data.X_train, mx.array)
        self.assertIsInstance(data.y_train, mx.array)
        self.assertEqual(data.X_train.ndim, 2)
        self.assertEqual(data.y_train.ndim, 2)
        self.assertGreater(len(data.feature_names), 0)
        self.assertEqual(data.task_type, "regression")

    def test_model_forward(self):
        in_dim = 10
        hidden_dims = [32, 16]
        out_dim = 1
        model = NumericalMLP(in_features=in_dim, hidden_dims=hidden_dims, out_features=out_dim)

        batch_size = 8
        dummy_input = mx.random.normal((batch_size, in_dim))
        output = model(dummy_input)

        self.assertEqual(output.shape, (batch_size, out_dim))

    def test_trainer_fit(self):
        # Synthetic dataset for fast deterministic training
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        # y = 2*x0 - 3*x1 + eps
        y = (2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.1 * np.random.randn(100)).astype(np.float32)[:, None]

        data = PreprocessedData(
            X_train=mx.array(X[:70]),
            y_train=mx.array(y[:70]),
            X_val=mx.array(X[70:85]),
            y_val=mx.array(y[70:85]),
            X_test=mx.array(X[85:]),
            y_test=mx.array(y[85:]),
            feature_names=[f"f_{i}" for i in range(5)],
            target_name="target",
            feature_means=[0.0] * 5,
            feature_stds=[1.0] * 5,
            target_mean=0.0,
            target_std=1.0,
            task_type="regression",
        )

        model = NumericalMLP(in_features=5, hidden_dims=[32, 16], out_features=1)
        trainer = MLXTrainer(model=model, lr=0.01, loss_name="mse")
        history = trainer.fit(data, epochs=30, batch_size=16, verbose=False)

        self.assertIn("train_loss", history)
        self.assertIn("val_loss", history)
        self.assertIn("test_metrics", history)
        # Loss should decrease
        self.assertLess(history["train_loss"][-1], history["train_loss"][0])

    def test_save_and_load_model(self):
        save_path = Path(self.test_dir) / "test_model"
        model = NumericalMLP(in_features=4, hidden_dims=[16, 8], out_features=1)
        metadata = {"creator": "mlx_tester", "task_type": "regression"}
        model.save_model(save_path, metadata=metadata)

        self.assertTrue(save_path.with_suffix(".safetensors").exists())
        self.assertTrue(save_path.with_suffix(".json").exists())

        loaded_model, loaded_meta = NumericalMLP.load_model(save_path)
        self.assertEqual(loaded_meta["creator"], "mlx_tester")

        dummy_x = mx.random.normal((2, 4))
        orig_out = model(dummy_x)
        loaded_out = loaded_model(dummy_x)

        np.testing.assert_allclose(np.array(orig_out), np.array(loaded_out), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
