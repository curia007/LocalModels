# LocalModels

**LocalModels** is a lightweight, high-performance machine learning pipeline built on [Apple MLX](https://github.com/ml-explore/mlx). It provides end-to-end capabilities for fetching remote numerical datasets, preprocessing and standardizing tabular data, training configurable Multi-Layer Perceptrons (MLPs) on Apple Silicon, serializing model weights via Safetensors, and running local inference.

---

## Key Features

- **Apple Silicon Acceleration**: Native, fast training and inference using Apple MLX (`mlx.core`, `mlx.nn`, `mlx.optimizers`).
- **Tabular Data Loading & Preprocessing**: Remote dataset download, automatic cleaning of missing values, train/validation/test splitting, and feature/target standardization.
- **Built-in Dataset Catalog**: Out-of-the-box support for popular benchmark datasets (`boston_housing`, `california_housing`, `wine_quality_red`, `diabetes`) as well as custom remote CSV URLs.
- **Configurable MLP Architecture**: Modular `NumericalMLP` supporting custom hidden dimensions, activations (`relu`, `gelu`, `silu`, `tanh`, `leaky_relu`), and dropout regularization.
- **Comprehensive Trainer**: `MLXTrainer` with AdamW optimizer, support for regression (MSE, MAE, Huber) and classification (Binary Cross-Entropy), metric computation (MSE, RMSE, MAE, R², Accuracy), and early stopping.
- **Safe Model Persistence**: Weight serialization in `.safetensors` format with associated architecture and scaler metadata stored in `.json`.
- **Command-Line Interface**: CLI tool (`main.py`) for training, saving, and verifying models with sample inference.

---

## Project Structure

```
LocalModels/
├── LICENSE
├── README.md
├── main.py                      # CLI entrypoint for training & evaluation
├── saved_models/                # Saved weights (.safetensors) and configs (.json)
│   ├── local_mlx_model.json
│   └── local_mlx_model.safetensors
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Dataset catalog, download, scaling, and splitting
│   ├── model.py                 # NumericalMLP module & safetensors save/load
│   └── trainer.py               # MLXTrainer, loss functions, metrics & early stopping
└── tests/
    └── test_pipeline.py         # Unit tests for data loading, model, and training
```

---

## Requirements & Installation

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4 or later recommended)
- Python 3.9+

### Installation

Clone the repository and install required dependencies:

```bash
git clone https://github.com/user/LocalModels.git
cd LocalModels
python3 -m venv .venv
source .venv/bin/activate
pip install mlx numpy pandas
```

---

## Usage

### 1. Command-Line Interface (`main.py`)

#### List Available Catalog Datasets

```bash
python3 main.py --list-datasets
```

Output:
```text
Available remote numerical datasets:
  - boston_housing       : Boston Housing Prices dataset (numerical regression)
  - wine_quality_red     : Wine Quality (Red) physicochemical properties dataset
  - california_housing   : California Housing prices based on 1990 census
  - diabetes             : Pima Indians Diabetes numerical dataset
```

#### Train a Regression Model

Train a model on the Boston Housing dataset with custom architecture:

```bash
python3 main.py \
  --dataset boston_housing \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.001 \
  --hidden-dims 64 32 \
  --save-path ./saved_models/local_mlx_model
```

#### Train a Classification Model

Train a binary classification model on the Diabetes dataset:

```bash
python3 main.py \
  --dataset diabetes \
  --epochs 60 \
  --batch-size 32 \
  --lr 0.001 \
  --hidden-dims 64 32 \
  --save-path ./saved_models/diabetes_model
```

#### Train with a Custom Remote CSV URL

```bash
python3 main.py \
  --dataset "https://example.com/dataset.csv" \
  --target-col "target_column" \
  --delimiter "," \
  --epochs 50
```

### CLI Arguments Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | `str` | `boston_housing` | Catalog dataset name or remote CSV URL. |
| `--target-col` | `str` | `None` | Target column name (required for custom URLs). |
| `--delimiter` | `str` | `None` | CSV delimiter (defaults to catalog setting or `,`). |
| `--epochs` | `int` | `50` | Maximum number of training epochs. |
| `--batch-size` | `int` | `32` | Training batch size. |
| `--lr` | `float` | `0.001` | Learning rate for AdamW optimizer. |
| `--hidden-dims` | `int ...` | `64 32` | Hidden layer dimensions. |
| `--save-path` | `str` | `./saved_models/local_mlx_model` | Base path (without extension) for model weights and metadata. |
| `--list-datasets` | flag | `False` | Lists available catalog datasets and exits. |

---

## Python API Usage

You can also use LocalModels programmatically in your Python workflows:

```python
from src.data_loader import fetch_remote_numerical_data
from src.model import NumericalMLP
from src.trainer import MLXTrainer

# 1. Fetch and preprocess data
data = fetch_remote_numerical_data("california_housing")

# 2. Instantiate MLP model
model = NumericalMLP(
    in_features=data.X_train.shape[1],
    hidden_dims=[128, 64],
    out_features=data.y_train.shape[1],
    activation="relu",
    dropout=0.1,
)

# 3. Train with early stopping
trainer = MLXTrainer(model=model, lr=1e-3, loss_name="mse")
history = trainer.fit(data, epochs=40, batch_size=64)

# 4. Save model and metadata (safetensors + json)
metadata = {
    "feature_names": data.feature_names,
    "feature_means": data.feature_means,
    "feature_stds": data.feature_stds,
    "target_mean": data.target_mean,
    "target_std": data.target_std,
    "task_type": data.task_type,
}
model.save_model("./saved_models/california_model", metadata=metadata)

# 5. Reload model for inference
loaded_model, loaded_meta = NumericalMLP.load_model("./saved_models/california_model")
predictions = loaded_model.predict(data.X_test[:5])
```

---

## Running Tests

Execute the unit test suite with `unittest`:

```bash
python3 -m unittest discover tests
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
