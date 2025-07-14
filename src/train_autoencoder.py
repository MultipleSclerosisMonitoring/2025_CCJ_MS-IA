"""
train_autoencoder.py

Loads preprocessed sensor data from a .hdf5 file, applies normalization,
trains a Transformer autoencoder with configurable parameters, and saves models and training logs.

Usage:
    python train_transformer.py --input data.hdf5 --epochs 100 --batch_size 32 \
      --head_size 128 --num_heads 4 --ff_dim 256 --dropout 0.2 --num_blocks 3 --output models/
"""

import os
import random
import numpy as np
import argparse
import joblib
import h5py
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Add,
    LayerNormalization,
    GlobalAveragePooling1D,
    RepeatVector,
    TimeDistributed,
    MultiHeadAttention,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam


def set_seeds(seed=42):
    """Set seeds for reproducibility across NumPy, TensorFlow, and Python.

    Args:
        seed (int): Seed value to ensure reproducibility. Default is 42.
    """

    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_and_normalize_from_hdf5(hdf5_path: str, output_dir="models") -> np.ndarray:
    """Load and normalize 3D sensor data from an HDF5 file.

    StandardScaler is applied to flatten features across time, then reshaped back.

    Args:
        hdf5_path (str): Path to the HDF5 file containing dataset 'X'.
        output_dir (str): Directory where the scaler will be saved.

    Returns:
        np.ndarray: Normalized array of shape (samples, timesteps, features).
    """
    with h5py.File(hdf5_path, "r") as f:
        X = f["X"][:]

    assert len(X.shape) == 3, "❌ X must have shape (samples, timesteps, features)"
    print(f"✅ Loaded X from HDF5 with shape: {X.shape}")

    flat = X.reshape(-1, X.shape[-1])
    scaler = StandardScaler().fit(flat)
    X_scaled = scaler.transform(flat).reshape(X.shape)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, Path(output_dir) / "standard_scaler.pkl")

    return X_scaled


def transformer_encoder(inputs, head_size=64, num_heads=2, ff_dim=128, dropout=0.1):
    """Creates a single Transformer encoder block with multi-head attention and feed-forward layers.

    Args:
        inputs (tf.Tensor): Input tensor of shape (batch_size, timesteps, features).
        head_size (int): Dimensionality of each attention head.
        num_heads (int): Number of parallel attention heads.
        ff_dim (int): Dimensionality of the feed-forward layer.
        dropout (float): Dropout rate applied after attention and feed-forward layers.

    Returns:
        tf.Tensor: Output tensor after the transformer encoder block.
    """

    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(inputs.shape[-1])(ff)
    ff = Dropout(dropout)(ff)
    x = Add()([x, ff])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x


def build_transformer_autoencoder(
    timesteps,
    features,
    head_size=64,
    num_heads=2,
    ff_dim=128,
    num_blocks=2,
    dropout=0.1,
) -> Model:
    """Builds a Transformer-based autoencoder for time series data.

    The architecture includes a symmetric encoder-decoder with multi-head self-attention.

    Args:
        timesteps (int): Number of time steps in input sequences.
        features (int): Number of input features per time step.
        head_size (int): Size of each attention head.
        num_heads (int): Number of attention heads.
        ff_dim (int): Hidden layer size of the feed-forward block.
        num_blocks (int): Number of transformer blocks used in encoder and decoder.
        dropout (float): Dropout rate for attention and feed-forward layers.

    Returns:
        Model: Keras Model representing the autoencoder.
    """
    inputs = Input(shape=(timesteps, features))
    x = inputs
    for _ in range(num_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    encoded = GlobalAveragePooling1D(name="encoder_output")(x)
    x = RepeatVector(timesteps)(encoded)

    for _ in range(num_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    outputs = TimeDistributed(Dense(features, activation="linear"))(x)
    return Model(inputs, outputs)


def train_model_kfold(
    X: np.ndarray,
    output_dir: str = "models",
    epochs: int = 50,
    batch_size: int = 32,
    head_size: int = 64,
    num_heads: int = 2,
    ff_dim: int = 128,
    dropout: float = 0.1,
    num_blocks: int = 2,
    n_splits: int = 5,
    resume_from_fold: int = 1,
):
    """Train a Transformer autoencoder using K-Fold cross-validation.

    Args:
        X (np.ndarray): Input data of shape (samples, timesteps, features).
        output_dir (str): Base directory to save models and logs.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        head_size (int): Size of each attention head.
        num_heads (int): Number of attention heads.
        ff_dim (int): Size of feed-forward layer.
        dropout (float): Dropout rate.
        num_blocks (int): Number of transformer blocks.
        n_splits (int): Number of K-folds for cross-validation.
    """
    Path(output_dir).mkdir(exist_ok=True)
    timesteps, features = X.shape[1], X.shape[2]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        if fold < resume_from_fold:
            print(f"⏭️  Skipping fold {fold} (already completed)")
            continue

        print(f"\n--- Fold {fold}/{n_splits} ---")
        fold_dir = Path(output_dir) / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        X_train, X_val = X[train_idx], X[val_idx]

        # Save validation (test) indices and data for later supervised evaluation
        np.save(fold_dir / "test_indices.npy", val_idx)

        with h5py.File(fold_dir / "test_data.hdf5", "w") as f:
            f.create_dataset("X", data=X_val)
            f.create_dataset("indices", data=val_idx)

        model = build_transformer_autoencoder(
            timesteps, features, head_size, num_heads, ff_dim, num_blocks, dropout
        )
        model.compile(optimizer=Adam(0.001), loss="mse")

        # ... [todo lo anterior sin cambios] ...

        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(str(fold_dir / "best_model.keras"), save_best_only=True),
            CSVLogger(str(fold_dir / "training_log.csv")),
        ]

        history = model.fit(
            X_train,
            X_train,
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=callbacks,
        )

        model.save(fold_dir / "transformer_autoencoder.keras")
        encoder = Model(model.input, model.get_layer("encoder_output").output)
        encoder.save(fold_dir / "encoder_transformer.keras")
        np.save(fold_dir / "loss_history.npy", history.history)

        print("\n✅ All folds completed.")


def main():
    """
    Entry point for training a Transformer autoencoder on time series data using K-Fold cross-validation.

    This routine performs the following steps:
    1. Parses command-line arguments for model configuration and data paths.
    2. Sets random seeds to ensure reproducible training.
    3. Loads time series data from an HDF5 file and normalizes it using StandardScaler.
    4. Applies K-Fold cross-validation to train multiple autoencoders.
    5. Saves each fold's model, encoder, training log, and loss history.

    Command-line Arguments:
        --input (str): Path to the HDF5 file containing the dataset with 'X'.
        --epochs (int): Number of training epochs for each fold (default: 50).
        --batch_size (int): Batch size used during training (default: 32).
        --output (str): Directory where model outputs and logs are saved (default: "models").
        --head_size (int): Dimensionality of each attention head (default: 64).
        --num_heads (int): Number of attention heads in the Transformer (default: 2).
        --ff_dim (int): Dimensionality of feed-forward layers (default: 128).
        --dropout (float): Dropout rate used in Transformer blocks (default: 0.1).
        --num_blocks (int): Number of encoder and decoder Transformer blocks (default: 2).
        --n_splits (int): Number of folds for K-Fold cross-validation (default: 5).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="Path to HDF5 with 'X'"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument("--output", type=str, default="models", help="Output directory")
    parser.add_argument(
        "--head_size", type=int, default=64, help="Size of attention heads"
    )
    parser.add_argument(
        "--num_heads", type=int, default=2, help="Number of attention heads"
    )
    parser.add_argument(
        "--ff_dim", type=int, default=128, help="Feed-forward layer dimension"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--num_blocks", type=int, default=2, help="Number of transformer blocks"
    )
    parser.add_argument(
        "--n_splits", type=int, default=5, help="Number of K-Fold splits"
    )
    parser.add_argument(
        "--resume_from_fold",
        type=int,
        default=1,
        help="Fold number (1-based) to resume training from",
    )

    args = parser.parse_args()

    set_seeds()
    X = load_and_normalize_from_hdf5(args.input, output_dir=args.output)
    train_model_kfold(
        X,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        head_size=args.head_size,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        num_blocks=args.num_blocks,
        n_splits=args.n_splits,
        resume_from_fold=args.resume_from_fold,
    )


if __name__ == "__main__":
    main()
