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
from sklearn.model_selection import train_test_split
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
    """Set seeds for reproducibility across NumPy, TensorFlow, and Python."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_and_normalize_from_hdf5(hdf5_path: str, output_dir="models") -> np.ndarray:
    """Load sensor data from an HDF5 file and normalize using StandardScaler.

    Args:
        hdf5_path (str): Path to HDF5 file containing 'X' dataset.
        output_dir (str): Directory to save the scaler object.

    Returns:
        np.ndarray: Normalized data with shape (samples, timesteps, features).
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
    """Builds a single Transformer encoder block."""
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
    """Constructs the Transformer-based autoencoder model.

    Args:
        timesteps (int): Number of time steps in input.
        features (int): Number of features per time step.
        head_size (int): Size of each attention head.
        num_heads (int): Number of attention heads.
        ff_dim (int): Size of feed-forward hidden layer.
        num_blocks (int): Number of Transformer blocks in encoder/decoder.
        dropout (float): Dropout rate used in all layers.

    Returns:
        Model: Keras Model of the autoencoder.
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


def train_model(
    X,
    output_dir="models",
    epochs=50,
    batch_size=32,
    head_size=64,
    num_heads=2,
    ff_dim=128,
    dropout=0.1,
    num_blocks=2,
):
    """Train the autoencoder model and save models and logs."""
    timesteps, features = X.shape[1], X.shape[2]
    model = build_transformer_autoencoder(
        timesteps,
        features,
        head_size=head_size,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        num_blocks=num_blocks,
    )
    model.compile(optimizer=Adam(0.001), loss="mse")

    Path(output_dir).mkdir(exist_ok=True)

    X_train, X_val = train_test_split(X, test_size=0.1, random_state=42)

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(
            f"{output_dir}/best_transformer_autoencoder.h5", save_best_only=True
        ),
        CSVLogger(f"{output_dir}/training_log.csv"),
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

    model.save(f"{output_dir}/transformer_autoencoder.h5")
    encoder = Model(model.input, model.get_layer("encoder_output").output)
    encoder.save(f"{output_dir}/encoder_transformer.h5")
    np.save(f"{output_dir}/loss_history.npy", history.history)

    print("\n✅ Training completed and models saved.")


def main():
    """
    Main entry point for training a Transformer-based autoencoder on time series sensor data.

    This function:
    1. Defines and parses command-line arguments.
    2. Sets seeds for reproducibility.
    3. Loads and normalizes time series data from an HDF5 file using StandardScaler.
    4. Builds and trains a Transformer autoencoder using the specified hyperparameters.
    5. Saves the model, encoder, training log, and loss history to the output directory.

    Command-line arguments:
    --input         Path to HDF5 file containing dataset with 'X' array.
    --epochs        Number of training epochs (default: 50).
    --batch_size    Size of training batches (default: 32).
    --output        Output directory to save models and logs (default: models).
    --head_size     Size of each attention head (default: 64).
    --num_heads     Number of attention heads (default: 2).
    --ff_dim        Dimensionality of feed-forward layers (default: 128).
    --dropout       Dropout rate for all dropout layers (default: 0.1).
    --num_blocks    Number of Transformer encoder/decoder blocks (default: 2).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="Path to .hdf5 dataset with X"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument("--output", type=str, default="models", help="Output directory")
    parser.add_argument(
        "--head_size", type=int, default=64, help="Size of each attention head"
    )
    parser.add_argument(
        "--num_heads", type=int, default=2, help="Number of attention heads"
    )
    parser.add_argument(
        "--ff_dim", type=int, default=128, help="Feed-forward layer size"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--num_blocks", type=int, default=2, help="Number of transformer blocks"
    )

    args = parser.parse_args()

    set_seeds()
    X = load_and_normalize_from_hdf5(args.input, output_dir=args.output)
    train_model(
        X,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        head_size=args.head_size,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        num_blocks=args.num_blocks,
    )


if __name__ == "__main__":
    main()
