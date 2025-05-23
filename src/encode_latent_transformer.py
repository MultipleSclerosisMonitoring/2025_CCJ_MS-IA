"""
encode_latent_transformer.py

Encodes longitudinal sensor data into latent representations using a trained Transformer encoder.

This script:
- Loads X (and optionally y, timestamps) from an HDF5 file.
- Applies a previously trained StandardScaler.
- Uses a trained encoder model to produce latent representations.
- Saves the encoded features (and metadata if available).

Usage:
    python encode_latent_transformer.py --input new_data.hdf5 \
        --encoder models/encoder_transformer.h5 \
        --scaler models/standard_scaler.pkl \
        --output encoded_latent/
"""

import argparse
from pathlib import Path
import numpy as np
import h5py
import joblib
from tensorflow.keras.models import load_model


def load_and_preprocess(hdf5_path: str, scaler_path: str):
    """Loads and normalizes X from a .hdf5 file using a saved StandardScaler.

    Args:
        hdf5_path (str): Path to .hdf5 file with dataset.
        scaler_path (str): Path to a joblib-serialized StandardScaler.

    Returns:
        tuple: (X_scaled, y, timestamps) — where y and timestamps may be None.
    """
    with h5py.File(hdf5_path, "r") as f:
        X = f["X"][:]
        y = f["y"][:] if "y" in f else None
        timestamps = f["timestamps"][:] if "timestamps" in f else None

    scaler = joblib.load(scaler_path)
    flat = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(flat).reshape(X.shape)

    return X_scaled, y, timestamps


def main():
    """Main routine to encode X using a trained Transformer encoder."""
    parser = argparse.ArgumentParser(
        description="Encode new sensor data into latent space."
    )
    parser.add_argument(
        "--input", required=True, help="Path to .hdf5 input file with X"
    )
    parser.add_argument(
        "--encoder", required=True, help="Path to encoder_transformer.h5 model"
    )
    parser.add_argument(
        "--scaler", required=True, help="Path to standard_scaler.pkl file"
    )
    parser.add_argument(
        "--output", default="latent_data", help="Folder to save encoded data"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for encoding"
    )
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    print("📦 Loading and preprocessing input...")
    X, y, timestamps = load_and_preprocess(args.input, args.scaler)

    print(f"✅ X loaded with shape: {X.shape}")

    print("🧠 Loading encoder model...")
    encoder = load_model(args.encoder)

    print("🔁 Encoding...")
    X_latent = encoder.predict(X, batch_size=args.batch_size, verbose=1)

    print(f"📐 Encoded latent shape: {X_latent.shape}")

    out_path = Path(args.output) / "X_latent_data.npz"
    np.savez_compressed(
        out_path,
        X_latent=X_latent,
        y=y if y is not None else [],
        timestamps=timestamps if timestamps is not None else [],
    )

    print(f"\n✅ Latent features saved to: {out_path}")


if __name__ == "__main__":
    main()
