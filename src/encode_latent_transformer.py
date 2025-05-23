"""
encode_latent_transformer.py

Uses a trained Transformer encoder model to generate latent representations from time-series
sensor data. Loads X and y from .npy files, applies the encoder, and saves the encoded output.

Example:
    python encode_latent_transformer.py --X data/X_balanced.npy --y data/y_balanced.npy \
           --encoder models/encoder_transformer.h5 --output latent_data/
"""

import numpy as np
import argparse
from pathlib import Path
from tensorflow.keras.models import load_model


def main():
    """
    Main function to encode sensor data using a pretrained Transformer encoder.

    CLI Arguments:
        --X (str): Path to X_balanced.npy containing sensor input data.
        --y (str): Path to y_balanced.npy containing class labels.
        --encoder (str): Path to a .h5 file containing the trained Transformer encoder.
        --output (str): Output directory to save encoded data (default: "data_model").
        --batch_size (int): Batch size to use during prediction (default: 32).

    The function:
        - Loads input features and labels.
        - Loads a trained encoder model.
        - Applies the encoder to extract latent features.
        - Saves latent representations and corresponding labels as .npy files.
    """
    parser = argparse.ArgumentParser(
        description="Encode sensor chunks using trained transformer encoder."
    )
    parser.add_argument("--X", required=True, help="Path to X_balanced.npy")
    parser.add_argument("--y", required=True, help="Path to y_balanced.npy")
    parser.add_argument(
        "--encoder", required=True, help="Path to encoder_transformer.h5"
    )
    parser.add_argument(
        "--output", default="data_model", help="Folder to save latent features"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for encoding"
    )
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    print("📦 Loading data...")
    X = np.load(args.X)
