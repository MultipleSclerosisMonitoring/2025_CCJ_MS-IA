"""
inference.py

This script performs inference on new time-series data using a trained encoder and a logistic
regression classifier. It accepts input files in .hdf5, .csv, or .xlsx format, segments the data,
applies the scaler used during training, extracts latent representations with a Keras encoder,
and predicts whether each segment corresponds to a 'walk' or 'no_walk' class using a trained
scikit-learn classifier.

The output is a CSV file containing the segment ID, estimated start time, prediction, and
mean of the latent vector.

Example:
    python inference.py \
        --input new_data.hdf5 \
        --encoder encoder_transformer.keras \
        --classifier latent_classifier.pkl \
        --scaler standard_scaler.pkl \
        --output predictions.csv
"""

import argparse
import pandas as pd
import numpy as np
import h5py
import joblib
import os
import logging
from tensorflow.keras.models import load_model

# GLOBAL CONFIGURATION
SEGMENT_LENGTH = 295
STRIDE = 50
SAMPLE_RATE = 50  # Hz

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(file_path):
    """Loads input data from .hdf5, .csv, or .xlsx files.

    Args:
        file_path (str): Path to the input file.

    Returns:
        np.ndarray: Loaded data as a 2D NumPy array (n_samples, n_features).

    Raises:
        KeyError: If 'data' key is not found in an HDF5 file.
        ValueError: If the file extension is not supported.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    logging.info(f"📥 Loading data from: {file_path}")

    if ext == ".hdf5":
        with h5py.File(file_path, "r") as f:
            key = "data"
            if key not in f:
                raise KeyError(f"Key '{key}' not found in HDF5 file.")
            return np.array(f[key])
    elif ext == ".csv":
        return pd.read_csv(file_path).values
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(file_path).values
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def segment_data(data, segment_len=SEGMENT_LENGTH, stride=STRIDE):
    """Segments time-series data into overlapping windows.

    Args:
        data (np.ndarray): 2D array of shape (n_samples, n_features).
        segment_len (int): Length of each segment (default: 295).
        stride (int): Step size between segment starts (default: 50).

    Returns:
        np.ndarray: 3D array of shape (n_segments, segment_len, n_features).

    Raises:
        ValueError: If data is not 2D or is shorter than one segment.
    """
    logging.info("🔪 Segmenting data...")

    if data.ndim != 2:
        raise ValueError("Data must be 2D: (n_samples, n_features)")
    if data.shape[0] < segment_len:
        raise ValueError(
            f"Insufficient data for segmentation: need at least {segment_len} rows"
        )

    segments = []
    for start in range(0, len(data) - segment_len + 1, stride):
        segment = data[start : start + segment_len]
        segments.append(segment)

    return np.array(segments)


def run_inference(input_path, encoder_path, classifier_path, scaler_path, output_path):
    """Runs inference on input data using an encoder and a logistic classifier.

    Args:
        input_path (str): Path to the input file (.hdf5, .csv, or .xlsx).
        encoder_path (str): Path to the Keras encoder model (.keras).
        classifier_path (str): Path to the trained logistic classifier (.pkl).
        scaler_path (str): Path to the fitted StandardScaler used during training (.pkl).
        output_path (str): Path where the prediction CSV will be saved.

    Raises:
        ValueError: If the feature dimensions of the scaler and input data do not match.
        Exception: For unexpected errors during inference.
    """
    raw_data = load_data(input_path)
    segments = segment_data(raw_data)

    logging.info("⚖️ Loading scaler...")
    scaler = joblib.load(scaler_path)
    expected_features = scaler.mean_.shape[0]
    if segments.shape[-1] != expected_features:
        raise ValueError(
            f"Scaler expects {expected_features} features, but got {segments.shape[-1]}"
        )

    logging.info("⚙️ Scaling segments...")
    flat_segments = segments.reshape(-1, segments.shape[-1])
    scaled_segments = scaler.transform(flat_segments).reshape(segments.shape)

    logging.info("📦 Loading encoder and classifier...")
    encoder = load_model(encoder_path, compile=False)
    classifier = joblib.load(classifier_path)

    logging.info("🤖 Generating latents and predictions...")
    latents = encoder.predict(scaled_segments)
    preds = classifier.predict(latents).astype(int)

    start_times = np.arange(len(preds)) * STRIDE / SAMPLE_RATE

    logging.info("💾 Saving predictions...")
    df = pd.DataFrame(
        {
            "segment_id": np.arange(len(preds)),
            "start_time_s": start_times,
            "prediction": preds,
            "latent_mean": latents.mean(axis=1),
        }
    )
    df.to_csv(output_path, index=False)
    logging.info(f"✅ Predictions saved to: {output_path}")


# CLI

if __name__ == "__main__":
    # Command-line interface to run inference using encoder + logistic classifier
    parser = argparse.ArgumentParser(
        description="Run inference on new data using an encoder and a logistic classifier."
    )
    parser.add_argument(
        "--input", required=True, help="Path to input file (.hdf5, .csv, .xlsx)"
    )
    parser.add_argument(
        "--encoder", required=True, help="Path to encoder model (.keras)"
    )
    parser.add_argument(
        "--classifier", required=True, help="Path to classifier model (.pkl)"
    )
    parser.add_argument("--scaler", required=True, help="Path to fitted scaler (.pkl)")
    parser.add_argument("--output", required=True, help="Path to output CSV file")

    args = parser.parse_args()

    try:
        run_inference(
            input_path=args.input,
            encoder_path=args.encoder,
            classifier_path=args.classifier,
            scaler_path=args.scaler,
            output_path=args.output,
        )
    except Exception as e:
        logging.error(f"❌ Inference failed: {e}")
        raise
