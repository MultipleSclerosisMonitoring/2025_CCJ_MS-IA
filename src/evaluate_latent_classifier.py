"""
Evaluate classifier performance on latent representations extracted from Transformer encoders.

This script:
- Iterates over multiple cross-validation folds from a given model directory.
- Loads test data and the corresponding encoder model per fold.
- Extracts latent representations using the encoder.
- Trains and evaluates a logistic regression classifier on the latent space.
- Computes and saves classification metrics, including confusion matrices.
- Stores all results in a CSV file for further analysis.

Usage:
    python evaluate_latent_classifier.py \
        --model_dir models_len150 \
        --original_dataset data_balanced/dataset_balanced_len150_50Hz.hdf5 \
        --output_csv results_latent_classifier.csv \
        --output_dir evaluation/eval_len150 \
        --save
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    recall_score,
    accuracy_score,
)


def extract_segment_length(path):
    """Extract the segment length string from a file or folder name.

    Args:
        path (str or Path): The file or folder name containing the segment length (e.g. 'len150').

    Returns:
        str: The extracted segment identifier (e.g. 'len150') or 'unknown' if not found.
    """
    match = re.search(r"len(\d+)", str(path))
    return f"len{match.group(1)}" if match else "unknown"


def evaluate_classifier(
    X_latent, y_true, file_name, segment_suffix, fold_id, output_dir=".", save=False
):
    """Train and evaluate a classifier on the latent space for one fold.

    Args:
        X_latent (np.ndarray): Latent representations of shape (n_samples, n_features).
        y_true (np.ndarray): True class labels for the test set.
        file_name (str): Name of the original dataset file.
        segment_suffix (str): Identifier for the segment length (e.g. 'len150').
        fold_id (str): Fold identifier (e.g. '1').
        output_dir (str): Path to save confusion matrix images.
        save (bool): Whether to save the confusion matrix plot as PNG.

    Returns:
        pd.DataFrame: A dataframe with performance metrics (F1, recall, accuracy, support) per class.
    """
    # Split latent space into train/validation for classification
    X_train, X_val, y_train, y_val = train_test_split(
        X_latent, y_true, test_size=0.3, random_state=42, stratify=y_true
    )

    # Train classifier
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    # Metrics
    report = classification_report(y_val, y_pred, output_dict=True, digits=4)
    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average="macro")
    recall_macro = recall_score(y_val, y_pred, average="macro")

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix Fold {fold_id} ({segment_suffix})")
    plt.tight_layout()
    if save:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(
            output_dir, f"confusion_matrix_{segment_suffix}_fold{fold_id}.png"
        )
        plt.savefig(filename, dpi=300)
        print(f"📁 Saved confusion matrix to: {filename}")
    plt.close()

    # Format results
    records = []
    for label in sorted(report.keys()):
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        row = report[label]
        records.append(
            {
                "file_name": file_name,
                "segment": segment_suffix,
                "fold": fold_id,
                "n_samples": X_latent.shape[0],
                "n_features": X_latent.shape[1],
                "accuracy": acc,
                "f1_macro": f1_macro,
                "recall_macro": recall_macro,
                "class": label,
                "f1_score": row["f1-score"],
                "recall": row["recall"],
                "support": row["support"],
            }
        )
    return pd.DataFrame(records)


def main():
    """Main routine to evaluate all folds in a model directory.

    This function:
    - Loads the original dataset to retrieve full class labels.
    - Iterates over each fold directory (e.g. 'fold_0', 'fold_1', ...).
    - Loads the encoder model and test data for the fold.
    - Extracts latent representations and evaluates a classifier.
    - Aggregates and saves the results to a CSV file.

    Command-line arguments:
        --model_dir: Path to folder with fold subdirectories and encoder models.
        --original_dataset: Path to the original balanced dataset (.hdf5).
        --output_csv: Name of the output CSV file.
        --output_dir: Directory where results will be stored.
        --save: If set, saves confusion matrix plots per fold.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", required=True, help="Path to models_lenXXX folder"
    )
    parser.add_argument(
        "--original_dataset", required=True, help="Path to original balanced HDF5"
    )
    parser.add_argument(
        "--output_csv",
        default="results_latent_classifier.csv",
        help="Output CSV base name",
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory to save confusion matrices and CSV (default: current directory)",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save confusion matrix plots"
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    segment_suffix = extract_segment_length(model_dir)
    file_name = os.path.basename(args.original_dataset)

    # Load full labels
    with h5py.File(args.original_dataset, "r") as f:
        y_full = f["y"][:]

    all_records = []

    for fold_dir in sorted(model_dir.glob("fold_*")):
        fold_id = fold_dir.name.split("_")[1]
        encoder_path = fold_dir / "encoder_transformer.keras"
        test_data_path = fold_dir / "test_data.hdf5"
        test_idx_path = fold_dir / "test_indices.npy"

        if not (
            encoder_path.exists() and test_data_path.exists() and test_idx_path.exists()
        ):
            print(f"❌ Skipping {fold_dir.name}, missing files.")
            continue

        print(f"\n🔍 Processing {fold_dir.name}...")

        # Load model
        encoder = tf.keras.models.load_model(encoder_path)

        # Load test data
        with h5py.File(test_data_path, "r") as f:
            X_test = f["X"][:]
        indices = np.load(test_idx_path)
        y_test = y_full[indices]

        # Encode
        X_latent = encoder.predict(X_test)

        # Evaluate
        df_fold = evaluate_classifier(
            X_latent,
            y_test,
            file_name,
            segment_suffix,
            fold_id,
            output_dir=args.output_dir,
            save=args.save,
        )
        all_records.append(df_fold)

    # Save all results
    if all_records:
        os.makedirs(args.output_dir, exist_ok=True)
        final_df = pd.concat(all_records, ignore_index=True)
        csv_name = os.path.join(
            args.output_dir, args.output_csv.replace(".csv", f"_{segment_suffix}.csv")
        )
        final_df.to_csv(csv_name, index=False)
        print(f"\n✅ All fold results saved to: {csv_name}")
    else:
        print("❌ No results to save.")


if __name__ == "__main__":
    main()
