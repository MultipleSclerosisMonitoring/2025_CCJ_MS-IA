"""
train_logistic_classifier.py

This script trains a logistic regression classifier on latent representations obtained from
an encoder model. It loads latent vectors and their corresponding binary labels, splits them
into training and testing subsets, trains a LogisticRegression model using scikit-learn, evaluates
its performance (accuracy, classification report, and optionally confusion matrix), and saves
the trained model to disk in .pkl format.

Usage:
    python train_logistic_classifier.py --latents X_latents.npy --labels y_labels.npy --output model.pkl [--plot]

Example:
    python train_logistic_classifier.py \
        --latents data/X_latents.npy \
        --labels data/y_labels.npy \
        --output output/logistic_model.pkl \
        --test_size 0.25 \
        --max_iter 500 \
        --seed 123 \
        --plot
"""

import argparse
import numpy as np
import joblib
import os
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


# CONFIGURATION

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def train_logistic_classifier(
    X_path,
    y_path,
    output_path,
    test_size=0.2,
    max_iter=1000,
    random_state=42,
    plot=False,
):
    """Train and evaluate a logistic regression classifier on latent representations.

    This function loads latent vectors and their corresponding binary labels, splits them into
    training and test sets, trains a logistic regression model, evaluates its performance, and
    optionally plots a confusion matrix. The trained model is saved to disk as a .pkl file.

    Args:
        X_path (str): Path to the .npy file containing latent representations (shape: [n_samples, n_features]).
        y_path (str): Path to the .npy file containing binary labels (shape: [n_samples,]).
        output_path (str): Path where the trained classifier will be saved (.pkl format).
        test_size (float, optional): Fraction of the data to use for testing. Defaults to 0.2.
        max_iter (int, optional): Maximum number of iterations for the logistic regression solver. Defaults to 1000.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        plot (bool, optional): Whether to display the confusion matrix. Defaults to False.

    Raises:
        ValueError: If the shapes of X and y do not match or have incorrect dimensions.
        Exception: If any other error occurs during training or saving.
    """
    logging.info("📦 Loading latent data and labels...")
    X = np.load(X_path)
    y = np.load(y_path)

    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} samples but y has {len(y)}")

    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features). Got: {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D (n_samples,). Got: {y.shape}")

    logging.info(f"✅ Loaded data: X.shape = {X.shape}, y.shape = {y.shape}")

    logging.info(
        f"📊 Splitting train/test (test_size={test_size}, random_state={random_state})..."
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logging.info("🧠 Training LogisticRegression...")
    clf = LogisticRegression(solver="lbfgs", max_iter=max_iter)
    clf.fit(X_train, y_train)

    logging.info("🧪 Evaluating on test set...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"✅ Accuracy on test set: {acc:.4f}")

    print("\n🧾 Classification report:\n")
    print(classification_report(y_test, y_pred, digits=4))

    if plot:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title("Confusion Matrix (Test)")
        plt.tight_layout()
        plt.show()

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"📁 Created output directory: {output_dir}")

    joblib.dump(clf, output_path)
    logging.info(f"✅ Classifier saved to: {output_path}")


# CLI
if __name__ == "__main__":
    # Command-line interface for training a logistic regression model using latent vectors.
    # Allows specification of file paths, training parameters, and optional confusion matrix visualization.
    parser = argparse.ArgumentParser(
        description="Train a logistic regression classifier on latent representations."
    )
    parser.add_argument(
        "--latents", required=True, help="Path to .npy file containing X_latents"
    )
    parser.add_argument(
        "--labels", required=True, help="Path to .npy file containing y_labels"
    )
    parser.add_argument(
        "--output", required=True, help="Path to save the trained classifier (.pkl)"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test set size fraction (default: 0.2)",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Maximum number of iterations (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument("--plot", action="store_true", help="Plot confusion matrix")

    args = parser.parse_args()

    try:
        train_logistic_classifier(
            X_path=args.latents,
            y_path=args.labels,
            output_path=args.output,
            test_size=args.test_size,
            max_iter=args.max_iter,
            random_state=args.seed,
            plot=args.plot,
        )
    except Exception as e:
        logging.error(f"❌ Error during training: {e}")
        raise
