import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
import os
import re


def extract_segment_length(file_name):
    """Extract segment length (e.g. 150 from 'latents_len150_fold1.npz')"""
    match = re.search(r"len(\d+)", file_name)
    return f"len{match.group(1)}" if match else "unknown"


def load_latents(npz_path):
    """Load latent vectors and labels from a .npz file."""
    data = np.load(npz_path)
    return data["X_latent"], data["y"]


def evaluate_classifier(
    X, y, file_name, segment_suffix, test_size=0.2, save=False, title="Confusion Matrix"
):
    """Train and evaluate a classifier on latent space. Returns a DataFrame with all results."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n📊 Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True, digits=4)
    print(classification_report(y_test, y_pred, digits=4))

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    recall_macro = recall_score(y_test, y_pred, average="macro")

    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ F1-score (macro): {f1_macro:.4f}")
    print(f"✅ Recall (macro): {recall_macro:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(title)
    plt.tight_layout()
    if save:
        filename = f"{title.replace(' ', '_').lower()}_{segment_suffix}.png"
        plt.savefig(filename, dpi=300)
        print(f"📁 Saved confusion matrix to: {filename}")
    plt.show()

    # Prepare CSV data
    records = []
    n_samples, n_features = X.shape
    for label in sorted(report.keys()):
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        row = report[label]
        records.append(
            {
                "file_name": file_name,
                "segment": segment_suffix,
                "n_samples": n_samples,
                "n_features": n_features,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to .npz latent data")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size ratio")
    parser.add_argument(
        "--save", action="store_true", help="Save confusion matrix plot"
    )
    parser.add_argument(
        "--output_csv",
        default="results_latent_classifier.csv",
        help="Base name for output CSV",
    )
    args = parser.parse_args()

    file_name = os.path.basename(args.input)
    segment_suffix = extract_segment_length(file_name)
    X_latent, y = load_latents(args.input)
    print(f"✅ Loaded latent shape: {X_latent.shape}, Labels: {np.unique(y)}")

    results_df = evaluate_classifier(
        X_latent,
        y,
        file_name=file_name,
        segment_suffix=segment_suffix,
        test_size=args.test_size,
        save=args.save,
        title=f"Confusion Matrix ({file_name})",
    )

    # Append to CSV (create or update)
    output_file = args.output_csv.replace(".csv", f"_{segment_suffix}.csv")
    if os.path.exists(output_file):
        results_df.to_csv(output_file, mode="a", header=False, index=False)
    else:
        results_df.to_csv(output_file, index=False)

    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
