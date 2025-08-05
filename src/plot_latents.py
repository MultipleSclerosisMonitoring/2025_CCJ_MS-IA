"""
plot_latents.py

Visualize latent representations from an autoencoder using dimensionality reduction.

This script:
- Loads latent feature vectors and labels from a compressed .npz file.
- Computes the Silhouette Score using cosine distance (if labels are available).
- Applies dimensionality reduction (UMAP or PCA) to project the latent space to 2D.
- Displays a scatter plot of the embedded data, optionally saving the plot.

Usage:
    python plot_latents.py --input path/to/X_latent_data.npz --method umap --save
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from pathlib import Path
import re

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def load_latents(npz_path):
    """Load latent vectors and labels from a .npz file.

    Args:
        npz_path (str): Path to the .npz file containing 'X_latent' and optionally 'y'.

    Returns:
        tuple: A tuple (X_latent, y) where:
            - X_latent (np.ndarray): Latent feature vectors.
            - y (np.ndarray or None): Labels if present in the file, else None.
    """
    data = np.load(npz_path)
    X_latent = data["X_latent"]
    y = data["y"] if "y" in data else None
    return X_latent, y


def extract_segment_length(path_str):
    """Extract segment length (e.g. 295) from path like 'latent_len295_fold4'.

    Args:
        path_str (str): Path or folder name.

    Returns:
        str: Extracted segment length (e.g. '295'), or 'unknown' if not found.
    """
    match = re.search(r"len(\d+)", path_str)
    return match.group(1) if match else "unknown"


def plot_embedding(
    X, y=None, method="umap", title="Latent Space", save=False, output_name=None
):
    """Reduce dimensionality of latent space and generate a scatter plot.

    Args:
        X (np.ndarray): Latent feature vectors of shape (n_samples, n_features).
        y (np.ndarray, optional): Class labels for coloring the points.
        method (str): Dimensionality reduction method: 'umap' or 'pca'.
        title (str): Title of the plot.
        save (bool): Whether to save the plot as a PNG file.
        output_name (str, optional): Custom filename for the output plot.

    Raises:
        ValueError: If method is unsupported or UMAP is not installed.
    """
    if method == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42)
    elif method == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Unsupported method. Use 'umap' or 'pca'.")

    X_embedded = reducer.fit_transform(X)

    # Compute Silhouette Score in the reduced space with appropriate metric
    if y is not None and len(np.unique(y)) > 1:
        try:
            silhouette_metric = "cosine" if method == "umap" else "euclidean"
            score = silhouette_score(X_embedded, y, metric=silhouette_metric)
            print(
                f"📊 Silhouette Score ({method.upper()} space, {silhouette_metric}): {score:.4f}"
            )
        except Exception as e:
            print(f"⚠️ Error computing silhouette score in {method.upper()} space: {e}")

    plt.figure(figsize=(10, 6))
    if y is not None:
        scatter = plt.scatter(
            X_embedded[:, 0], X_embedded[:, 1], c=y, cmap="Set1", alpha=0.6
        )
        plt.legend(*scatter.legend_elements(), title="Class")
    else:
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.6)

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()

    if save:
        filename = (
            output_name
            if output_name
            else f"{title.replace(' ', '_').lower()}_{method}.png"
        )
        plt.savefig(filename, dpi=300)
        print(f"📁 Plot saved as: {filename}")

    plt.show()


def extract_model_suffix(path_str):
    """Extract model suffix (e.g., 'B') from folder name like 'latent_len295_B_fold2'.

    Args:
        path_str (str): Input path string.

    Returns:
        str: Extracted model suffix, or '' if not found.
    """
    match = re.search(r"len\d+_([A-Z])_", path_str)
    return match.group(1) if match else ""


def extract_fold_number(path_str):
    """Extract fold number (e.g., 'fold3') from folder or file path.

    Args:
        path_str (str): Input path string.

    Returns:
        str: Fold number string (e.g., 'fold3'), or '' if not found.
    """
    match = re.search(r"fold(\d+)", path_str)
    return f"fold{match.group(1)}" if match else ""


def main():
    """Main function to load latent data, compute Silhouette Score, and visualize embeddings."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Path to .npz file with latent vectors"
    )
    parser.add_argument(
        "--method",
        choices=["umap", "pca"],
        default="umap",
        help="Dimensionality reduction method",
    )
    parser.add_argument("--save", action="store_true", help="Save plot as PNG")
    args = parser.parse_args()

    X_latent, y = load_latents(args.input)
    print(f"✅ Loaded latent shape: {X_latent.shape}")

    # Optional: Silhouette Score in original latent space (for reference)
    # if y is not None and len(np.unique(y)) > 1:
    #   try:
    #        score = silhouette_score(X_latent, y, metric="cosine")
    #        print(f"📊 Silhouette Score (original space, cosine): {score:.4f}")
    #    except Exception as e:
    #        print(f"⚠️ Error computing silhouette score: {e}")

    segment_length = extract_segment_length(args.input)
    model_suffix = extract_model_suffix(args.input)
    fold_suffix = extract_fold_number(args.input)

    parts = ["latent", args.method, segment_length]
    if model_suffix:
        parts.append(model_suffix)
    if fold_suffix:
        parts.append(fold_suffix)

    filename = "_".join(parts) + ".png"

    plot_embedding(
        X_latent,
        y,
        method=args.method,
        title=f"Latent Visualization ({args.method.upper()})",
        save=args.save,
        output_name=filename,
    )


if __name__ == "__main__":
    main()
