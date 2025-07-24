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


def plot_embedding(X, y=None, method="umap", title="Latent Space", save=False):
    """Reduce dimensionality of latent space and generate a scatter plot.

    Args:
        X (np.ndarray): Latent feature vectors of shape (n_samples, n_features).
        y (np.ndarray, optional): Class labels for coloring the points. Defaults to None.
        method (str): Dimensionality reduction method. Options: 'umap', 'pca'. Defaults to 'umap'.
        title (str): Title of the plot. Defaults to "Latent Space".
        save (bool): Whether to save the plot as a PNG file. Defaults to False.

    Raises:
        ValueError: If the specified method is unsupported or UMAP is not installed.
    """
    if method == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42)
    elif method == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Unsupported method. Use 'umap' or 'pca'.")

    X_embedded = reducer.fit_transform(X)

    plt.figure(figsize=(10, 6))
    if y is not None:
        scatter = plt.scatter(
            X_embedded[:, 0], X_embedded[:, 1], c=y, cmap="Set1", alpha=0.6
        )
        plt.legend(*scatter.legend_elements(), title="Clase")
    else:
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.6)

    plt.title(title)
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(output_name, dpi=300)
        print(f"📁 Plot saved as: {output_name}")

    plt.show()


def main():
    """Main function to load latent data, compute Silhouette Score, and visualize embeddings.

    Parses command-line arguments to:
    - Load a .npz file containing latent vectors and optional labels.
    - Compute the Silhouette Score (if labels are present and valid).
    - Apply dimensionality reduction (UMAP or PCA).
    - Plot the resulting 2D embeddings and optionally save the plot.
    """
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

    if y is not None and len(np.unique(y)) > 1:
        try:
            score = silhouette_score(X_latent, y, metric="cosine")
            print(f"📊 Silhouette Score (original space, cosine): {score:.4f}")
        except Exception as e:
            print(f"⚠️ Error computing silhouette score: {e}")

    # Extract segment length
    segment_length = extract_segment_length(args.input)

    filename = f"latent_{args.method}_{segment_length}.png"

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
