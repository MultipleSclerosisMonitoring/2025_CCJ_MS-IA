import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def load_latents(npz_path):
    data = np.load(npz_path)
    X_latent = data["X_latent"]
    y = data["y"] if "y" in data else None
    return X_latent, y


def plot_embedding(X, y=None, method="tsne", title="Latent Space"):
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42)
    elif method == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Unsupported method or UMAP not installed")

    X_embedded = reducer.fit_transform(X)

    plt.figure(figsize=(10, 6))
    if y is not None:
        scatter = plt.scatter(
            X_embedded[:, 0], X_embedded[:, 1], c=y, cmap="coolwarm", alpha=0.6
        )
        plt.legend(*scatter.legend_elements(), title="Class")
    else:
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.6)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Path to .npz file with latent vectors"
    )
    parser.add_argument(
        "--method",
        choices=["tsne", "umap", "pca"],
        default="tsne",
        help="Dimensionality reduction method",
    )
    args = parser.parse_args()

    X_latent, y = load_latents(args.input)
    print(f"✅ Loaded latent shape: {X_latent.shape}")

    plot_embedding(
        X_latent,
        y,
        method=args.method,
        title=f"Latent Visualization ({args.method.upper()})",
    )


if __name__ == "__main__":
    main()
