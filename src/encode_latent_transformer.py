import numpy as np
import argparse
from pathlib import Path
from tensorflow.keras.models import load_model


def main():
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
    y = np.load(args.y)

    assert len(X.shape) == 3, "❌ X must have shape (samples, timesteps, features)"
    assert X.shape[0] == y.shape[0], "❌ X and y must have the same number of samples"

    print("🧠 Loading encoder model...")
    encoder = load_model(args.encoder)

    print("🔁 Encoding with transformer encoder...")
    X_latent = encoder.predict(X, batch_size=args.batch_size, verbose=1)

    print(f"\n✅ Encoded shape: {X_latent.shape}, saving to {args.output}")
    np.save(Path(args.output) / "X_latent.npy", X_latent)
    np.save(Path(args.output) / "y_latent.npy", y)
    print("📁 Saved: X_latent.npy and y_latent.npy")


if __name__ == "__main__":
    main()
