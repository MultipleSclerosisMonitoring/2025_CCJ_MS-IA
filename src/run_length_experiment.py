import argparse
import h5py
import numpy as np
import os
from pathlib import Path
import subprocess
import matplotlib.pyplot as plt


def truncate_dataset(X, length):
    """Truncate each sample in a 3D dataset to the specified number of timesteps.

    Args:
        X (np.ndarray): Input array of shape (samples, timesteps, features).
        length (int): Number of timesteps to retain.

    Returns:
        np.ndarray: Truncated array of shape (samples, length, features).

    Raises:
        ValueError: If the requested length exceeds the current number of timesteps.
    """
    if X.shape[1] < length:
        raise ValueError(
            f"Dataset only has {X.shape[1]} timesteps, can't truncate to {length}."
        )
    return X[:, :length, :]


def save_truncated_hdf5(X, original_path, length, output_dir):
    """Save the truncated dataset to a new HDF5 file.

    Args:
        X (np.ndarray): Truncated dataset.
        original_path (str): Path to the original dataset (unused, for reference).
        length (int): Number of timesteps in the truncated dataset.
        output_dir (str): Directory to save the new file.

    Returns:
        Path: Path to the saved HDF5 file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"dataset_balanced_len{length}_50Hz.hdf5"
    with h5py.File(out_path, "w") as f:
        f.create_dataset("X", data=X)
    return out_path


def main():
    """Main function to run sensitivity experiments over different segment lengths.

    Loads a dataset, truncates it to various lengths, saves each version,
    and invokes a training script for each one with the same hyperparameters.
    Then summarizes and plots the best val_loss for each length.

    Command-line Arguments:
        --input (str): Path to original HDF5 file with 'X' dataset.
        --lengths (list of int): Segment lengths to test.
        --train_script (str): Path to training script (default: src/train_autoencoder.py).
        --epochs (int): Number of training epochs per experiment.
        --batch_size (int): Batch size during training.
        --head_size (int): Attention head size.
        --num_heads (int): Number of attention heads.
        --ff_dim (int): Feed-forward layer dimensionality.
        --dropout (float): Dropout rate.
        --num_blocks (int): Number of Transformer blocks.
        --n_splits (int): Number of folds in K-Fold cross-validation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="Path to original HDF5 file"
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        required=True,
        help="List of segment lengths to evaluate",
    )
    parser.add_argument(
        "--train_script",
        type=str,
        default="src/train_autoencoder.py",
        help="Path to training script",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--head_size", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--ff_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load original data
    with h5py.File(args.input, "r") as f:
        X = f["X"][:]

    for length in args.lengths:
        print(f"\n🚀 Running experiment for segment length: {length}")
        try:
            X_trunc = truncate_dataset(X, length)
        except ValueError as e:
            print(f"❌ Skipping length {length}: {e}")
            continue

        # Save truncated dataset
        truncated_path = save_truncated_hdf5(
            X_trunc, args.input, length, output_dir="data_balanced"
        )
        output_dir = f"models_len{length}"

        cmd = [
            "python3",
            args.train_script,
            "--input",
            str(truncated_path),
            "--output",
            output_dir,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--head_size",
            str(args.head_size),
            "--num_heads",
            str(args.num_heads),
            "--ff_dim",
            str(args.ff_dim),
            "--dropout",
            str(args.dropout),
            "--num_blocks",
            str(args.num_blocks),
            "--n_splits",
            str(args.n_splits),
        ]

        with open(Path(output_dir) / "args.txt", "w") as f:
            for key, value in vars(args).items():
                f.write(f"{key}: {value}\n")
            f.write(f"length: {length}\n")

        print("\n▶️ Executing:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Training failed for length {length}. Error:\n{result.stderr}")
            continue

    # Summarize results
    print("\n📊 Generating summary across lengths...")
    length_results = []

    for length in args.lengths:
        model_dir = Path(f"models_len{length}")
        val_losses = []

        for fold in range(1, args.n_splits + 1):
            loss_file = model_dir / f"fold_{fold}" / "loss_history.npy"
            if loss_file.exists():
                history = np.load(loss_file, allow_pickle=True).item()
                val_losses.append(min(history["val_loss"]))
            else:
                print(f"⚠️  No loss_history for length {length}, fold {fold}, skipping.")

        if val_losses:
            mean_val = np.mean(val_losses)
            std_val = np.std(val_losses)
            length_results.append((length, mean_val, std_val))

    # Save summary
    summary_path = Path("summary_lengths.txt")
    with open(summary_path, "w") as f:
        for length, mean_val, std_val in length_results:
            f.write(
                f"Length {length}: mean val_loss = {mean_val:.6f}, std = {std_val:.6f}\n"
            )

    print("\n📈 Summary (mean ± std val_loss per segment length):")
    for length, mean_val, std_val in length_results:
        print(f"  Length {length}: {mean_val:.6f} ± {std_val:.6f}")

    # Plot with error bars
    if length_results:
        lengths, means, stds = zip(*length_results)
        plt.figure(figsize=(8, 5))
        plt.errorbar(lengths, means, yerr=stds, fmt="o-", capsize=5)
        plt.title("Validation Loss vs Segment Length (Mean ± STD)")
        plt.xlabel("Segment Length")
        plt.ylabel("Mean Validation Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("summary_plot.png")
        print("\n📊 Saved plot with error bars to summary_plot.png")


if __name__ == "__main__":
    main()
