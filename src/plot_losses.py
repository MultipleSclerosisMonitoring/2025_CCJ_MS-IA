from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import traceback


def plot_loss_histories(models_dir="models"):
    """Plots training and validation loss curves for each fold.

    Iterates over fold directories in the specified models directory, reads
    their loss history files, and plots the training and validation loss curves
    for each fold. Handles missing or corrupt files gracefully.

    Args:
        models_dir (str): Path to the directory containing fold subdirectories.
    """
    fold_dirs = sorted(
        [f for f in os.listdir(models_dir) if f.startswith("fold_")],
        key=lambda x: int(x.split("_")[1]),
    )

    plt.figure(figsize=(12, 6))

    for fold in fold_dirs:
        history_path = os.path.join(models_dir, fold, "loss_history.npy")
        if os.path.exists(history_path):
            try:
                history = np.load(history_path, allow_pickle=True).item()
                if "loss" in history and "val_loss" in history:
                    plt.plot(history["loss"], label=f"{fold} - Train")
                    plt.plot(history["val_loss"], label=f"{fold} - Val", linestyle="--")
                else:
                    print(f"⚠️  {fold}: Missing keys in history file.")
            except Exception:
                print(f"❌ Error loading {history_path}:\n{traceback.format_exc()}")
        else:
            print(f"⚠️  {fold}: ❌ loss_history.npy not found.")

    plt.title("Training and Validation Loss per Fold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = Path(models_dir) / "loss_curves.png"
    plt.savefig(plot_path)
    print(f"\n📊 Saved loss plot to {plot_path}")

    plt.show()


def summarize_final_losses(models_dir="models"):
    """Prints and saves the best validation loss per fold and computes averages.

    This function loads the loss history from each fold directory inside the given models directory,
    retrieves the best validation loss per fold, prints them, and saves the summary to disk.
    Handles corrupt or missing files gracefully.

    Args:
        models_dir (str): Directory containing `fold_1`, `fold_2`, ..., each with `loss_history.npy`.
    """
    val_losses = []

    fold_dirs = sorted(
        [f for f in os.listdir(models_dir) if f.startswith("fold_")],
        key=lambda x: int(x.split("_")[1]),
    )

    print("Best validation losses per fold:")
    for fold in fold_dirs:
        history_path = os.path.join(models_dir, fold, "loss_history.npy")
        if os.path.exists(history_path):
            try:
                history = np.load(history_path, allow_pickle=True).item()
                best_val = min(history["val_loss"])
                val_losses.append(best_val)
                print(f"  {fold}: Best Val Loss = {best_val:.6f}")
            except Exception:
                print(f"❌ Error reading {history_path}:\n{traceback.format_exc()}")
        else:
            print(f"  {fold}: ❌ loss_history.npy not found.")

    if val_losses:
        mean = np.mean(val_losses)
        std = np.std(val_losses)

        # Save as .npy and .txt
        np.save(Path(models_dir) / "final_val_losses.npy", val_losses)

        summary_path = Path(models_dir) / "summary.txt"
        with open(summary_path, "w") as f:
            f.write("Best Validation Loss per Fold:\n")
            for i, loss in enumerate(val_losses, 1):
                f.write(f"  Fold {i}: {loss:.6f}\n")
            f.write(f"\nMean: {mean:.6f}\n")
            f.write(f"Std Dev: {std:.6f}\n")

        print("\n📈 Summary across folds:")
        print(f"  Mean Val Loss: {mean:.6f}")
        print(f"  Std Dev:       {std:.6f}")
        print(f"\n📝 Saved summary to {summary_path}")
    else:
        print("❌ No valid loss histories found.")


# Entry point
if __name__ == "__main__":
    plot_loss_histories("models")
    summarize_final_losses("models")
