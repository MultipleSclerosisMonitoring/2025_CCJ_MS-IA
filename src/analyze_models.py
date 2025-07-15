import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import traceback
import pandas as pd
import re


def plot_and_summarize_all_models(base_dir="."):
    """Generate loss plots and validation summaries for all models.

    This function scans directories matching `models_len*`, plots training and validation
    loss curves for each fold, calculates the mean and standard deviation of the best
    validation losses, and saves both plots and summaries inside each model directory.

    Args:
        base_dir (str, optional): Base directory containing `models_len*` folders.
                                  Defaults to current directory ".".

    Returns:
        pd.DataFrame: A DataFrame with model names, segment length, mean validation loss,
                      and standard deviation.
    """
    summary_records = []

    for model_dir in sorted(Path(base_dir, "models_len").glob("models_len*")):
        model_name = model_dir.name
        match = re.search(r"models_len(\d+)", model_name)
        segment_len = int(match.group(1)) if match else None

        val_losses = []
        plt.figure(figsize=(10, 5))
        fold_dirs = sorted(
            [f for f in model_dir.glob("fold_*") if f.is_dir()],
            key=lambda x: int(x.name.split("_")[1]),
        )

        for fold_dir in fold_dirs:
            history_path = fold_dir / "loss_history.npy"
            try:
                if history_path.exists():
                    history = np.load(history_path, allow_pickle=True).item()
                    if "loss" in history and "val_loss" in history:
                        plt.plot(history["loss"], label=f"{fold_dir.name} - Train")
                        plt.plot(
                            history["val_loss"],
                            label=f"{fold_dir.name} - Val",
                            linestyle="--",
                        )
                        val_losses.append(min(history["val_loss"]))
                    else:
                        print(f"⚠️ {fold_dir.name}: Missing keys.")
                else:
                    print(f"⚠️ {fold_dir.name}: loss_history.npy not found.")
            except Exception:
                print(f"❌ Error loading {history_path}:\n{traceback.format_exc()}")

        if val_losses:
            mean_val = np.mean(val_losses)
            std_val = np.std(val_losses)
            summary_records.append(
                {
                    "Model": model_name,
                    "Segment Length": segment_len,
                    "Mean Val Loss": mean_val,
                    "Std Dev": std_val,
                }
            )

            plt.title(f"Loss Curves - {model_name}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plot_path = model_dir / "loss_curves.png"
            plt.savefig(plot_path)
            plt.close()

            with open(model_dir / "summary.txt", "w") as f:
                for i, loss in enumerate(val_losses, 1):
                    f.write(f"  Fold {i}: {loss:.6f}\n")
                f.write(f"\nMean: {mean_val:.6f}\n")
                f.write(f"Std Dev: {std_val:.6f}\n")
        else:
            print(f"⚠️ No valid val_loss for {model_name}")

    return pd.DataFrame(summary_records)


def plot_summary_comparison(df, output_path="comparison_plot.png"):
    """Plot comparison of mean validation loss across models.

    Args:
        df (pd.DataFrame): Summary DataFrame with segment lengths and mean val losses.
        output_path (str): Path to save the comparison plot.
    """
    df_sorted = df.sort_values("Segment Length")
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        df_sorted["Segment Length"],
        df_sorted["Mean Val Loss"],
        yerr=df_sorted["Std Dev"],
        fmt="o-",
        capsize=4,
    )

    plt.title("Mean Validation Loss vs Segment Length")
    plt.xlabel("Segment Length (timesteps)")
    plt.ylabel("Mean Validation Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"📈 Saved comparison plot to {output_path}")
    plt.close()


if __name__ == "__main__":
    """Main execution block.

    Loads and summarizes all models in the current directory.
    Saves a global CSV file with validation statistics.
    Plots global comparison across segment lengths.
    """
    df_summary = plot_and_summarize_all_models(".")
    if not df_summary.empty:
        print("\n📋 Resumen comparativo de modelos:")
        print(df_summary.to_string(index=False))
        df_summary.to_csv("summary_all_models.csv", index=False)
        print("📝 Guardado en summary_all_models.csv")

        plot_summary_comparison(df_summary, "comparison_plot.png")
    else:
        print("❌ No se han encontrado resultados válidos.")
