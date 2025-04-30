import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict


def plot_chunk_signals(filepaths: list[str], output_dir: str = None, show: bool = True):
    """
    Plot pressure signals (S0, S1, S2) from multiple chunk files grouped by qtok.

    :param filepaths: List of paths to .xlsx chunk files.
    :param output_dir: Directory to save plots as PNG (optional).
    :param show: Whether to display plots on screen (default: True).
    """

    qtok_data = defaultdict(list)

    # Group files by qtok (assumed in filename format: qtok+move+start+end+leg.xlsx)
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        parts = filename.split("+")
        if len(parts) < 5:
            print(f"Skipping invalid filename: {filename}")
            continue

        qtok = parts[0]
        leg = parts[4].split(".")[0]  # "Left" or "Right"

        try:
            df = pd.read_excel(filepath)
            if not all(col in df.columns for col in ["_time", "S0", "S1", "S2"]):
                print(f"Missing required columns in: {filename}")
                continue

            df["_time"] = pd.to_datetime(df["_time"])
            qtok_data[qtok].append((leg, df))

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # Plot one figure per qtok
    for qtok, leg_dfs in qtok_data.items():
        plt.figure(figsize=(14, 6))
        for leg, df in leg_dfs:
            for signal in ["S0", "S1", "S2"]:
                plt.plot(df["_time"], df[signal], label=f"{signal} - {leg}")

        plt.title(f"Pressure Signals for {qtok}")
        plt.xlabel("Time")
        plt.ylabel("Pressure (S0/S1/S2)")
        plt.legend()
        plt.grid(True)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, f"{qtok}_pressures.png")
            plt.savefig(plot_path)
            print(f"✅ Saved plot: {plot_path}")

        if show:
            plt.show()
        else:
            plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot pressure signals from chunk files."
    )
    parser.add_argument(
        "--input", required=True, help="Path to folder with .xlsx chunk files."
    )
    parser.add_argument("--output", help="Path to save generated plots (optional).")
    parser.add_argument("--no-show", action="store_true", help="Disable plot display.")
    args = parser.parse_args()

    # Obtener todos los .xlsx del input
    files = [
        os.path.join(args.input, f)
        for f in os.listdir(args.input)
        if f.endswith(".xlsx")
    ]
    plot_chunk_signals(files, output_dir=args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
