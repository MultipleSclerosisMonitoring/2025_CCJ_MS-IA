import os
import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict


def plot_chunk_signals(filepaths: list[str], output_dir: str = None, show: bool = True):
    from collections import defaultdict
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    qtok_data = defaultdict(list)
    saved_count = 0
    skipped_count = 0

    # Agrupar por qtok
    for filepath in tqdm(filepaths, desc="Processing chunks"):
        filename = Path(filepath).name
        parts = filename.split("+")
        if len(parts) < 5:
            print(f"⚠️ Skipping invalid filename: {filename}")
            continue

        qtok = parts[0]
        leg = parts[4].split(".")[0]

        try:
            df = pd.read_excel(filepath)
            if not all(col in df.columns for col in ["_time", "S0", "S1", "S2"]):
                print(f"Missing required columns in: {filename}")
                continue

            if df[["S0", "S1", "S2"]].dropna().empty:
                print(f"⚠️ No valid pressure data in {filename}")
                continue

            df["_time"] = pd.to_datetime(df["_time"])
            qtok_data[qtok].append((leg, df))

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # Crear gráficos
    for qtok, leg_dfs in qtok_data.items():
        plot_path = Path(output_dir) / f"{qtok}_pressures.png"
        if plot_path.exists():
            print(f"⏩ Skipping already plotted: {plot_path.name}")
            skipped_count += 1
            continue

        # Agrupar por pierna
        grouped_by_leg = defaultdict(list)
        for leg, df in leg_dfs:
            grouped_by_leg[leg].append(df)

        fig, axs = plt.subplots(
            nrows=1,
            ncols=len(grouped_by_leg),
            figsize=(16, 5),
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        for idx, (leg, dfs) in enumerate(grouped_by_leg.items()):
            ax = axs[0][idx]
            legend_drawn = set()
            for df in dfs:
                for signal in ["S0", "S1", "S2"]:
                    if signal in df.columns:
                        label = signal if signal not in legend_drawn else None
                        ax.plot(df["_time"], df[signal], label=label)
                        legend_drawn.add(signal)

            ax.set_title(f"{qtok} - {leg}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Pressure")
            ax.grid(True)
            ax.legend(loc="upper right")

        fig.autofmt_xdate()
        fig.tight_layout()

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path)
        print(f"✅ Saved plot: {plot_path}")
        saved_count += 1

        if show:
            plt.show()
        else:
            plt.close(fig)

    plt.close("all")
    print(
        f"\n📊 Summary: {saved_count} images saved, {skipped_count} skipped (already existed)."
    )


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

    files = [
        os.path.join(args.input, f)
        for f in os.listdir(args.input)
        if f.lower().endswith(".xlsx") and "+" in f
    ]
    plot_chunk_signals(files, output_dir=args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
