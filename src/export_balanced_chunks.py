"""
export_balanced_chunks.py

This script reads segmented .xlsx files containing sensor data, processes them into fixed-length arrays,
labels them according to movement type (walking or not), balances the class distribution, and saves the
resulting datasets as NumPy arrays.

Example:
    python export_balanced_chunks.py --input output/ --output data/ --length 250 --mode pad --verbose 2
"""

import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter


def main():
    """
    Main execution function.

    Parses CLI arguments to process chunk files in a folder, converts them to fixed-length
    arrays for machine learning, balances the dataset classes, and saves the arrays to disk.

    CLI Args:
        --input (str): Folder containing .xlsx chunk files.
        --output (str): Output directory for the resulting .npy files.
        --length (int): Fixed length of the output samples.
        --mode (str): Padding or truncation strategy for short samples ('pad' or 'truncate').
        --verbose (int): Verbosity level (0-3).

    Output:
        Saves X_balanced.npy and y_balanced.npy arrays in the output directory.
        Also prints summary and skipped file stats.
    """
    parser = argparse.ArgumentParser(
        description="Export balanced chunks to X/y arrays."
    )
    parser.add_argument("--input", default="output", help="Folder with .xlsx chunks.")
    parser.add_argument(
        "--output", default=".", help="Output directory for .npy files."
    )
    parser.add_argument("--length", type=int, default=250, help="Fixed chunk length.")
    parser.add_argument(
        "--mode",
        choices=["pad", "truncate"],
        default="pad",
        help="Handle short chunks.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="Verbose level (0-3).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sensors = ["S0", "S1", "S2", "Ax", "Ay", "Az", "Gx", "Gy", "Gz", "Mx", "My", "Mz"]
    label_map = {"walking": 1, "not_walking": 0}

    X_all, y_all, files_all = [], [], []
    label_counter = Counter()
    skipped_too_short = 0
    skipped_missing_sensors = 0
    files = sorted(input_dir.glob("*.xlsx"))

    if args.verbose >= 1:
        print(f"📂 Input: {input_dir} | 💾 Output: {output_dir}")
        print(f"🎯 Length: {args.length} | Mode: {args.mode}\n")

    for file in tqdm(files, desc="📦 Processing chunks"):
        if "+" not in file.name or file.name == "resumen_chunks.xlsx":
            continue

        parts = file.stem.split("+")
        if len(parts) < 5:
            continue

        move_type = parts[1]
        if move_type not in label_map:
            continue

        try:
            df = pd.read_excel(file)

            if not all(col in df.columns for col in sensors):
                skipped_missing_sensors += 1
                if args.verbose == 3:
                    print(f"⚠️ Skipped {file.name}: missing required sensors")
                continue

            df = df.dropna(subset=sensors)
            if df.empty or len(df) < 10:
                skipped_too_short += 1
                if args.verbose == 3:
                    print(f"⚠️ Skipped {file.name}: too few rows ({len(df)})")
                continue

            data = df[sensors].values

            if args.mode == "truncate":
                if len(data) >= args.length:
                    data = data[: args.length]
                else:
                    skipped_too_short += 1
                    if args.verbose == 3:
                        print(f"⚠️ Skipped {file.name}: too short for truncation")
                    continue
            elif args.mode == "pad":
                if len(data) < args.length:
                    pad = np.zeros((args.length - len(data), len(sensors)))
                    data = np.vstack([data, pad])
                else:
                    data = data[: args.length]

            X_all.append(data)
            y_all.append(label_map[move_type])
            files_all.append(file.name)
            label_counter[move_type] += 1

            if args.verbose == 2:
                print(f"✅ {file.name}: {data.shape}")

        except Exception as e:
            if args.verbose >= 2:
                print(f"⚠️ Error processing {file.name}: {e}")

    # Convertir a arrays
    X_all = np.array(X_all)
    y_all = np.array(y_all)

    # Balancear clases
    walking_idx = np.where(y_all == 1)[0]
    not_walking_idx = np.where(y_all == 0)[0]
    min_class_count = min(len(walking_idx), len(not_walking_idx))

    np.random.shuffle(walking_idx)
    np.random.shuffle(not_walking_idx)

    selected_idx = np.concatenate(
        [walking_idx[:min_class_count], not_walking_idx[:min_class_count]]
    )
    np.random.shuffle(selected_idx)

    X_bal = X_all[selected_idx]
    y_bal = y_all[selected_idx]

    np.save(output_dir / "X_balanced.npy", X_bal)
    np.save(output_dir / "y_balanced.npy", y_bal)

    print(f"\n✅ Balanced dataset saved: {len(X_bal)} samples")
    print(f"📐 X_balanced shape: {X_bal.shape} | y_balanced shape: {y_bal.shape}")

    if args.verbose == 3:
        print("\n📊 Class distribution after balancing:")
        print(f"  - walking: {sum(y_bal == 1)}")
        print(f"  - not_walking: {sum(y_bal == 0)}")
        print("\n🚫 Skipped files:")
        print(f"  - Too short: {skipped_too_short}")
        print(f"  - Missing sensors: {skipped_missing_sensors}")


if __name__ == "__main__":
    main()
