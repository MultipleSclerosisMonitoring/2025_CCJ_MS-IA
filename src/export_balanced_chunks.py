"""
export_balanced_chunks.py

Version that:
- Processes sensor segment .xlsx files.
- Computes modulus of acceleration,gyroscope and magnetometer, keeps S0–S2.
- Truncates to median length.
- Discards chunks longer than 125% of the median.
- Balances class samples.
- Saves output as a .hdf5 dataset (X, y) for training.

Usage:
    python export_balanced_chunks.py --input output/ --output data_balanced/ --verbose 2
"""

import argparse
import os
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from collections import Counter


def save_to_hdf5(
    X, y, output_dir, base_name="dataset_balanced", sample_rate_hz=50, length=None
):
    """Save the preprocessed data into an HDF5 file.

    Args:
        X (np.ndarray): Sensor data of shape (samples, timesteps, features).
        y (np.ndarray): Corresponding labels.
        output_dir (str or Path): Folder to store the resulting file.
        base_name (str): Base filename (default 'dataset_balanced').
        sample_rate_hz (int): Sampling frequency to include in filename.
        length (int): Number of timesteps (default: inferred from X).

    Returns:
        Path: Path to saved HDF5 file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if length is None:
        length = X.shape[1]
    filename = f"{base_name}_len{length}_{sample_rate_hz}Hz.hdf5"
    filepath = output_dir / filename

    with h5py.File(filepath, "w") as f:
        f.create_dataset("X", data=X, compression="gzip")
        f.create_dataset("y", data=y, compression="gzip")
    return filepath


def compute_mod_columns(df):
    """Compute ModA, ModG, ModM from raw accelerometer, gyroscope and magnetometer data.

    Args:
        df (pd.DataFrame): DataFrame with raw columns ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz'].

    Returns:
        pd.DataFrame: DataFrame with ['S0', 'S1', 'S2', 'ModA', 'ModG', 'ModM'].
    """
    df["ModA"] = np.sqrt(df["Ax"] ** 2 + df["Ay"] ** 2 + df["Az"] ** 2)
    df["ModG"] = np.sqrt(df["Gx"] ** 2 + df["Gy"] ** 2 + df["Gz"] ** 2)
    df["ModM"] = np.sqrt(df["Mx"] ** 2 + df["My"] ** 2 + df["Mz"] ** 2)
    return df[["S0", "S1", "S2", "ModA", "ModG", "ModM"]]


def main():
    """Main entry point to export balanced sensor chunks as HDF5.

    This function processes Excel files containing sensor time series,
    computes signal magnitudes, truncates each sample to a uniform length,
    balances class distributions, and saves the result in a compressed
    HDF5 dataset for training.

    Behavior depends on whether a fixed length is specified:

    - If `--fixed_length` is provided, only segments with length >= fixed_length
      are kept and truncated to that value.
    - Otherwise, the median length of all segments is computed, and segments are
      kept if they are within 125% of that median. Truncation is then applied.

    Args:
        Uses argparse to parse the following arguments:
            --input (str): Folder with input .xlsx chunk files.
            --output (str): Output folder to store the resulting HDF5 dataset.
            --fixed_length (int, optional): Fixed length to truncate all segments.
            --verbose (int): Verbosity level (0–3).

    Raises:
        Prints error messages if no valid segments are found or if loading fails.
    """

    def main():
        parser = argparse.ArgumentParser(
            description="Export smart-truncated balanced chunks to HDF5."
        )
        parser.add_argument(
            "--input", default="output", help="Folder with .xlsx chunks."
        )
        parser.add_argument(
            "--output", default=".", help="Output directory for .hdf5 file."
        )
        parser.add_argument(
            "--fixed_length",
            type=int,
            default=None,
            help="Force fixed length for all segments.",
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

        all_columns = [
            "S0",
            "S1",
            "S2",
            "Ax",
            "Ay",
            "Az",
            "Gx",
            "Gy",
            "Gz",
            "Mx",
            "My",
            "Mz",
        ]
        label_map = {"walking": 1, "not_walking": 0}

        files = sorted(input_dir.glob("*.xlsx"))
        chunk_info = []
        chunk_lengths = []
        skipped_too_short = 0
        skipped_too_long = 0
        skipped_missing_sensors = 0

        if args.verbose >= 1:
            print(f"📂 Input: {input_dir} | 💾 Output: {output_dir}")

        for file in tqdm(files, desc="📊 Analyzing chunks"):
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
                if not all(col in df.columns for col in all_columns):
                    skipped_missing_sensors += 1
                    continue
                df = df.dropna(subset=all_columns)
                if df.empty or len(df) < 10:
                    skipped_too_short += 1
                    continue
                df_mod = compute_mod_columns(df)
                chunk_info.append((file.name, move_type, df_mod.values))
                chunk_lengths.append(len(df_mod))
            except Exception as e:
                if args.verbose >= 3:
                    print(f"⚠️ Error reading {file.name}: {e}")

        if not chunk_lengths:
            print("❌ No valid chunks found.")
            return

        if args.fixed_length is not None:
            truncate_len = args.fixed_length
            if args.verbose >= 1:
                print(f"\n📏 Forcing fixed length: {truncate_len}")
        else:
            truncate_len = int(np.median(chunk_lengths))
            length_threshold = int(truncate_len * 1.30)
            if args.verbose >= 1:
                print(
                    f"\n📏 Truncating to median: {truncate_len}, max allowed: {length_threshold}"
                )

        X_all, y_all = [], []
        label_counter = Counter()

        for filename, move_type, data in tqdm(
            chunk_info, desc="✂️ Filtering & truncating"
        ):
            if args.fixed_length is not None:
                if len(data) < truncate_len:
                    skipped_too_short += 1
                    continue
            else:
                if len(data) < truncate_len:
                    skipped_too_short += 1
                    continue
                if len(data) > length_threshold:
                    skipped_too_long += 1
                    continue

            data = data[:truncate_len]
            X_all.append(data)
            y_all.append(label_map[move_type])
            label_counter[move_type] += 1
            if args.verbose == 2:
                print(f"✅ {filename}: kept {data.shape}")

        X_all = np.array(X_all)
        y_all = np.array(y_all)

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

        save_to_hdf5(X_bal, y_bal, output_dir, length=truncate_len)

        print(f"\n✅ Balanced dataset saved: {len(X_bal)} samples")
        print(f"📐 X shape: {X_bal.shape} | y shape: {y_bal.shape}")

        if args.verbose == 3:
            print("\n📊 Class distribution after balancing:")
            print(f"  - walking: {sum(y_bal == 1)}")
            print(f"  - not_walking: {sum(y_bal == 0)}")
            print("\n🚫 Skipped files:")
            print(f"  - Too short: {skipped_too_short}")
            print(f"  - Too long: {skipped_too_long}")
            print(f"  - Missing sensors: {skipped_missing_sensors}")


if __name__ == "__main__":
    main()
