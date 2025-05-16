import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter


def main():
    parser = argparse.ArgumentParser(
        description="Export structured chunks to X/y arrays."
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

    X, y = [], []
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

            X.append(data)
            y.append(label_map[move_type])
            label_counter[move_type] += 1

            if args.verbose == 2:
                print(f"✅ {file.name}: {data.shape}")

        except Exception as e:
            if args.verbose >= 2:
                print(f"⚠️ Error processing {file.name}: {e}")

    # Guardar
    X = np.array(X)
    y = np.array(y)

    np.save(output_dir / "X_chunks.npy", X)
    np.save(output_dir / "y_chunks.npy", y)

    print(f"\n✅ Saved: {len(X)} samples")
    print(f"📐 X shape: {X.shape} | y shape: {y.shape}")

    if args.verbose == 3:
        print("\n📊 Class distribution:")
        for label, count in label_counter.items():
            print(f"  - {label}: {count}")

        print("\n🚫 Skipped files:")
        print(f"  - Too short: {skipped_too_short}")
        print(f"  - Missing sensors: {skipped_missing_sensors}")


if __name__ == "__main__":
    main()
