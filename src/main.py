"""
main.py

Script to extract walking segments from InfluxDB using time intervals provided in an Excel file.
Segments are saved to files, and a summary of the extraction process is generated.

Requires a configuration YAML file for InfluxDB credentials.

Example:
    python main.py --input segments.xlsx --output ./chunks/ --durations 5 10 --verbose 2
"""

import argparse
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from .InfluxDBms.cInfluxDB import cInfluxDB

def load_config_path(cli_config: str = None) -> str:
    """
    Determines which configuration file to use to connect to InfluxDB.

    Priority:
    1. CLI argument (--config).
    2. File named '..config_db.yaml' in the current directory.
    3. File named '.config_db.yaml' in the current directory.

    Args:
        cli_config (str, optional): Path to a config YAML file passed via CLI.

    Returns:
        str: The path to the config file to be used.

    Raises:
        FileNotFoundError: If no valid configuration file is found.
    """
    if cli_config and os.path.exists(cli_config):
        return cli_config
    elif os.path.exists("..config_db.yaml"):
        return "..config_db.yaml"
    elif os.path.exists("../.config_db.yaml"):
        return ".config_db.yaml"
    else:
        raise FileNotFoundError(
            "No config file found. Expected ..config_db.yaml or .config_db.yaml"
        )


def main():
    """
    Main execution function.

    Performs the following steps:
    1. Parses CLI arguments.
    2. Loads the Excel file containing movement segments.
    3. Prepares the dataframe by selecting and converting necessary columns.
    4. Initializes the InfluxDB extractor.
    5. Iterates over provided durations and extracts matching data chunks.
    6. Saves a summary of the extraction as an Excel file.
    """
    parser = argparse.ArgumentParser(
        description="Extract MS walking segments from InfluxDB."
    )
    parser.add_argument("--input", required=True, help="Path to the input Excel file.")
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where the extracted chunks will be saved.",
    )
    parser.add_argument(
        "--durations",
        nargs="+",
        type=int,
        default=[5, 7, 10, 15, 20],
        help="List of durations in seconds for the chunks (e.g., --durations 5 10 15).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the config YAML file (default: ..config_db.yaml or .config_db.yaml).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="Verbosity level: 0=none, 1=chunk info, 2=rows per leg, 3=summary by foot and type.",
    )
    args = parser.parse_args()

    dataframe = pd.read_excel(args.input, sheet_name="data")
    print(dataframe.head(10).to_string())

    columns_to_keep = ["ry_to_use", "datefrom", "dateuntil", "move_type"]
    dataframe = dataframe[columns_to_keep]

    try:
        dataframe["datefrom"] = pd.to_datetime(dataframe["datefrom"])
        dataframe["dateuntil"] = pd.to_datetime(dataframe["dateuntil"])
    except Exception as e:
        print(f"Error processing date columns: {e}")

    os.makedirs(args.output, exist_ok=True)

    config_path = load_config_path(args.config)
    db = cInfluxDB(config_path=config_path)

    results = []
    for dur in args.durations:
        print(f"\n⏳ Extracting {dur}-second segments...")

        global_summary = {"duration": dur, "extracted": 0, "skipped": 0}

        for _, row in dataframe.iterrows():
            row_summary = db.export_chunks_for_segment(
                datefrom=row["datefrom"],
                dateuntil=row["dateuntil"],
                ry_to_use=row["ry_to_use"],
                move_type=row["move_type"],
                output_dir=args.output,
                chunk_duration=dur,
                verbose=args.verbose,
            )
            global_summary["extracted"] += row_summary["extracted"]
            global_summary["skipped"] += row_summary["skipped"]

        results.append(global_summary)

    df_summary = pd.DataFrame(results)
    print("\n📊 Extraction summary by duration:\n")
    print(df_summary)

    df_summary.to_excel(os.path.join(args.output, "resumen_chunks.xlsx"), index=False)


if __name__ == "__main__":
    main()
