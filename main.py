import argparse
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from InfluxDBms.cInfluxDB import cInfluxDB


def load_config_path(cli_config: str = None) -> str:
    """
    Determines which configuration file to use.
    Priority: CLI argument > ..config_db.yaml > .config_db.yaml
    """
    if cli_config and os.path.exists(cli_config):
        return cli_config
    elif os.path.exists("..config_db.yaml"):
        return "..config_db.yaml"
    elif os.path.exists(".config_db.yaml"):
        return ".config_db.yaml"
    else:
        raise FileNotFoundError(
            "No config file found. Expected ..config_db.yaml or .config_db.yaml"
        )


# Main function
def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Extract MS data chunks from InfluxDB."
    )
    parser.add_argument("--input", required=True, help="Path to the input Excel file.")
    parser.add_argument(
        "--output", required=True, help="Directory to save the extracted chunks."
    )
    parser.add_argument(
        "--durations",
        nargs="+",
        type=int,
        default=[5, 7, 10, 15, 20],
        help="Duration in seconds (ex: --durations 5 10 15)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML file (default: ..config_db.yaml or .config_db.yaml)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="Verbosity level: 0=nothing, 1=chunk info, 2=rows per leg, 3=summary by foot and type",
    )
    args = parser.parse_args()

    # Read the input Excel file
    dataframe = pd.read_excel(args.input, sheet_name="data")

    # Check the columns and the first n rows
    print(dataframe.head(10).to_string())

    # Select only required columns
    columns_to_keep = ["ry_to_use", "datefrom", "dateuntil", "move_type"]
    dataframe = dataframe[columns_to_keep]

    # Ensure correct data types
    try:
        dataframe["datefrom"] = pd.to_datetime(dataframe["datefrom"])
        dataframe["dateuntil"] = pd.to_datetime(dataframe["dateuntil"])
    except Exception as e:
        print(f"Error processing date columns: {e}")

    # Ensure the output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Initialize InfluxDB extractor
    config_path = load_config_path(args.config)
    db = cInfluxDB(config_path=config_path)

    # Initialize the Extractor class with the specified durations
    results = []
    for dur in args.durations:
        print(f"\n⏳ Extrayendo segmentos de {dur} segundos...")

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

    # Summary Table
    df_summary = pd.DataFrame(results)
    print("\n📊 Resumen de extracción por duración:\n")
    print(df_summary)

    # Output Excel
    df_summary.to_excel(os.path.join(args.output, "resumen_chunks.xlsx"), index=False)


if __name__ == "__main__":
    main(
