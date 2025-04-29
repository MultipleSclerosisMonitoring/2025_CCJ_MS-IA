import os
import pandas as pd

from datetime import datetime, timedelta, timezone
from InfluxDBms.cInfluxDB import cInfluxDB


# Main function
def main():
    # Read the input Excel file
    dataframe = pd.read_excel("./data/gait_class_references_v1.xlsx", sheet_name="data")

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
    os.makedirs("./output/", exist_ok=True)

    # Initialize InfluxDB extractor
    db = cInfluxDB(config_path="./src/InfluxDBms/config_db.yaml")

    # Try different durations
    durations = [5, 7, 10, 15, 20]
    results = []

    # Initialize the Extractor class
    for dur in durations:
        print(f"\n⏳ Extrayendo segmentos de {dur} segundos...")
        # Extract
        summary = db.export_chunks_to_excel(dataframe, "./output/", dur)
        summary["duration"] = dur
        results.append(summary)

    # Summary Table
    df_summary = pd.DataFrame(results)
    print("\n📊 Resumen de extracción por duración:\n")
    print(df_summary)

    # Output Excel
    df_summary.to_excel("./output/resumen_chunks.xlsx", index=False)


if __name__ == "__main__":
    main()
