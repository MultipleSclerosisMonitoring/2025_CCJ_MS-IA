import os
import pandas as pd

from datetime import datetime, timedelta, timezone
from InfluxDBms.cInfluxDB import cInfluxDB


class InfluxDBExtractor:
    """
    Handles querying InfluxDB and exporting results.
    """

    def __init__(self, config_path):
        self.config_path = config_path
        try:
            self.idb = cInfluxDB(config_path=config_path)
        except FileNotFoundError as exc:
            raise Exception(f"Config file not found. Error: {exc}")
        except Exception as exc:
            raise Exception(f"Error initializing cInfluxDB: {exc}")

    def query_and_export(
        self,
        from_time: datetime,
        until_time: datetime,
        qtok: str,
        leg: str,
        output_file: str,
    ):
        """
        Query InfluxDB for a time range and export to an Excel file.
        """

        if not isinstance(from_time, datetime):
            raise Exception(f"from_time must be datetime: {from_time}")
        if not isinstance(until_time, datetime):
            raise Exception(f"until_time must be datetime: {until_time}")

        try:
            df = self.idb.query_data(from_time, until_time, qtok=qtok, pie=leg)

            # Convert to GMT+1 and drop timezone info
            gmt_plus_1 = timezone(timedelta(hours=1))
            df["_time"] = df["_time"].dt.tz_convert(gmt_plus_1).dt.tz_localize(None)

            print(f"Results of the query: Dataset size {df.shape}")
            df_sorted = df.sort_values(by="_time", ascending=False)
            df_sorted.to_excel(output_file)
        except Exception as exc:
            raise Exception(f"Error querying data: {exc}")


class ChunkExtractor:
    """
    Splits data into time chunks and runs extraction for each.
    """

    def __init__(self, df, output_dir: str, chunk_duration: int, config_path: str):
        self.df = df
        self.output_dir = output_dir
        self.chunk_duration_td = timedelta(seconds=chunk_duration)
        self.extractor = InfluxDBExtractor(config_path)
        self.extracted_count = 0
        self.not_extracted_count = 0

    def create_chunks_and_extract(self):
        """
        Iterate through DataFrame rows, split into chunks, and extract.
        """

        for _, row in self.df.iterrows():
            datefrom = row["datefrom"]
            dateuntil = row["dateuntil"]
            move_type = row["move_type"]
            current_time = datefrom

            while current_time + self.chunk_duration_td <= dateuntil:
                chunk_end_time = current_time + self.chunk_duration_td

                for leg in ["Left", "Right"]:
                    filename = (
                        f"{move_type}+{current_time.strftime('%Y-%m-%d_%H-%M-%S')}"
                        f"+{chunk_end_time.strftime('%Y-%m-%d_%H-%M-%S')}+{leg}.xlsx"
                    )
                    output_file = os.path.join(self.output_dir, filename)

                    try:
                        self.extractor.query_and_export(
                            current_time,
                            chunk_end_time,
                            row["ry_to_use"],
                            leg,
                            output_file,
                        )
                        self.extracted_count += 1
                    except Exception as exc:
                        print(exc)

                current_time = chunk_end_time

            # Count any remaining time not fitting a full chunk
            if current_time < dateuntil:
                self.not_extracted_count += 1

        print(f"Total references extracted: {self.extracted_count}")
        print(f"Total references not extracted: {self.not_extracted_count}")


# Main function

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

# Initialize the Extractor class
extractor = ChunkExtractor(
    df=dataframe,
    output_dir="./output/",
    chunk_duration=5,
    config_path="./InfluxDBms/config_db.yaml",
)

# Extract
extractor.create_chunks_and_extract()
