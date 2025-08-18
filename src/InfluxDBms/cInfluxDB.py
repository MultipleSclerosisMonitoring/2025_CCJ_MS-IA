"""
cInfluxDB.py

Provides a class interface to connect to InfluxDB, query sensor data, and export it for analysis.
Supports raw and windowed queries, chunk segmentation, and Excel export functionalities.
"""

import os
import pandas as pd
from influxdb_client import InfluxDBClient
from datetime import datetime, timedelta, timezone
import urllib3
import yaml

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class cInfluxDB:
    def __init__(self, config_path: str, timeout: int = 500_000):
        """
        Initializes the connection to InfluxDB using a YAML configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.
            timeout (int): Connection timeout in milliseconds. Defaults to 500000.
        """
        ...

    def query_data(
        self, from_date: datetime, to_date: datetime, qtok: str, pie: str, metrics=None
    ) -> pd.DataFrame:
        """
        Queries raw sensor data from InfluxDB, filtering by subject code, foot, and time interval.

        Args:
            from_date (datetime): Start date in datetime format (local time).
            to_date (datetime): End date in datetime format (local time).
            qtok (str): Subject or CodeID identifier.
            pie (str): Foot to query ('Left' or 'Right').
            metrics (list[str], optional): List of metrics to retrieve. Defaults to standard sensor metrics.

        Returns:
            pd.DataFrame: DataFrame with columns for each metric and timestamps, sorted by time descending.

        Raises:
            Exception: If the query to InfluxDB fails.
        """
        ...

    def query_with_aggregate_window(
        self,
        from_date: datetime,
        to_date: datetime,
        window_size: str = "20ms",
        qtok: str = None,
        pie: str = None,
        metrics=None,
    ) -> pd.DataFrame:
        """
        Queries sensor data with aggregation using an aggregate window (e.g., last value per 20ms interval).

        Args:
            from_date (datetime): Start time in UTC string format.
            to_date (datetime): End time in UTC string format.
            window_size (str): Window duration for aggregation. Defaults to "20ms".
            qtok (str): Subject or CodeID identifier.
            pie (str): Foot to query ('Left' or 'Right').
            metrics (list[str], optional): Metrics to include. Defaults to all relevant metrics.

        Returns:
            pd.DataFrame: Aggregated data with time and sensor columns, sorted descending by time.

        Raises:
            ValueError: If `qtok` or `pie` are not provided.
            Exception: If the query fails.
        """
        ...

    def extract_ms_by_codeid_leg(
        self,
        from_time: datetime,
        until_time: datetime,
        qtok: str,
        leg: str,
        output_file: str,
    ):
        """
        Exports sensor data for a specific leg and subject within a time range to an Excel file.

        Args:
            from_time (datetime): Start time of the segment.
            until_time (datetime): End time of the segment.
            qtok (str): CodeID or subject identifier.
            leg (str): Leg name, either 'Left' or 'Right'.
            output_file (str): File path to save the extracted Excel.

        Raises:
            Exception: If input types are incorrect or query fails.
        """
        ...

    def export_chunks_for_segment(
        self,
        datefrom: datetime,
        dateuntil: datetime,
        ry_to_use: str,
        move_type: str,
        output_dir: str,
        chunk_duration: int,
        verbose: int = 1,
    ) -> dict:
        """
        Processes a motion segment and exports sensor data chunks for each leg.

        Args:
            datefrom (datetime): Start timestamp of the segment.
            dateuntil (datetime): End timestamp of the segment.
            ry_to_use (str): CodeID to use for naming files and queries.
            move_type (str): Type of movement (e.g., 'Walk', 'Rest').
            output_dir (str): Directory to save the output chunk files.
            chunk_duration (int): Duration of each chunk in seconds.
            verbose (int): Verbosity level (0 to 3).

        Returns:
            dict: Dictionary with counts of extracted and skipped chunks, and optional details by foot/movement type.
        """
        ...
