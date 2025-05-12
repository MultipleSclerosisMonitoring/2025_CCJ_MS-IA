import os
import pandas as pd
from influxdb_client import InfluxDBClient
from datetime import datetime, timedelta, timezone
import urllib3
import yaml

# Desactiva las advertencias de SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class cInfluxDB:
    def __init__(self, config_path: str, timeout: int = 500_000):
        """
        Initializes the connection to InfluxDB using a YAML configuration file.

        :param config_path: Path to the YAML configuration file.
        :type config_path: str
        :param timeout: Connection timeout in milliseconds.
        :type timeout: int

        """
        # Load the configuration from the YAML file
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Extract the necessary values
        self.bucket = config["influxdb"]["bucket"]
        self.org = config["influxdb"]["org"]
        self.token = config["influxdb"]["token"]
        self.url = config["influxdb"]["url"]

        # Initialises the InfluxDB client
        self.client = InfluxDBClient(
            url=self.url,
            token=self.token,
            org=self.org,
            verify_ssl=False,
            timeout=timeout,
        )
        self.measurement = (
            self.bucket.split("/")[0] if "/" in self.bucket else self.bucket
        )

    def query_data(
        self, from_date: datetime, to_date: datetime, qtok: str, pie: str, metrics=None
    ) -> pd.DataFrame:
        """
        Query data in InfluxDB, pivoting the results to get the metrics in columns.

        :param from_date: Start date (ISO 8601 format: 'YYYYY'-MM-DDTHH:MM:SSZ).
        :type from_date: datetime
        :param to_date: End date (ISO 8601 format: 'YYYYY'-MM-DDTHH:MM:SSZ).
        :type to_date: datetime
        :param qtok: CodeID
        :type qtok: str
        :param pie: Left or Right foot ('Right', 'Left')
        :type pie: str
        :param metrics: List of metrics to query (default: predefined set)
        :type metrics: list[str], optional

        :return: DataFrame with the metrics pivoted on columns, ordered by _time descending.
        :rtype: pd.DataFrame
        """

        from_date_str = (
            from_date.replace(tzinfo=timezone(timedelta(hours=1)))
            .astimezone(timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        )  # UTC con 'Z'
        to_date_str = (
            to_date.replace(tzinfo=timezone(timedelta(hours=1)))
            .astimezone(timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        )  # UTC con 'Z'

        # Default metrics
        if metrics is None:
            metrics = [
                "Ax",
                "Ay",
                "Az",
                "Gx",
                "Gy",
                "Gz",
                "Mx",
                "My",
                "Mz",
                "S0",
                "S1",
                "S2",
            ]

        metrics_str = " or ".join([f'r._field == "{metric}"' for metric in metrics])
        columns_str = ", ".join([f'"{metric}"' for metric in metrics])

        query = f"""
        from(bucket: "{self.bucket}")
        |> range(start: {from_date_str}, stop: {to_date_str})
        |> filter(fn: (r) => r._measurement == "{self.measurement}")
        |> filter(fn: (r) => {metrics_str})
        |> filter(fn: (r) => r["CodeID"] == "{qtok}" and r["type"] == "SCKS" and r["Foot"] == "{pie}")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> keep(columns: ["_time", {columns_str}])
        """

        try:
            result = self.client.query_api().query(org=self.org, query=query)
        except Exception as e:
            print(f"Error in the query: {str(e)}")
            raise

        # Process the results in a DataFrame
        data = []
        for table in result:
            for record in table.records:
                data.append(record.values)

        df = pd.DataFrame(data).drop(["result", "table"], axis=1)
        return df.sort_values(by="_time", ascending=False).reset_index(drop=True)

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
        Query data in InfluxDB with aggregateWindow, pivoting the results to get metrics as columns.

        :param from_date: Start datetime (ISO 8601 format: 'YYYY-MM-DDTHH:MM:SSZ').
        :type from_date: datetime
        :param to_date: End datetime (ISO 8601 format: 'YYYY-MM-DDTHH:MM:SSZ').
        :type to_date: datetime
        :param window_size: Aggregation window size (default: '20ms').
        :type window_size: str
        :param qtok: CodeID (required).
        :type qtok: str
        :param pie: Left or Right foot ('Right', 'Left') (required).
        :type pie: str
        :param metrics: List of metrics to query (default: predefined set).
        :type metrics: list[str], optional
        :return: DataFrame with metrics as columns, ordered by _time.
        :rtype: pd.DataFrame
        """

        if not qtok or not pie:
            raise ValueError(
                "Los argumentos 'qtok' y 'pie' son obligatorios para esta consulta."
            )

        from_date_str = from_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        to_date_str = to_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Default metrics
        if metrics is None:
            metrics = [
                "Ax",
                "Ay",
                "Az",
                "Gx",
                "Gy",
                "Gz",
                "Mx",
                "My",
                "Mz",
                "S0",
                "S1",
                "S2",
            ]

        metrics_str = " or ".join([f'r._field == "{metric}"' for metric in metrics])
        columns_str = ", ".join([f'"{metric}"' for metric in metrics])

        query = f"""
        from(bucket: "{self.bucket}")
            |> range(start: time(v: "{from_date_str}"), stop: time(v: "{to_date_str}"))
            |> filter(fn: (r) => r._measurement == "{self.measurement}")
            |> filter(fn: (r) => {metrics_str})
            |> filter(fn: (r) => r["CodeID"] == "{qtok}" and r["type"] == "SCKS" and r["Foot"] == "{pie}")
            |> group(columns: ["_field"])
            |> aggregateWindow(every: {window_size}, fn: last, createEmpty: true)
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> keep(columns: ["_time", {columns_str}])
        """

        try:
            result = self.client.query_api().query(org=self.org, query=query)
        except Exception as e:
            print(f"Error en la consulta: {str(e)}")
            raise

        # Procesar los resultados en un DataFrame
        data = []
        for table in result:
            for record in table.records:
                data.append(record.values)

        df = pd.DataFrame(data).drop(["result", "table"], axis=1)

        # Asegurar que todas las métricas están presentes en el DataFrame
        for col in ["_time"] + metrics:
            if col not in df:
                df[col] = None  # Rellenar con None si falta alguna columna

        return df.sort_values(by="_time", ascending=False).reset_index(drop=True)

    def extract_ms_by_codeid_leg(
        self,
        from_time: datetime,
        until_time: datetime,
        qtok: str,
        leg: str,
        output_file: str,
    ):
        """
        Exports data for a given leg and subject code from InfluxDB to an Excel file.

        :param from_time: Start time of the segment.
        :param until_time: End time of the segment.
        :param qtok: Subject or test identifier code.
        :param leg: 'Left' or 'Right' leg.
        :param output_file: Path to save the resulting Excel file.
        """

        if not isinstance(from_time, datetime):
            raise Exception(f"from_time must be datetime: {from_time}")
        if not isinstance(until_time, datetime):
            raise Exception(f"until_time must be datetime: {until_time}")

        try:
            df = self.query_data(from_time, until_time, qtok=qtok, pie=leg)
            # Convert to GMT+1 and drop timezone info
            gmt_plus_1 = timezone(timedelta(hours=1))
            df["_time"] = df["_time"].dt.tz_convert(gmt_plus_1).dt.tz_localize(None)

            print(f"Results of the query: Dataset size {df.shape}")
            df_sorted = df.sort_values(by="_time", ascending=False)
            df_sorted.to_excel(output_file)
        except Exception as exc:
            raise Exception(f"Error querying data: {exc}")

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
        Procesa un segmento individual del Excel y extrae los chunks por pierna.
        """
        chunk_td = timedelta(seconds=chunk_duration)
        current_time = datefrom
        extracted = 0
        skipped = 0
        resumen_por_tipo = {}

        os.makedirs(output_dir, exist_ok=True)

        while current_time + chunk_td <= dateuntil:
            chunk_end = current_time + chunk_td
            for leg in ["Left", "Right"]:
                filename = (
                    f"{ry_to_use}+{move_type}+{current_time.strftime('%Y-%m-%d_%H-%M-%S')}"
                    f"+{chunk_end.strftime('%Y-%m-%d_%H-%M-%S')}+{leg}.xlsx"
                )
                output_file = os.path.join(output_dir, filename)

                try:
                    if verbose >= 1:
                        print(f"[Chunk] {filename}")

                    self.extract_ms_by_codeid_leg(
                        current_time, chunk_end, ry_to_use, leg, output_file
                    )
                    extracted += 1

                    if verbose >= 2:
                        df = pd.read_excel(output_file)
                        print(f"   └── {leg}: {df.shape[0]} filas")

                    if verbose >= 3:
                        key = f"{leg}-{move_type}"
                        resumen_por_tipo[key] = resumen_por_tipo.get(key, 0) + 1

                except Exception as exc:
                    print(f"⚠️ Error al procesar {filename}: {exc}")

            current_time = chunk_end

        if current_time < dateuntil:
            skipped += 1

        if verbose >= 3 and resumen_por_tipo:
            print("\n📦 Resumen parcial por pie y tipo:")
            for key, count in resumen_por_tipo.items():
                print(f"  - {key}: {count} ficheros")

        return {"extracted": extracted, "skipped": skipped, "details": resumen_por_tipo}
