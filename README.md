# 2025_CCJ_MS-IA
Gait identification using AI and Deep Learning


# InfluxDBMS - Gait Data Extraction and Segmentation from InfluxDB

This project is part of a Bachelor's Thesis (TFG) aimed at automatically extracting human gait segments in patients with Multiple Sclerosis (MS), using sensor data stored in an InfluxDB database.

The extracted segments are prepared in structured files to train AI/deep learning models for gait phase detection and mobility analysis.

---

## Project Structure

```
src/
├── InfluxDBms/           # Module for querying and exporting from InfluxDB
│   └── cInfluxDB.py
├── plotting/             # Scripts for data visualization
│   └── plot_chunks.py
main.py                   # Main CLI extraction script
pyproject.toml            # Poetry project definition
```

---

## Installation (using Poetry)

```bash
# Clone the repository
git clone https://github.com/MultipleSclerosisMonitoring/2025_CCJ_MS-IA
cd 2025_CCJ_MS-IA


# Install dependencies using Poetry
poetry install
```

---

## Configuration

1. Create a `.config_db.yaml` file (DO NOT commit to GitHub) with the following structure:

```yaml
influxdb:
  bucket: "Gait/autogen"
  org: "UPM"
  token: "YOUR_PRIVATE_TOKEN"
  url: "https://YOUR_SERVER_IP:8086"
```
You can use the provided config.yaml as a template.

2. Make sure `.config_db.yaml` is listed in `.gitignore`.

---

## Run the Extraction Script

```bash
python main.py \
  --input ./data/segments.xlsx \
  --output ./output \
  --durations 5 10 15 \
  --verbose 2
```

- `--input`: Excel file with time-labeled movement segments
- `--output`: Folder where extracted chunks will be saved
- `--durations`: List of chunk durations in seconds
- `--verbose`: Level of detail in output (0 to 3)

---

## Visualize Chunked Signals

```bash
python src/plotting/plot_chunks.py --input ./output --output ./plots --no-show
```

This generates PNG plots per subject and leg from the exported `.xlsx` chunks.

---

## License

This project is licensed under the MIT License.
