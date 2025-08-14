# 2025_CCJ_MS-IA
Gait identification using AI and Deep Learning


# InfluxDBMS - Gait Data Extraction and Segmentation from InfluxDB

This project is part of a Bachelor's Thesis (TFG) aimed at automatically extracting human gait segments in patients with Multiple Sclerosis (MS), using sensor data stored in an InfluxDB database.

The extracted segments are prepared in structured files to train AI/deep learning models for gait phase detection and mobility analysis.

---

## Project Structure

src/ 
- `InfluxDBms/ `: Query and clean data from InfluxDB
- `plotting/ `: Plot and visualize latent representations
- `export_balanced_chunks.py`:Segment and balance time series
- `main.py`: Orchestrates the data pipeline 
- `encode_latent_transformer.py`:Compress sequences using Transformer autoencoder 
- `train_autoencoder.py`: Train Transformer-based autoencoder 
- `evaluate_latent_classifier.py`: Evaluate latent classifiers with metrics 
- `train_logistic_classifier.py`: Train logistic classifier on latents 
- `inference.py`: Predict 'walk' vs 'no_walk' from new input

Other folders:
- `data_balanced/`, `latent_data/`, `output/`, `plots/`: intermediate artifacts and outputs
- `docs/`: Sphinx documentation folder (see below)
- `README.md`: Project overview (this file)
- `requirements.txt` / `pyproject.toml`: dependencies and environment
- `pyproject.toml`: poetry project definition

## Documentation

Auto-generated docs available at:  
📎 https://2025-ai-gait-identification.readthedocs.io

You can build it locally with:

```bash
cd docs
make html
start build/html/index.html  # On Windows
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
