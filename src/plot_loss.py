import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).resolve().parent
log_path = script_dir.parent / "models" / "base_run" / "training_log.csv"


# Ruta al archivo de log de entrenamiento
log_path = "models/base_run/training_log.csv"

# Cargar CSV
df = pd.read_csv(log_path)

# Graficar curvas de pérdida
plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["loss"], label="Training Loss")
plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Transformer Autoencoder Training Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
