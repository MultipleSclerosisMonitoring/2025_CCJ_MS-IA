import os
import h5py
from collections import Counter

# Ruta local de tus datasets balanceados
base_path = r"C:\Users\epepc\OneDrive\Escritorio\TFG\2025_CCJ_MS-IA\data_balanced"

# Recorre todos los archivos HDF5 de la carpeta
for filename in sorted(os.listdir(base_path)):
    if filename.endswith(".hdf5"):
        file_path = os.path.join(base_path, filename)
        with h5py.File(file_path, "r") as f:
            y = f["y"][:]
            total = len(y)
            counts = Counter(y)
            print(f"\n📁 {filename}")
            print(f"  Total segments: {total}")
            print(f"  Class 0: {counts.get(0, 0)}")
            print(f"  Class 1: {counts.get(1, 0)}")
