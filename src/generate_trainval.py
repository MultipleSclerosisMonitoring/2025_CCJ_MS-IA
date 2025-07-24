import numpy as np
import h5py

# Cargar índices de test
test_idx = np.load("models_len295/fold_4/test_indices.npy")

# Cargar datos completos
with h5py.File("data_balanced/dataset_balanced_len295_50Hz.hdf5", "r") as f:
    X = f["X"][:]
    y = f["y"][:]

# Crear conjunto sin test
mask = np.ones(len(X), dtype=bool)
mask[test_idx] = False

X_trainval = X[mask]
y_trainval = y[mask]

# Guardar como nuevo HDF5
with h5py.File("data_balanced/len295_trainval_only.hdf5", "w") as f:
    f.create_dataset("X", data=X_trainval)
    f.create_dataset("y", data=y_trainval)
