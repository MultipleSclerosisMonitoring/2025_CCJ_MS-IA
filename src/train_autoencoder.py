import os
import random
import numpy as np
import argparse
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, RepeatVector, TimeDistributed, Dense,
    Bidirectional, Dropout, LayerNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger


def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data(path: str):
    X = np.load(path)
    assert len(X.shape) == 3, "❌ X must have shape (samples, timesteps, features)"
    print(f"✅ Loaded X with shape: {X.shape}")

    original_shape = X.shape
    X_flat = X.reshape(-1, X.shape[-1])
    scaler = StandardScaler().fit(X_flat)
    X_scaled = scaler.transform(X_flat).reshape(original_shape)

    Path("models").mkdir(exist_ok=True)
    np.save("models/scaler_mean.npy", scaler.mean_)
    np.save("models/scaler_scale.npy", scaler.scale_)

    return X_scaled


def build_autoencoder(timesteps: int, features: int, encoding_dim: int) -> Model:
    inputs = Input(shape=(timesteps, features))
    x = Bidirectional(LSTM(encoding_dim, activation='tanh'), name="encoder_lstm")(inputs)
    x = LayerNormalization()(x)
    x = RepeatVector(timesteps)(x)
    x = LSTM(encoding_dim, activation='tanh', return_sequences=True)(x)
    x = Dropout(0.2)(x)
    outputs = TimeDistributed(Dense(features, activation="linear"))(x)
    return Model(inputs, outputs)


def train_autoencoder(X, encoding_dim=64, batch_size=32, epochs=50):
    timesteps, features = X.shape[1], X.shape[2]
    model = build_autoencoder(timesteps, features, encoding_dim)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    Path("models").mkdir(exist_ok=True)

    with open("models/model_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint("models/best_autoencoder.h5", save_best_only=True),
        ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
        CSVLogger("models/training_log.csv")
    ]

    ds = tf.data.Dataset.from_tensor_slices((X, X)).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    history = model.fit(
        ds,
        epochs=epochs,
        callbacks=callbacks
    )

    model.save("models/autoencoder_lstm.h5")
    np.save("models/loss_history.npy", history.history)

    # Acceder al encoder por nombre
    encoder = Model(model.input, model.get_layer("encoder_lstm").output)
    encoder.save("models/encoder_lstm.h5")

    print("✅ Training completed. Model and artifacts saved.")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to X_chunks.npy")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--encoding_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    set_seeds()
    X = load_data(args.input)
    train_autoencoder(X, encoding_dim=args.encoding_dim, batch_size=args.batch_size, epochs=args.epochs)
