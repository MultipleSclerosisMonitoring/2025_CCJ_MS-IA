import os
import random
import numpy as np
import argparse
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Add,
    LayerNormalization,
    GlobalAveragePooling1D,
    RepeatVector,
    TimeDistributed,
    MultiHeadAttention,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam


def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_and_normalize(path: str):
    X = np.load(path)
    assert len(X.shape) == 3, "❌ X must have shape (samples, timesteps, features)"
    print(f"✅ Loaded X with shape: {X.shape}")

    flat = X.reshape(-1, X.shape[-1])
    scaler = StandardScaler().fit(flat)
    X_scaled = scaler.transform(flat).reshape(X.shape)

    Path("models").mkdir(exist_ok=True)
    np.save("models/scaler_mean.npy", scaler.mean_)
    np.save("models/scaler_scale.npy", scaler.scale_)

    return X_scaled


def transformer_encoder(inputs, head_size=64, num_heads=2, ff_dim=128, dropout=0.1):
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(inputs.shape[-1])(ff)
    ff = Dropout(dropout)(ff)
    x = Add()([x, ff])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x


def build_transformer_autoencoder(
    timesteps, features, head_size=64, num_heads=2, ff_dim=128, num_blocks=2
):
    inputs = Input(shape=(timesteps, features))
    x = inputs
    for _ in range(num_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim)

    encoded = GlobalAveragePooling1D(name="encoder_output")(x)
    x = RepeatVector(timesteps)(encoded)

    for _ in range(num_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim)

    outputs = TimeDistributed(Dense(features, activation="linear"))(x)
    return Model(inputs, outputs)


def train_model(X, output_dir="models", epochs=50, batch_size=32):
    timesteps, features = X.shape[1], X.shape[2]
    model = build_transformer_autoencoder(timesteps, features)
    model.compile(optimizer=Adam(0.001), loss="mse")

    Path(output_dir).mkdir(exist_ok=True)

    X_train, X_val = train_test_split(X, test_size=0.1, random_state=42)

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(
            f"{output_dir}/best_transformer_autoencoder.h5", save_best_only=True
        ),
        CSVLogger(f"{output_dir}/training_log.csv"),
    ]

    history = model.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
    )

    model.save(f"{output_dir}/transformer_autoencoder.h5")
    encoder = Model(model.input, model.get_layer("encoder_output").output)
    encoder.save(f"{output_dir}/encoder_transformer.h5")
    np.save(f"{output_dir}/loss_history.npy", history.history)

    print("\n✅ Training completed and models saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="Path to X_balanced.npy"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", type=str, default="models")
    args = parser.parse_args()

    set_seeds()
    X = load_and_normalize(args.input)
    train_model(
        X, output_dir=args.output, epochs=args.epochs, batch_size=args.batch_size
    )
