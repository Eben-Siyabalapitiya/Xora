"""
train.py — Xora position regression model
Trains a model to predict x,y position directly from CSI data.
"""

import os
import numpy as np

NUM_SUB = 52
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xora_model.h5")
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data")


def _load_position_dataset(data_dir: str):
    """Load all pos_*.npy files — each row is [csi(156), x, y]"""
    X, Y = [], []
    for f in os.listdir(data_dir):
        if f.startswith("pos_") and f.endswith(".npy"):
            arr = np.load(os.path.join(data_dir, f))
            print(f"[train] loaded {f}: {arr.shape[0]} samples")
            X.append(arr[:, :-2])   # CSI features
            Y.append(arr[:, -2:])   # x, y labels
    if not X:
        raise ValueError("No position training data found. Record zone data first.")
    return np.vstack(X).astype(np.float32), np.vstack(Y).astype(np.float32)


def train_and_save(data_dir: str = DATA_DIR, model_path: str = MODEL_PATH) -> float:
    from tensorflow import keras

    X, Y = _load_position_dataset(data_dir)

    # Normalise features
    mu  = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X   = (X - mu) / std

    # Save normalisation stats alongside model
    np.save(model_path.replace(".h5", "_norm.npy"), {"mu": mu, "std": std})

    # Reshape for LSTM: (N, 1, features)
    X = X[:, np.newaxis, :]

    # Shuffle
    idx = np.random.permutation(len(X))
    X, Y = X[idx], Y[idx]

    split  = int(0.85 * len(X))
    X_tr, X_val = X[:split], X[split:]
    Y_tr, Y_val = Y[:split], Y[split:]

    # Regression model: LSTM → Dense → (x, y)
    inp = keras.Input(shape=(1, NUM_SUB * 3))
    x   = keras.layers.LSTM(128, return_sequences=False)(inp)
    x   = keras.layers.Dense(128, activation="relu")(x)
    x   = keras.layers.Dropout(0.3)(x)
    x   = keras.layers.Dense(64, activation="relu")(x)
    out = keras.layers.Dense(2, activation="sigmoid")(x)   # x,y both 0-1
    model = keras.Model(inp, out)

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    print("[train] training position regression model...")
    model.fit(
        X_tr, Y_tr,
        validation_data=(X_val, Y_val),
        epochs=80,
        batch_size=64,
        verbose=1,
        callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    )

    loss, mae = model.evaluate(X_val, Y_val, verbose=0)
    print(f"[train] val MAE: {mae:.4f} (avg position error ~{mae*100:.1f}% of room)")
    model.save(model_path)
    print(f"[train] model saved → {model_path}")
    # Return accuracy-like score (1 - normalised MAE)
    return float(1.0 - mae)


def load_model(model_path: str = MODEL_PATH):
    from tensorflow import keras
    return keras.models.load_model(model_path)


def predict_position(model, csi_row: np.ndarray, model_path: str = MODEL_PATH):
    """Given a CSI feature vector (156,), return predicted (x, y)."""
    norm_path = model_path.replace(".h5", "_norm.npy")
    if os.path.exists(norm_path):
        norm = np.load(norm_path, allow_pickle=True).item()
        csi_row = (csi_row - norm["mu"]) / (norm["std"] + 1e-8)
    x = csi_row[np.newaxis, np.newaxis, :]
    pred = model.predict(x, verbose=0)[0]
    return float(pred[0]), float(pred[1])


# Legacy classifier interface (kept so old train button still works)
def train():
    return train_and_save()