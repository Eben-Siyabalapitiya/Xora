"""
train.py
Trains a CNN+LSTM model on collected CSI data.
Called automatically by the dashboard — or run manually:

    python train.py
"""

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # suppress TF noise

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collect_data import load_all

MODEL_PATH = 'xora_model.keras'

def build_model(input_shape, num_classes):
    """
    CNN+LSTM model for CSI time-series classification.
    input_shape: (boards, window, subcarriers) e.g. (3, 50, 52)
    """
    inp = keras.Input(shape=input_shape)

    # Flatten boards into channels: (window, boards*subcarriers)
    x = keras.layers.Reshape((input_shape[1], input_shape[0] * input_shape[2]))(inp)

    # CNN — extract spatial patterns across subcarriers
    x = keras.layers.Conv1D(64,  kernel_size=3, activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling1D(2)(x)

    # LSTM — extract temporal patterns over time
    x = keras.layers.LSTM(64)(x)

    # Output
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train(on_epoch_end=None):
    """
    Load data, train model, save to disk.
    on_epoch_end(epoch, logs) called after each epoch — used by dashboard.
    Returns final validation accuracy.
    """
    print('[train] loading data...')
    X, y, num_classes = load_all()

    # Normalize
    X = X.astype(np.float32)
    X = (X - X.mean()) / (X.std() + 1e-8)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f'[train] train: {len(X_train)}  val: {len(X_val)}')
    print(f'[train] input shape: {X_train.shape[1:]}')

    model = build_model(X_train.shape[1:], num_classes)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=7, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            patience=3, factor=0.5, verbose=1
        ),
    ]

    if on_epoch_end:
        class ProgressCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                on_epoch_end(epoch, logs)
        callbacks.append(ProgressCallback())

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # Save
    model.save(MODEL_PATH)
    print(f'[train] model saved → {MODEL_PATH}')

    val_acc = max(history.history['val_accuracy'])
    print(f'[train] best val accuracy: {val_acc:.4f}')
    return val_acc


def load_model():
    if os.path.exists(MODEL_PATH):
        print(f'[train] loading model from {MODEL_PATH}')
        return keras.models.load_model(MODEL_PATH)
    return None


if __name__ == '__main__':
    acc = train()
    print(f'\nFinal accuracy: {acc*100:.1f}%')