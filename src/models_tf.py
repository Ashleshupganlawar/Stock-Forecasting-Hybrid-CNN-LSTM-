from __future__ import annotations
from dataclasses import dataclass
from tensorflow.keras import layers, Model
import tensorflow as tf

@dataclass(frozen=True)
class ModelConfig:
    window_size: int
    features: int = 1
    lstm_units: int = 50
    lstm_layers: int = 3
    dropout: float = 0.2
    cnn_filters: int = 32
    cnn_kernel: int = 10
    dense_units: int = 32

def build_lstm(cfg: ModelConfig) -> Model:
    inp = layers.Input(shape=(cfg.window_size, cfg.features))
    x = inp
    for i in range(cfg.lstm_layers):
        return_seq = i < (cfg.lstm_layers - 1)
        x = layers.LSTM(cfg.lstm_units, return_sequences=return_seq)(x)
        if return_seq:
            x = layers.Dropout(cfg.dropout)(x)
    x = layers.Dense(cfg.dense_units, activation="relu")(x)
    out = layers.Dense(1)(x)
    return Model(inp, out, name="lstm_only")

def build_hybrid(cfg: ModelConfig) -> Model:
    inp = layers.Input(shape=(cfg.window_size, cfg.features))

    xl = inp
    for i in range(cfg.lstm_layers):
        return_seq = i < (cfg.lstm_layers - 1)
        xl = layers.LSTM(cfg.lstm_units, return_sequences=return_seq)(xl)
        if return_seq:
            xl = layers.Dropout(cfg.dropout)(xl)
    xl = layers.Dense(cfg.dense_units, activation="relu")(xl)

    xc = layers.Conv1D(cfg.cnn_filters, kernel_size=cfg.cnn_kernel, padding="same", activation="relu")(inp)
    xc = layers.Conv1D(cfg.cnn_filters, kernel_size=max(3, cfg.cnn_kernel // 2), padding="same", activation="relu")(xc)
    xc = layers.Conv1D(cfg.cnn_filters, kernel_size=3, padding="same", activation="relu")(xc)
    xc = layers.GlobalAveragePooling1D()(xc)
    xc = layers.Dense(cfg.dense_units, activation="relu")(xc)

    x = layers.Concatenate()([xl, xc])
    x = layers.Dense(cfg.dense_units, activation="relu")(x)
    out = layers.Dense(1)(x)
    return Model(inp, out, name="hybrid_cnn_lstm")

def compile_model(model: Model, lr: float = 1e-3) -> Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model
