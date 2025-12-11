from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Literal, Tuple

import numpy as np
import tensorflow as tf

from src.data import AVConfig, download_adjusted_close, make_windows, split_train_val


ModelName = Literal["lstm", "hybrid"]


def _minmax_fit(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-8:
        mx = mn + 1.0
    return mn, mx


def _minmax_transform(x: np.ndarray, mn: float, mx: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _minmax_inverse(x: np.ndarray, mn: float, mx: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x * (mx - mn) + mn


def _build_lstm(window_size: int) -> tf.keras.Model:
    inp = tf.keras.layers.Input(shape=(window_size, 1))
    x = tf.keras.layers.LSTM(64)(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model


def _build_hybrid(window_size: int) -> tf.keras.Model:
    inp = tf.keras.layers.Input(shape=(window_size, 1))
    x = tf.keras.layers.Conv1D(32, 3, padding="causal", activation="relu")(inp)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model


def train(
    *,
    symbol: str,
    api_key: str = "",
    outputsize: str = "compact",
    window_size: int = 60,
    epochs: int = 5,
    model_name: ModelName = "lstm",
    max_points: int = 2500,  # train on recent ~10 years (trading days)
) -> Dict[str, Any]:
    # 1) Load series (uses cache -> AlphaVantage -> Stooq)
    close_series, source_range = download_adjusted_close(
        AVConfig(api_key=api_key or "", symbol=symbol.upper(), outputsize=outputsize)
    )

    # 2) Use only recent points to avoid huge historical scale drift
    if len(close_series) > max_points:
        close_series = close_series.iloc[-max_points:]

    dates = close_series.index
    close = close_series.values.astype(np.float32)

    # 3) Fit scaler on the *training* portion only (avoid leakage)
    split_raw = int(len(close) * 0.7)
    mn, mx = _minmax_fit(close[:split_raw])

    close_scaled = _minmax_transform(close, mn, mx)

    # 4) Make windows on scaled series
    X, y, last_window = make_windows(close_scaled, window_size=window_size)
    X_train, X_val, y_train, y_val = split_train_val(X, y, train_split=0.7)

    # Align dates with y (y starts at index window_size)
    y_dates = dates[window_size:]
    split_idx = len(X_train)
    val_dates = y_dates[split_idx:]

    # 5) Build model
    if model_name == "lstm":
        model = _build_lstm(window_size)
    else:
        model = _build_hybrid(window_size)

    # 6) Train
    cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=int(epochs),
        batch_size=64,
        verbose=0,
        callbacks=[cb],
    )

    # 7) Predict (still scaled)
    y_pred_scaled = model.predict(X_val, verbose=0).reshape(-1)
    y_true_scaled = y_val.reshape(-1)

    next_scaled = model.predict(last_window.reshape(1, window_size, 1), verbose=0).reshape(-1)[0]

    # 8) Inverse back to dollars
    y_pred = _minmax_inverse(y_pred_scaled, mn, mx)
    y_true = _minmax_inverse(y_true_scaled, mn, mx)
    next_pred = float(_minmax_inverse(np.array([next_scaled], dtype=np.float32), mn, mx)[0])

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return {
        "symbol": symbol.upper(),
        "source_range": source_range,
        "n_points_used": int(len(close_series)),
        "window_size": int(window_size),
        "epochs": int(epochs),
        "model": model_name,
        "val_dates": val_dates,
        "y_true": y_true,
        "y_pred": y_pred,
        "next_pred": next_pred,
        "mae": mae,
        "rmse": rmse,
        "history": history.history,
    }
