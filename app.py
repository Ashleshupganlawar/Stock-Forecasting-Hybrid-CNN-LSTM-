import os
import streamlit as st
import pandas as pd
import plotly.express as px

from src.data import AVConfig, download_adjusted_close
from src.train_tf import train

st.set_page_config(page_title="Stock Market Nostradamus (Rebuilt)", layout="wide")
st.title("Stock Market Nostradamus (Rebuilt)")

api_key = os.getenv("ALPHAVANTAGE_API_KEY", "")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Controls")

    symbol = st.text_input("Ticker", value="AAPL")

    # Keep the label the same, but the backend can choose source (AlphaVantage vs stooq) depending on your src/data.py
    outputsize = st.selectbox("AlphaVantage data size", ["compact", "full"], index=0)

    # Defaults that worked
    window_size = st.slider("Window size", 5, 200, 60)
    epochs = st.slider("Epochs", 1, 50, 15)
    model_name = st.selectbox("Model", ["lstm", "hybrid"], index=0)  # lstm default

    load_btn = st.button("Load data")
    train_btn = st.button("Train + predict")

# ---------------- Helpers ----------------
def load_data(sym: str, out: str):
    close, src = download_adjusted_close(
        AVConfig(api_key=api_key or "", symbol=sym.upper(), outputsize=out)
    )
    df = pd.DataFrame({"date": close.index, "close": close.values})
    return df, src


# ---------------- Session state ----------------
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.src = None

# ---------------- Load data ----------------
if load_btn or st.session_state.df is None:
    try:
        df, src = load_data(symbol, outputsize)
        st.session_state.df = df
        st.session_state.src = src
        st.success(f"Loaded {symbol.upper()} data: {src} ({len(df):,} points)")
    except Exception as e:
        st.error("Failed to load data.")
        st.exception(e)

df = st.session_state.df
src = st.session_state.src

if df is not None:
    st.caption(f"Data range: {df['date'].min().date()} → {df['date'].max().date()} | points: {len(df):,}")
    fig = px.line(df, x="date", y="close", title=f"{symbol.upper()} Adjusted Close")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------- Train + predict ----------------
if train_btn:
    try:
        st.info("Training started… this can take a bit the first time TensorFlow loads.")
        st.caption(f"Run settings — Model: {model_name.upper()} | Window: {window_size} | Epochs: {epochs}")

        with st.spinner("Training model…"):
            res = train(
                symbol=symbol,
                api_key=api_key or "",
                outputsize=outputsize,
                window_size=window_size,
                epochs=epochs,
                model_name=model_name,
                max_points=2500,
            )

        st.subheader(f"{res['symbol']}: Actual vs Predicted (validation window)")

        out_df = pd.DataFrame(
            {
                "date": pd.to_datetime(res["val_dates"]),
                "Actual": res["y_true"],
                "Predicted": res["y_pred"],
            }
        )
        out_long = out_df.melt(id_vars=["date"], var_name="variable", value_name="value")
        fig2 = px.line(out_long, x="date", y="value", color="variable")
        st.plotly_chart(fig2, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Next-day forecast (Close)", f"{res['next_pred']:.2f}")
        c2.metric("MAE (val)", f"{res['mae']:.2f}")
        c3.metric("RMSE (val)", f"{res['rmse']:.2f}")

        st.caption(f"Training used last {res['n_points_used']:,} points to keep scale stable.")

        # Training curve
        hist = res.get("history", {})
        if "loss" in hist and "val_loss" in hist:
            hdf = pd.DataFrame({"loss": hist["loss"], "val_loss": hist["val_loss"]})
            st.subheader("Training curve")
            st.line_chart(hdf)

        # Optional: save predictions every run
        out_df.to_csv("last_run_predictions.csv", index=False)

    except Exception as e:
        st.error("Training failed.")
        st.exception(e)
