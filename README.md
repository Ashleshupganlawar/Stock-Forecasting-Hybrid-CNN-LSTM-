# 📈 Stock Forecasting with Hybrid CNN–LSTM (Streamlit App)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vzq5cnaugdeyx3hy8esazw.streamlit.app/)

Interactive web app for **time series stock forecasting** using **LSTM** and **Hybrid CNN–LSTM–style models**, built with Streamlit and TensorFlow.  
The app lets you:

- Load historical stock prices
- Train a forecasting model with tunable hyperparameters
- Visualize **Actual vs Predicted** prices on a validation window
- Inspect the **training curve** (loss vs val_loss)
- See **next-day forecast**, **MAE**, and **RMSE**

> 🔍 This project is for learning and demonstration only. It is **not** financial advice.

---

## 🌐 Live Demo

Open the app in your browser (no setup required):

👉 **https://vzq5cnaugdeyx3hy8esazw.streamlit.app/**

---

## 🔍 What the App Does

### 1. Data loading

- Accepts a **ticker symbol** (e.g. `AAPL`)
- Uses a data-loading pipeline (see `src/data.py`) that can:
  - Pull from **Alpha Vantage** using `ALPHAVANTAGE_API_KEY`
  - Or fall back to cached / alternative data sources (e.g. Stooq), depending on your implementation
- Allows switching **output size** (`compact` / `full`)
- Displays:
  - Historical close price chart
  - Data range & number of points used

### 2. Modeling

From the sidebar, you can control:

- **Model type**
  - `lstm`
  - `hybrid` (e.g., Conv1D + LSTM)
- **Window size**
  - Number of past days used to predict the next day
- **Epochs**
  - Number of training iterations

Under the hood (`src/train_tf.py`):

- Prices are **scaled** (MinMax) for stable training
- Sliding windows are generated for supervised learning
- Data is split into train / validation
- A Keras model is built & trained (EarlyStopping on val_loss)
- Predictions and next-day forecast are **inverse-transformed** back to price units

### 3. Evaluation & Visualization

After training, the app shows:

- **Actual vs Predicted** validation chart
- Metrics:
  - **MAE (val)** – mean absolute error on validation window
  - **RMSE (val)** – root mean squared error
  - **Next-day forecast (Close)** – model’s prediction for the next time step
- **Training curve**:
  - Train loss vs validation loss per epoch
- Optionally saves `last_run_predictions.csv` for further analysis

---

## 🛠 Tech Stack

- **Frontend / App**
  - [Streamlit](https://streamlit.io/)
  - Plotly Express for interactive charts

- **Modeling**
  - TensorFlow / Keras (LSTM, Conv1D, hybrid models)
  - NumPy, Pandas

- **Data**
  - Alpha Vantage (via API key) and/or alternative data sources (e.g., Stooq)
  - Local caching to avoid hitting rate limits repeatedly

---

## 🖼 Screenshots

> You can update these paths to match your actual filenames in the `assets/` folder.

```text
assets/
├── app_load.png       # Initial data loaded view
├── app_hybrid.png     # Hybrid model result
└── app_lstm.png       # LSTM model result
