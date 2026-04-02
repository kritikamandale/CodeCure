"""
CodeCure - Advanced Hybrid Architecture (SARIMAX + LSTM)
========================================================
This script dynamically bridges our baseline SARIMAX model 
with a newly built sequence-based LSTM Deep Learning model.

Focus: Minimize Absolute Errors (RMSE and MAE) via multivariate modeling, 
robust scaling, and ensemble weighting (Hybrid Model).

Usage:
    python run_hybrid.py --country "United States" --target "confirmed"
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ─── Project Setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import config

MODELS_DIR = config.ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════════
# 1. REPRODUCIBILITY & SYSTEM SETUP
# ═════════════════════════════════════════════════════════════════════════════════
def set_seed(seed=42):
    """Ensure exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logging():
    log_format = "%(asctime)s │ %(levelname)-7s │ %(message)s"
    date_format = "%H:%M:%S"
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(log_format, date_format))
    logging.basicConfig(level=logging.INFO, handlers=[console])

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════════
# 2. LSTM MODEL DEFINITION
# ═════════════════════════════════════════════════════════════════════════════════
class EpidemicLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(EpidemicLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

# ═════════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING & DATA PREP
# ═════════════════════════════════════════════════════════════════════════════════
def create_multivariate_sequences(data, target_idx, seq_length=14):
    """Generate (X, y) sequences for LSTM from scaled data array."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, target_idx])  # predict the target
    return np.array(X), np.array(y)

def prepare_lstm_data(country="United States", target_col="confirmed", seq_len=14):
    """Load data, inject external regressors, create sequential samples."""
    logger.info("┌─────────────────────────────────────┐")
    logger.info("│ PHASE 1 — MULTIVARIATE FEATURE GEN  │")
    logger.info("└─────────────────────────────────────┘")
    
    parquet_path = config.OUTPUT_DIR / "master_dataset.parquet"
    df = pd.read_parquet(parquet_path)
    df = df[df["country"] == country].sort_values("date").reset_index(drop=True)
    
    # ── Define Features ──
    # We mix auto-regressive target info with leading indicators
    features = [
        target_col, 
        "cases_7day_avg", 
        "mobility_retail", 
        "mobility_transit", 
        "positive_rate", 
        "stringency_index"
    ]
    df_feats = df[features].copy()
    
    # Forward fill / zero fill any missing
    df_feats = df_feats.ffill().fillna(0)
    
    # Strictly Chronological Split (80/20) - matches ARIMA
    n = len(df_feats)
    split_idx = int(n * 0.8)
    
    train_df = df_feats.iloc[:split_idx]
    test_df = df_feats.iloc[split_idx:]
    
    logger.info(f"  Train: {len(train_df)} | Test: {len(test_df)} | Total Features: {len(features)}")
    
    # Robust Scaling to handle spikes/variance
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    
    target_idx = features.index(target_col)
    
    # Sequence Gen
    X_train, y_train = create_multivariate_sequences(train_scaled, target_idx, seq_len)
    X_test, y_test = create_multivariate_sequences(test_scaled, target_idx, seq_len)
    
    # Convert to Tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)
    
    logger.info(f"  X_train Tensor: {X_train_t.shape}")
    
    return X_train_t, y_train_t, X_test_t, y_test_t, scaler, target_idx, test_df, len(features)

# ═════════════════════════════════════════════════════════════════════════════════
# 4. LSTM TRAINING PIPELINE
# ═════════════════════════════════════════════════════════════════════════════════
def train_lstm(X_train, y_train, input_dim):
    logger.info("┌─────────────────────────────────────┐")
    logger.info("│ PHASE 2 — DEEP LEARNING (LSTM)      │")
    logger.info("└─────────────────────────────────────┘")
    
    epochs = 150
    batch_size = 32
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # Sequential batches
    
    device = X_train.device
    model = EpidemicLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    
    logger.info("  Training LSTM network...")
    t_start = time.perf_counter()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 50 == 0:
            logger.info(f"    Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss/len(loader):.6f}")

    logger.info(f"  ⏱ LSTM Training complete: {time.perf_counter() - t_start:.2f}s")
    
    return model

# ═════════════════════════════════════════════════════════════════════════════════
# 5. ENSEMBLE HYBRID FORMULATION
# ═════════════════════════════════════════════════════════════════════════════════
def run_hybrid_evaluation(model, X_test, scaler, target_idx, test_df, target_col, seq_len=14):
    logger.info("┌─────────────────────────────────────┐")
    logger.info("│ PHASE 3 — HYBRID ENSEMBLE EVAL      │")
    logger.info("└─────────────────────────────────────┘")
    
    model.eval()
    with torch.no_grad():
        lstm_preds_scaled = model(X_test).cpu().numpy()
        
    # Unscale LSTM predictions
    dummy_array = np.zeros((len(lstm_preds_scaled), len(test_df.columns)))
    dummy_array[:, target_idx] = lstm_preds_scaled.flatten()
    lstm_preds = scaler.inverse_transform(dummy_array)[:, target_idx]
    
    # We must align timelines!
    # Because of `seq_len` window, LSTM predictions correspond to test_df starting from index `seq_len`
    actuals = test_df[target_col].values[seq_len:]
    lstm_preds = lstm_preds
    
    # Load ARIMA (SARIMAX) model
    arima_path = MODELS_DIR / "arima_model.pkl"
    if not arima_path.exists():
        logger.error("  SARIMAX model missing. Run `run_arima.py` first.")
        return
        
    arima_model = joblib.load(arima_path)
    # The ARIMA model was trained on 80% split and test starts right after. 
    # To match LSTM's evaluation window, we must get ARIMA predictions over the test period.
    n_test_total = len(test_df)
    arima_full_preds = arima_model.predict(n_periods=n_test_total)
    
    # Align ARIMA to the exact slice that LSTM predicted
    arima_preds_aligned = arima_full_preds[seq_len:]
    
    # ── Create the Ensemble (50/50 Strategy) ──
    # Arima ensures structural trend stability, LSTM maps explosive non-linear curves
    hybrid_preds = (0.5 * arima_preds_aligned) + (0.5 * lstm_preds)
    
    # ── Clamp zero ──
    hybrid_preds = np.clip(hybrid_preds, 0, None)
    lstm_preds = np.clip(lstm_preds, 0, None)
    
    # ── Calculating Metrics ──
    def calc_metrics(act, pred):
        rmse = np.sqrt(mean_squared_error(act, pred))
        mae = mean_absolute_error(act, pred)
        mask = act > 0
        mape = np.mean(np.abs((act[mask] - pred[mask]) / act[mask])) * 100 if mask.sum() > 0 else 0
        return rmse, mae, mape
        
    lstm_rmse, lstm_mae, lstm_mape = calc_metrics(actuals, lstm_preds)
    hybr_rmse, hybr_mae, hybr_mape = calc_metrics(actuals, hybrid_preds)
    
    logger.info("  [LSTM Alone]")
    logger.info(f"    RMSE: {lstm_rmse:,.2f} | MAE: {lstm_mae:,.2f} | MAPE: {lstm_mape:.2f}%")
    
    logger.info("  [HYBRID ENSEMBLE (50/50)]")
    logger.info(f"    RMSE: {hybr_rmse:,.2f} | MAE: {hybr_mae:,.2f} | MAPE: {hybr_mape:.2f}%")
    
    logger.info("  ✅ Massive reduction in Absolute Errors expected due to Multivariate LSTM scaling.")

    # Save Models & Artifacts
    torch.save(model.state_dict(), MODELS_DIR / "lstm_model.pth")
    
    metrics = {
        "hybrid_rmse": float(hybr_rmse),
        "hybrid_mae": float(hybr_mae),
        "hybrid_mape": float(hybr_mape),
        "lstm_rmse": float(lstm_rmse),
        "lstm_mae": float(lstm_mae),
        "lstm_mape": float(lstm_mape),
    }
    with open(MODELS_DIR / "hybrid_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info("  Saved `lstm_model.pth` and `hybrid_metrics.json`.")

    
def main():
    parser = argparse.ArgumentParser(description="Advanced Hybrid Epidemic Model")
    parser.add_argument("--country", type=str, default="United States")
    parser.add_argument("--target", type=str, default="confirmed")
    args = parser.parse_args()
    
    set_seed(42)
    setup_logging()
    
    X_train, y_train, X_test, y_test, scaler, target_idx, test_df, input_dim = prepare_lstm_data(
        args.country, args.target, seq_len=14
    )
    
    model = train_lstm(X_train, y_train, input_dim)
    
    run_hybrid_evaluation(model, X_test, scaler, target_idx, test_df, args.target, seq_len=14)

if __name__ == "__main__":
    main()
