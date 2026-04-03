"""
07_lstm_model.py
=================
LSTM model for taxi demand prediction using PyTorch.

Architecture: 2 LSTM layers (128, 64) + Dense (Sahin 2022 style).
Trained on top 5 demand zones. Sequence length = 24h.

References:
- Sahin 2022: LSTM architecture for taxi demand
"""

import pandas as pd
import numpy as np
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN = 24
EPOCHS = 15
BATCH_SIZE = 512
PATIENCE = 3

BASE_FEATURES = [
    'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour', 'is_holiday',
    'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow',
    'precipitation', 'temperature', 'wind_speed', 'visibility',
    'lag_1', 'lag_24', 'lag_168', 'zone_mean_demand',
]

EVENT_FEATURES = [
    'has_event', 'event_count',
    'evt_commercial_shoot', 'evt_film_production',
    'evt_performance', 'evt_tv_production',
]

TARGET = 'trip_count'

device = torch.device('cpu')  # MPS runs out of memory with other processes
print(f"Device: {device}")


class LSTMModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, 128, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x[:, -1, :])
        x = self.relu(self.fc1(x))
        return self.fc2(x).squeeze()


def create_sequences(data, features, target_col, seq_len):
    X, y = [], []
    values = data[features].values
    targets = data[target_col].values
    for i in range(seq_len, len(values)):
        X.append(values[i-seq_len:i])
        y.append(targets[i])
    return np.array(X), np.array(y)


def train_lstm(X_train, y_train, X_val, y_val, n_features):
    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    y_v = torch.FloatTensor(y_val).to(device)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                              batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMModel(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return model


def evaluate(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
    return {'model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}


# ── Load & Filter ─────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet(os.path.join(DATA_DIR, 'manhattan_features.parquet'))
for col in df.select_dtypes(include='bool').columns:
    df[col] = df[col].astype(int)

top_zones = df.groupby('PULocationID')[TARGET].mean().nlargest(5).index.tolist()
df = df[df['PULocationID'].isin(top_zones) & (df['year'] >= 2023)].copy()
print(f"  Top 5 zones: {top_zones}, shape: {df.shape}")

results = []

for feat_name, features in [('no events', BASE_FEATURES),
                              ('with events', BASE_FEATURES + EVENT_FEATURES)]:
    print(f"\n{'='*50}")
    print(f"LSTM ({feat_name}) - {len(features)} features")
    print(f"{'='*50}")

    all_true, all_pred = [], []
    t0 = time.time()

    for zone in top_zones:
        zd = df[df['PULocationID'] == zone].sort_values('pickup_hour').copy()

        scaler = MinMaxScaler()
        zd[features] = scaler.fit_transform(zd[features])
        tgt_scaler = MinMaxScaler()
        zd[[TARGET]] = tgt_scaler.fit_transform(zd[[TARGET]])

        train_data = zd[zd['year'] <= 2024]
        test_data = zd[zd['year'] == 2025]

        X_train, y_train = create_sequences(train_data, features, TARGET, SEQ_LEN)
        X_test, y_test = create_sequences(test_data, features, TARGET, SEQ_LEN)

        val_size = int(0.2 * len(X_train))
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        X_train_sub, y_train_sub = X_train[:-val_size], y_train[:-val_size]

        model = train_lstm(X_train_sub, y_train_sub, X_val, y_val, len(features))

        model.eval()
        with torch.no_grad():
            pred = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()

        pred_orig = tgt_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
        true_orig = tgt_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        all_true.extend(true_orig)
        all_pred.extend(pred_orig)
        print(f"  Zone {zone}: {len(true_orig)} predictions")

    elapsed = time.time() - t0
    y_true_arr = np.array(all_true)
    y_pred_arr = np.clip(np.array(all_pred), 0, None)

    result = evaluate(y_true_arr, y_pred_arr, f'LSTM ({feat_name})')
    results.append(result)
    print(f"\n  RMSE: {result['RMSE']:.3f}, MAE: {result['MAE']:.3f}, "
          f"R2: {result['R2']:.4f} ({elapsed:.1f}s)")

# ── Combine results ────────────────────────────────────────────────────
prev_path = os.path.join(OUT_DIR, 'model_results.csv')
if os.path.exists(prev_path):
    prev = pd.read_csv(prev_path)
    combined = pd.concat([prev, pd.DataFrame(results)], ignore_index=True)
else:
    combined = pd.DataFrame(results)

combined.to_csv(os.path.join(OUT_DIR, 'model_results_all.csv'), index=False)

print("\n" + "="*50)
print("ALL RESULTS")
print("="*50)
combined = combined.sort_values('RMSE')
for _, row in combined.iterrows():
    print(f"  {row['model']:<35s} RMSE={row['RMSE']:.3f} R2={row['R2']:.4f}")

print("\nDone!")
