import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset, DataLoader


FEATURE_COLS = ['HR', 'RR', 'SpO2', 'HR_mask', 'RR_mask', 'SpO2_mask']
TARGET_COLS = ['HR_clean', 'RR_clean', 'SpO2_clean']
FAULT_COLS = ['HR_fault', 'RR_fault', 'SpO2_fault']


def fill_model_inputs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ['HR', 'RR', 'SpO2']:
        df[col] = df[col].interpolate().bfill().ffill()

    return df


def create_reliability_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['HR_reliability'] = 1 - df['HR_fault']
    df['RR_reliability'] = 1 - df['RR_fault']
    df['SpO2_reliability'] = 1 - df['SpO2_fault']

    return df


def create_sequences(df: pd.DataFrame, window_size: int = 20):
    sequences = []
    targets = []
    faults = []

    for case_id in df['caseid'].unique():
        case_data = df[df['caseid'] == case_id].reset_index(drop=True)

        x_case = case_data[FEATURE_COLS].values
        y_case = case_data[TARGET_COLS].values
        f_case = case_data[FAULT_COLS].values

        for i in range(len(case_data) - window_size):
            sequences.append(x_case[i:i + window_size])
            targets.append(y_case[i + window_size - 1])
            faults.append(f_case[i + window_size - 1])

    return np.array(sequences), np.array(targets), np.array(faults)


def scale_sequences(sequences: np.ndarray, targets: np.ndarray):
    num_samples, seq_len, num_features = sequences.shape

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_reshaped = sequences.reshape(-1, num_features)
    x_scaled = x_scaler.fit_transform(x_reshaped).reshape(num_samples, seq_len, num_features)
    y_scaled = y_scaler.fit_transform(targets)

    return x_scaled, y_scaled, x_scaler, y_scaler


class VitalDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class GRURegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


def prepare_dataloaders(x_scaled, y_scaled, batch_size: int = 64):
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y_scaled, test_size=0.2, random_state=42
    )

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = VitalDataset(x_train_tensor, y_train_tensor)
    test_dataset = VitalDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, x_train, x_test, y_train, y_test, x_train_tensor, x_test_tensor


def train_gru_model(model, train_loader, epochs: int = 10, lr: float = 0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        history.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    return history


def predict_gru(model, x_test_tensor, y_scaler):
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test_tensor).cpu().numpy()

    y_pred_inv = y_scaler.inverse_transform(y_pred)
    return y_pred_inv


def predict_with_uncertainty(model, x_test_tensor, y_scaler, n_samples: int = 20):
    model.train()

    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x_test_tensor).cpu().numpy()
            pred = y_scaler.inverse_transform(pred)
            preds.append(pred)

    preds = np.array(preds)
    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)

    return pred_mean, pred_std


def inverse_targets(y_test, y_scaler):
    return y_scaler.inverse_transform(y_test)


def evaluate_predictions(y_true_inv, y_pred_inv):
    overall_mae = mean_absolute_error(y_true_inv, y_pred_inv)
    overall_rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))

    signal_metrics = {}
    signal_names = ['HR', 'RR', 'SpO2']

    for i, signal in enumerate(signal_names):
        mae_signal = mean_absolute_error(y_true_inv[:, i], y_pred_inv[:, i])
        rmse_signal = np.sqrt(mean_squared_error(y_true_inv[:, i], y_pred_inv[:, i]))
        signal_metrics[signal] = {
            'mae': mae_signal,
            'rmse': rmse_signal,
        }

    return overall_mae, overall_rmse, signal_metrics


def reconstruct_observed_test_signals(x_test, x_scaler):
    x_test_last = x_test[:, -1, :]

    temp_full = np.zeros((x_test_last.shape[0], x_test_last.shape[1]))
    temp_full[:, :] = x_test_last[:, :]

    x_test_last_inv = x_scaler.inverse_transform(temp_full)
    observed_test_signals = x_test_last_inv[:, :3]

    return observed_test_signals


def compute_error_based_reliability(observed_test_signals, y_pred_inv):
    hr_error = np.abs(observed_test_signals[:, 0] - y_pred_inv[:, 0])
    rr_error = np.abs(observed_test_signals[:, 1] - y_pred_inv[:, 1])
    spo2_error = np.abs(observed_test_signals[:, 2] - y_pred_inv[:, 2])

    alpha_hr = 0.2
    alpha_rr = 0.3
    alpha_spo2 = 0.5

    hr_rel = np.exp(-alpha_hr * hr_error)
    rr_rel = np.exp(-alpha_rr * rr_error)
    spo2_rel = np.exp(-alpha_spo2 * spo2_error)

    return hr_error, rr_error, spo2_error, hr_rel, rr_rel, spo2_rel
