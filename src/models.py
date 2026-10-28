# Định nghĩa kiến trúc LSTM và baseline model. Không đặt toàn bộ vòng lặp huấn luyện ở đây.
# src/models.py
# Nhiệm vụ: định nghĩa kiến trúc mô hình.
import torch
import torch.nn as nn
import numpy as np


# ============================================================
# 1. NAIVE MODEL
# Dự đoán y[t+1] = y[t] (giá trị cuối cùng trong window)
# ============================================================
class NaiveModel:
    """
    Naive forecast: dự đoán ngày tiếp theo = giá trị ngày cuối trong window.
    Input X: (batch, window, n_features) — Sales ở index target_idx
    """
    def __init__(self, target_idx: int):
        self.target_idx = target_idx

    def predict(self, X):
        # X: torch.Tensor (batch, window, n_features) hoặc numpy
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        # lấy giá trị target ở ngày cuối cùng của window
        return X[:, -1, self.target_idx]          # (batch,)


# ============================================================
# 2. SEASONAL NAIVE MODEL
# Dự đoán y[t+1] = y[t+1 - season] (cùng ngày tuần trước)
# Với dữ liệu bán hàng: season = 7 (chu kỳ tuần)
# ============================================================
class SeasonalNaiveModel:
    """
    Seasonal naive: dự đoán = giá trị cùng ngày trong tuần trước.
    season=7 → lấy giá trị cách đây 7 ngày (cuối window - 7 + 1).
    """
    def __init__(self, target_idx: int, season: int = 7):
        self.target_idx = target_idx
        self.season = season

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        # window có W ngày: index cuối = W-1 (ngày t)
        # muốn lấy ngày t+1-season → index = W-1 - (season-1) = W - season
        idx = X.shape[1] - self.season
        if idx < 0:
            raise ValueError(f"window ({X.shape[1]}) phải >= season ({self.season})")
        return X[:, idx, self.target_idx]         # (batch,)


# ============================================================
# 3. MOVING AVERAGE 7 DAYS
# Dự đoán y[t+1] = trung bình 7 ngày cuối trong window
# ============================================================
class MovingAvgModel:
    """
    Moving average: dự đoán = trung bình `window_avg` ngày gần nhất.
    """
    def __init__(self, target_idx: int, window_avg: int = 7):
        self.target_idx = target_idx
        self.window_avg = window_avg

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        # lấy `window_avg` ngày cuối, cột target, rồi mean
        last = X[:, -self.window_avg:, self.target_idx]   # (batch, window_avg)
        return last.mean(dim=1)                            # (batch,)


# ============================================================
# 4. LSTM MODEL
# Input: (batch, window, n_features) → Output: (batch,)
# ============================================================
class LSTMModel(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,                  # input (batch, seq, feat)
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)    # dự đoán 1 giá trị Sales

    def forward(self, x):
        # x: (batch, window, n_features)
        out, _ = self.lstm(x)                  # out: (batch, window, hidden)
        last_hidden = out[:, -1, :]            # lấy hidden state cuối
        return self.fc(last_hidden).squeeze(-1)  # (batch,)