# Tạo đặc trưng đầu vào, chuẩn bị chuỗi thời gian, tạo sliding window 30 ngày, chia train/validation/test theo thời gian.
# src/features.py
# Nhiệm vụ: biến dữ liệu thành đầu vào cho mô hình.
import torch
import numpy as np
import pandas as pd
from data import load_data

def create_sliding_windows(data, lookback=30):
    # 1. Sắp xếp dữ liệu theo ngày.
    # 2. Lấy 30 ngày liên tiếp làm X.
    FEATURE_COLS = [1, 2, 3, 4, 5]   # 5 cột feature (KHÔNG gồm Sales)
    TARGET_COL = 0                 # cột Sales

    X_list, y_list = [], []
    for i in range(len(data) - lookback):
        X_list.append(data[i : i + lookback, FEATURE_COLS])   # (30, 5)
        y_list.append(data[i + lookback, TARGET_COL])         # scalar (Sales ngày 31)

    X = torch.stack(X_list)   # (912, 30, 5)
    y = torch.stack(y_list)   # (912,)
    n = len(X)                    # 912
    train_end = int(0.7 * n)      # 70%
    val_end   = int(0.85 * n)     # 15% tiếp theo (70% → 85%)

    X_train, y_train = X[:train_end],          y[:train_end]
    X_val,   y_val   = X[train_end:val_end],   y[train_end:val_end]
    X_test,  y_test  = X[val_end:],            y[val_end:]
    # 3. Lấy ngày tiếp theo làm y.
    # 4. Trả về X, y với shape phù hợp.
    return (X_train, X_val, X_test, y_train, y_val, y_test)

path = r'C:\Users\admin\Pictures\20261_ProjectI_202416766\data\raw\raw_data.csv'
df = load_data(path)
data = torch.tensor(df.values)
(X_train, X_val, X_test, y_train, y_val, y_test) = create_sliding_windows(data)
torch.save(X_train, r'C:\Users\admin\Pictures\20261_ProjectI_202416766\data\processed\X_train.pt')
torch.save(X_val, r'C:\Users\admin\Pictures\20261_ProjectI_202416766\data\processed\X_val.pt')
torch.save(X_test, r'C:\Users\admin\Pictures\20261_ProjectI_202416766\data\processed\X_test.pt')
torch.save(y_train, r'C:\Users\admin\Pictures\20261_ProjectI_202416766\data\processed\y_train.pt')
torch.save(y_val, r'C:\Users\admin\Pictures\20261_ProjectI_202416766\data\processed\y_val.pt')
torch.save(y_test, r'C:\Users\admin\Pictures\20261_ProjectI_202416766\data\processed\y_test.pt')