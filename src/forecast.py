# Tải model đã huấn luyện, tạo dự báo trên dữ liệu đầu vào và lưu kết quả dự báo.
# src/forecast.py
# Nhiệm vụ: tạo dự báo từ model đã huấn luyện.

def predict(model, X):
    # 1. Chuyển model sang chế độ evaluation.
    # 2. Tắt tính gradient khi dự báo.
    # 3. Tạo prediction.
    # 4. Chuyển kết quả về đơn vị doanh số gốc
    #    nếu trước đó đã chuẩn hóa target.
    pass







import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# 1. Xây dựng mô hình LSTM
class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden_size=64):
        super().__init__()

        # Đọc chuỗi dữ liệu gồm 30 bước thời gian.
        # Mỗi bước có n_features đặc trưng.
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Chuyển trạng thái LSTM thành một giá trị dự báo.
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch_size, 30, n_features)
        output, (hidden, cell) = self.lstm(x)

        # Lấy output tại bước thời gian cuối cùng.
        last_output = output[:, -1, :]

        # Dự báo doanh số ngày tiếp theo.
        prediction = self.fc(last_output)

        # Trả về tensor có shape (batch_size,)
        return prediction.squeeze(-1)


# 2. Chuyển dữ liệu đã tạo sliding window thành tensor.
# X_train: (N, 30, n_features)
# y_train: (N,)
X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor)

# Không shuffle theo thời gian ở bước chia dữ liệu.
# Với các window đã tạo từ train, shuffle mẫu trong train
# thường được chấp nhận khi huấn luyện mô hình dự báo.
loader = DataLoader(dataset, batch_size=64, shuffle=True)


# 3. Khởi tạo mô hình, hàm mất mát và bộ tối ưu.
model = LSTMForecaster(n_features=X_tensor.shape[2])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 4. Huấn luyện mô hình.
for epoch in range(20):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        optimizer.zero_grad()

        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)

    average_loss = total_loss / len(dataset)
    print(f"Epoch {epoch + 1}: loss={average_loss:.4f}")
