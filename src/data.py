# Đọc dữ liệu Rossmann, kiểm tra cột, chuyển kiểu dữ liệu, xử lý missing values, lọc cửa hàng/ngày và lưu dữ liệu đã xử lý.
# src/data.py
# Nhiệm vụ: đọc và chuẩn bị dữ liệu gốc.
import pandas as pd

def load_data(path):
    # 1. Đọc file CSV.
    df = pd.read_csv(path)
    df = df[df['Store'] == 1] # Chỉ lấy store 1

    # 2. Kiểm tra các cột bắt buộc.
    # 3. Chuyển Date sang kiểu datetime.
    # 4. Xử lý các dòng không hợp lệ.
    df = df.sort_values('Date').reset_index(drop=True)
    # Bước 1: chuẩn hóa về string trước (vì đang lẫn '0' và 0)
    df['StateHoliday'] = df['StateHoliday'].astype(str)

    # Bước 2: map sang số
    mapping_holiday = {'0': 1, 'a': 2, 'b': 3, 'c': 4}
    df['StateHoliday'] = df['StateHoliday'].map(mapping_holiday)
    cols = ['Sales', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 'Open']
    df = df[cols]
    # 5. Trả về DataFrame.
    return df

if __name__ == '__main__':
    path = r'C:\Users\admin\Pictures\20261_ProjectI_202416766\data\raw\raw_data.csv'
    load_data(path)