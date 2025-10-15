import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 1. Đọc dữ liệu gốc
df = pd.read_csv("du_lieu_oto.csv")

# 2. Chọn các cột số cần chuẩn hóa
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Nếu dữ liệu số nằm trong dạng text (vd: "120,000 km"), thì chuyển đổi:
for col in ["Năm SX", "Số km đã đi", "Giá"]:  # đổi theo tên cột thật trong file
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"[^0-9]", "", regex=True), errors="coerce")
        if col not in numeric_cols:
            numeric_cols.append(col)

# 3. Chuẩn hóa dữ liệu số
# Dùng StandardScaler
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Nếu muốn thử loại khác thì thay scaler = MinMaxScaler() hoặc RobustScaler()

# 4. Lưu ra file mới
df_scaled.to_csv("du_lieu_oto_scaled.csv", index=False, encoding="utf-8-sig")

print("Đã chuẩn hóa xong. File mới lưu tại: du_lieu_oto_scaled.csv")
