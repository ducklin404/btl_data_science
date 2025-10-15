import pandas as pd

# Đọc dữ liệu gốc
df = pd.read_csv("oto_chitiet.csv")

# ===== Xử lý dữ liệu thiếu =====
# Nếu "Tình trạng" là "Mới" và "Số km đã đi" rỗng thì gán 0
df.loc[df["Tình trạng"].eq("Mới") & df["Số km đã đi"].isna(), "Số km đã đi"] = "0 km"

# Thay NaN trong "Kiểu dáng" thành "Unknown"
df["Kiểu dáng"] = df["Kiểu dáng"].fillna("Unknown")

# Lấy phần số trong cột "Năm SX" và chuyển sang numeric
df["Năm SX"] = df["Năm SX"].astype(str).str.extract(r"(\d+)")
df["Năm SX"] = pd.to_numeric(df["Năm SX"], errors="coerce")

# Chuẩn hóa "Số km đã đi" -> bỏ chữ "km", giữ số
df["Số km đã đi"] = df["Số km đã đi"].astype(str).str.replace(" km", "", regex=False)
df["Số km đã đi"] = pd.to_numeric(df["Số km đã đi"], errors="coerce")

# Chuẩn hóa cột "Giá" -> bỏ ký tự không phải số
df["Giá"] = df["Giá"].astype(str).str.replace(r"[^0-9]", "", regex=True)
df["Giá"] = pd.to_numeric(df["Giá"], errors="coerce")

# ===== Lọc ngoại lai theo quy tắc 3 sigma =====
initial_rows = len(df)

# --- Cho Năm SX ---
mean_year = df["Năm SX"].mean()
std_year = df["Năm SX"].std()
lower_year, upper_year = mean_year - 3*std_year, mean_year + 3*std_year

df = df[(df["Năm SX"] >= lower_year) & (df["Năm SX"] <= upper_year)]
after_year_rows = len(df)
print("Số hàng bị xoá do ngoại lai ở Năm SX:", initial_rows - after_year_rows)

# --- Cho Số km đã đi ---
mean_km = df["Số km đã đi"].mean()
std_km = df["Số km đã đi"].std()
lower_km, upper_km = mean_km - 3*std_km, mean_km + 3*std_km

df = df[(df["Số km đã đi"] >= lower_km) & (df["Số km đã đi"] <= upper_km)]
after_km_rows = len(df)
print("Số hàng bị xoá do ngoại lai ở Số km đã đi:", after_year_rows - after_km_rows)

# --- Cho Giá ---
mean_price = df["Giá"].mean()
std_price = df["Giá"].std()
lower_price, upper_price = mean_price - 3*std_price, mean_price + 3*std_price

df = df[(df["Giá"] >= lower_price) & (df["Giá"] <= upper_price)]
after_price_rows = len(df)
print("Số hàng bị xoá do ngoại lai ở Giá:", after_km_rows - after_price_rows)

# ===== Xuất dữ liệu đã xử lý =====
df.to_csv("du_lieu_oto.csv", index=False, header=True, encoding="utf-8-sig")
