import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import os

# Đường dẫn file
dir_path = os.path.dirname(__file__)

# 1. Đọc dữ liệu gốc
df = pd.read_csv(os.path.join(dir_path, "data", "du_lieu_oto.csv"))

# Các cột numeric để phân cụm
features = ["Giá", "Năm SX", "Số km đã đi"]
X = df[features].dropna()

# 2. Chuẩn hóa dữ liệu để chạy K-means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Chạy K-means
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df.loc[X.index, "cluster"] = kmeans.fit_predict(X_scaled)

# 4. Bảng phân tích phân khúc (cluster_summary còn giữ cột 'cluster' để mapping)
cluster_summary = df.groupby("cluster").agg(
    so_xe=("cluster", "count"),
    gia_tb=("Giá", lambda x: x.mean() / 1_000_000),
    nam_sx_tb=("Năm SX", "mean"),
    km_tb=("Số km đã đi", "mean"),
    gia_min=("Giá", lambda x: x.min() / 1_000_000),
    gia_max=("Giá", lambda x: x.max() / 1_000_000),
).reset_index()

# 4.1. Gán nhãn phân khúc dựa trên giá trung bình (gia_tb) chứ không dựa vào số cluster
# Sắp xếp cluster theo giá trung bình tăng dần
cluster_order = cluster_summary.sort_values("gia_tb")["cluster"].tolist()

# Nhãn theo thứ tự giá: rẻ -> trung -> cao
rank_labels = ["Xe cũ, giá rẻ", "Xe tầm trung", "Xe mới, cao cấp"]

# Map từ mã cluster sang tên phân khúc
cluster_label_map = {cl: lbl for cl, lbl in zip(cluster_order, rank_labels)}

# Thêm cột phân khúc vào bảng summary
cluster_summary["Phân khúc"] = cluster_summary["cluster"].map(cluster_label_map)

# 4.2. Đổi tên cột cho đẹp
cluster_summary_renamed = cluster_summary.rename(columns={
    "gia_tb": "Giá TB (triệu)",
    "gia_min": "Giá Min (triệu)",
    "gia_max": "Giá Max (triệu)",
    "nam_sx_tb": "Năm SX TB",
    "km_tb": "Km TB"
})

# 4.3. Bảng xuất ra file
cluster_summary_export = cluster_summary_renamed.drop(columns=["cluster"])

# Hiển thị bảng
print("\nBảng phân tích phân khúc (theo triệu đồng):")
print(cluster_summary_export)

# 5. Xuất bảng ra file CSV
output_path = os.path.join(dir_path, "phan_tich_phan_khuc.csv")
cluster_summary_export.to_csv(output_path, index=False, encoding="utf-8-sig")

# 6. Chuẩn bị dữ liệu để vẽ biểu đồ 3D
df["Giá (triệu)"] = df["Giá"] / 1_000_000   
df["Phân khúc"] = df["cluster"].map(cluster_label_map)

# Vẽ scatter 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

colors = ["purple", "green", "gold"]

# Đảm bảo duyệt cluster theo thứ tự tăng dần để màu ổn định
for cluster_id, color in zip(sorted(df["cluster"].dropna().unique()), colors):
    cluster_points = df[df["cluster"] == cluster_id]
    label = cluster_label_map.get(cluster_id, f"Cluster {cluster_id}")
    ax.scatter(
        cluster_points["Năm SX"],
        cluster_points["Số km đã đi"],
        cluster_points["Giá (triệu)"],
        c=color,
        label=label,
        alpha=0.6
    )

ax.set_xlabel("Năm sản xuất")
ax.set_ylabel("Số km đã đi")
ax.set_zlabel("Giá (triệu đồng)")
ax.set_title("Phân cụm ô tô 3D (K-means)")
ax.legend()

plt.show()
