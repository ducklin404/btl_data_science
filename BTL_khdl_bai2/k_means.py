import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Đọc dữ liệu gốc
df = pd.read_csv("du_lieu_oto.csv")

# Các cột numeric để phân cụm
features = ["Giá", "Năm SX", "Số km đã đi"]
X = df[features].dropna()

# 2. Chuẩn hóa dữ liệu để chạy K-means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Chạy K-means
kmeans = KMeans(n_clusters=3, random_state=42)
df.loc[X.index, "cluster"] = kmeans.fit_predict(X_scaled)

# 4. Bảng phân tích phân khúc
cluster_summary = df.groupby("cluster").agg(
    so_xe = ("cluster", "count"),
    gia_tb = ("Giá", lambda x: x.mean()/1_000_000),
    nam_sx_tb = ("Năm SX", "mean"),
    km_tb = ("Số km đã đi", "mean"),
    gia_min = ("Giá", lambda x: x.min()/1_000_000),
    gia_max = ("Giá", lambda x: x.max()/1_000_000)
).reset_index()

# Đổi tên cột
cluster_summary = cluster_summary.rename(columns={
    "gia_tb": "Giá TB (triệu)",
    "gia_min": "Giá Min (triệu)",
    "gia_max": "Giá Max (triệu)",
    "nam_sx_tb": "Năm SX TB",
    "km_tb": "Km TB"
})

# Gán nhãn phân khúc
labels = {
    0: "Xe cũ, giá rẻ",
    1: "Xe tầm trung",
    2: "Xe mới, cao cấp"
}
cluster_summary["Phân khúc"] = cluster_summary["cluster"].map(labels)

# Bỏ cột cluster
cluster_summary = cluster_summary.drop(columns=["cluster"])

# Hiển thị bảng
print("\n📊 Bảng phân tích phân khúc (theo triệu đồng):")
print(cluster_summary)

# 5. Xuất bảng ra file CSV
cluster_summary.to_csv("phan_tich_phan_khuc.csv", index=False, encoding="utf-8-sig")

# 6. Chuẩn bị dữ liệu để vẽ biểu đồ 3D
df["Giá (triệu)"] = df["Giá"] / 1_000_000   # đổi giá sang triệu đồng

# Vẽ scatter 3D
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection="3d")

colors = ["purple", "green", "gold"]
labels_text = ["Xe cũ, giá rẻ", "Xe tầm trung", "Xe mới, cao cấp"]

for cluster, color, label in zip(range(3), colors, labels_text):
    cluster_points = df[df["cluster"] == cluster]
    ax.scatter(cluster_points["Năm SX"], 
               cluster_points["Số km đã đi"], 
               cluster_points["Giá (triệu)"],
               c=color, label=label, alpha=0.6)

ax.set_xlabel("Năm sản xuất")
ax.set_ylabel("Số km đã đi")
ax.set_zlabel("Giá (triệu đồng)")
ax.set_title("Phân cụm ô tô 3D (K-means)")
ax.legend()

plt.show()
