import os
from io import StringIO
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dir_path = os.path.dirname(__file__)

output_folder = dir_path + os.sep + "data"
INPUT_CSV = output_folder + os.sep + "du_lieu_oto.csv"
OUT_DIR = "./plots_main"
CLEANED_PATH = "./cleaned_car_data.csv"

os.makedirs(OUT_DIR, exist_ok=True)

def extract_khuvuc(dia_diem):
    if pd.isna(dia_diem):
        return None
    s = str(dia_diem)
    for marker in ["Quận", "Huyện", "Thị xã", "Xã", "Phường"]:
        if marker in s:
            start = s.find(marker)
            rest = s[start:]
            parts = rest.split(",")
            return parts[0].strip()
    return s.split(",")[0].strip()

def simple_kde(xs, data, bw):
    data = np.asarray(data)
    factor = 1.0 / (bw * math.sqrt(2 * math.pi))
    return np.array([np.mean(factor * np.exp(-0.5 * ((x - data) / bw) ** 2)) for x in xs])

def load_and_clean(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"INPUT_CSV not found: {path}")
    df = pd.read_csv(path, parse_dates=["Ngày đăng"], dayfirst=False)
    df['Năm SX'] = pd.to_numeric(df.get('Năm SX'), errors='coerce')
    df['Số km đã đi'] = pd.to_numeric(df.get('Số km đã đi'), errors='coerce')
    df['Giá'] = pd.to_numeric(df.get('Giá'), errors='coerce')
    df['Khu vực'] = df['Địa điểm'].apply(extract_khuvuc)
    return df

def plot_line_mean_price_by_year(df, out_dir):
    g = df.groupby('Năm SX')['Giá'].agg(['mean','count','std']).reset_index().dropna()
    plt.figure(figsize=(10,5))
    plt.plot(g['Năm SX'], g['mean'], marker='o')
    plt.title('Giá trung bình theo Năm SX')
    plt.xlabel('Năm sản xuất')
    plt.ylabel('Giá trung bình (VND)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "line_mean_price_by_year.png"), dpi=200)
    plt.close()
    return g

def plot_scatter_price_vs_km(df, out_dir):
    year_vals = df['Năm SX'].fillna(df['Năm SX'].median())
    sizes = (year_vals - year_vals.min()).fillna(0) * 2 + 20
    plt.figure(figsize=(9,6))
    plt.scatter(df['Số km đã đi'], df['Giá'], s=sizes)
    plt.title('Giá vs Số km đã đi')
    plt.xlabel('Số km đã đi')
    plt.ylabel('Giá (VND)')
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter_price_vs_km.png"), dpi=200)
    plt.close()

def plot_errorbar_mean_price_by_year(g, out_dir):
    valid = g[g['count'] >= 2]
    plt.figure(figsize=(10,5))
    plt.errorbar(valid['Năm SX'], valid['mean'], yerr=valid['std'], marker='o', linestyle='-')
    plt.title('Giá trung bình theo Năm SX với thanh lỗi (std)')
    plt.xlabel('Năm sản xuất')
    plt.ylabel('Giá (VND)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "errorbar_mean_price_by_year.png"), dpi=200)
    plt.close()

def plot_contour_price_km_year(df, out_dir):
    cont_df = df[['Số km đã đi','Năm SX','Giá']].dropna()
    if len(cont_df) >= 10 and cont_df['Số km đã đi'].nunique() >= 5:
        plt.figure(figsize=(9,6))
        plt.tricontourf(cont_df['Số km đã đi'], cont_df['Năm SX'], cont_df['Giá'], levels=12)
        plt.title('Contour of Giá over (Số km, Năm SX)')
        plt.xlabel('Số km đã đi')
        plt.ylabel('Năm SX')
        cbar = plt.colorbar()
        cbar.set_label('Giá (VND)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "contour_price_km_year.png"), dpi=200)
        plt.close()
    else:
        print("Contour skipped: not enough diverse points (need >=10 records and varied km).")

def plot_histogram_price_kde(df, out_dir):
    prices = df['Giá'].dropna().values
    plt.figure(figsize=(9,5))
    plt.hist(prices, bins=20, density=True, alpha=0.7)
    if len(prices) > 1:
        std_p = prices.std()
        bw = 1.06 * std_p * (len(prices) ** (-1/5))
        xs = np.linspace(prices.min(), prices.max(), 300)
        ys = simple_kde(xs, prices, bw)
        plt.plot(xs, ys)
    plt.title('Histogram of Giá (density) with KDE estimate')
    plt.xlabel('Giá (VND)')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "histogram_price_kde.png"), dpi=200)
    plt.close()

def plot_3d_scatter(df, out_dir):
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['Năm SX'], df['Số km đã đi'], df['Giá'], s=20)
    ax.set_xlabel('Năm SX')
    ax.set_ylabel('Số km đã đi')
    ax.set_zlabel('Giá (VND)')
    ax.set_title('3D scatter: Năm SX vs Số km vs Giá')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "3d_scatter_year_km_price.png"), dpi=200)
    plt.close()

def plot_bar_count_by_khuvuc(df, out_dir):
    counts = df['Khu vực'].value_counts().nlargest(15)
    plt.figure(figsize=(11,6))
    counts.plot(kind='bar')
    plt.title('Số lượng tin theo Khu vực (heuristic từ Địa điểm)')
    plt.xlabel('Khu vực')
    plt.ylabel('Số tin')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bar_count_by_khuvuc.png"), dpi=200)
    plt.close()

def main():
    df = load_and_clean(INPUT_CSV)
    g = plot_line_mean_price_by_year(df, OUT_DIR)
    plot_scatter_price_vs_km(df, OUT_DIR)
    plot_errorbar_mean_price_by_year(g, OUT_DIR)
    plot_contour_price_km_year(df, OUT_DIR)
    plot_histogram_price_kde(df, OUT_DIR)
    plot_3d_scatter(df, OUT_DIR)
    plot_bar_count_by_khuvuc(df, OUT_DIR)
    print("Plots saved to:", os.path.abspath(OUT_DIR))

if __name__ == "__main__":
    main()
