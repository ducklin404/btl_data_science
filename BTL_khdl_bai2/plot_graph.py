# eda_main_plots.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", context="notebook", rc={"figure.dpi": 150})

# config 
CSV_PATH = "du_lieu_oto_scaled.csv"
OUTPUT_DIR = Path("plots_main")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load Data
df = pd.read_csv(CSV_PATH)
if 'Ngày đăng' in df.columns:
    df['Ngày đăng'] = pd.to_datetime(df['Ngày đăng'], errors='coerce')

# Detect columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in df.columns if c not in numeric_cols and not np.issubdtype(df[c].dtype, np.datetime64)]
print("Numeric:", numeric_cols)
print("Categorical:", categorical_cols)

# Phân bố giá (Histogram + Boxplot)
if 'Giá' in df.columns:
    fig, axs = plt.subplots(1, 2, figsize=(10,4))
    sns.histplot(df['Giá'].dropna(), kde=True, ax=axs[0])
    axs[0].axvline(0, color='gray', linestyle='--')
    axs[0].set_title("Phân bố Giá (chuẩn hóa, 0 = trung bình)")
    sns.boxplot(x=df['Giá'], ax=axs[1])
    axs[1].axvline(0, color='gray', linestyle='--')
    axs[1].set_title("Boxplot Giá (chuẩn hóa)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "gia_distribution.png", dpi=150)
    plt.close(fig)

# Heatmap tương quan
if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(7,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Ma trận tương quan (dữ liệu đã co giãn)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "corr_heatmap.png", dpi=150)
    plt.close()

# Scatter: Năm SX vs Giá
if all(col in df.columns for col in ['Năm SX','Giá']):
    plt.figure(figsize=(6,5))
    sns.regplot(x='Năm SX', y='Giá', data=df, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    plt.axvline(0, color='gray', linestyle='--')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Tương quan Năm SX vs Giá (dữ liệu đã co giãn)")
    plt.xlabel("Năm SX (Z-score / scaled)")
    plt.ylabel("Giá (Z-score / scaled)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scatter_namsx_gia.png", dpi=150)
    plt.close()

# Scatter: Số km đã đi vs Giá
if all(col in df.columns for col in ['Số km đã đi','Giá']):
    plt.figure(figsize=(6,5))
    sns.regplot(x='Số km đã đi', y='Giá', data=df, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    plt.axvline(0, color='gray', linestyle='--')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Tương quan Số km đã đi vs Giá (dữ liệu đã co giãn)")
    plt.xlabel("Số km đã đi (Z-score / scaled)")
    plt.ylabel("Giá (Z-score / scaled)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scatter_km_gia.png", dpi=150)
    plt.close()

# Barplot: Giá trung bình theo Xuất xứ
if all(col in df.columns for col in ['Xuất xứ','Giá']):
    plt.figure(figsize=(8,4))
    avg = df.groupby('Xuất xứ')['Giá'].mean().sort_values(ascending=False).reset_index()
    sns.barplot(data=avg, x='Xuất xứ', y='Giá')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Giá trung bình theo Xuất xứ (đã co giãn)")
    plt.xlabel("Xuất xứ")
    plt.ylabel("Giá (Z-score / scaled)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bar_xuatxu_gia.png", dpi=150)
    plt.close()


print("Đã lưu các biểu đồ chính trong thư mục:", OUTPUT_DIR.resolve())
