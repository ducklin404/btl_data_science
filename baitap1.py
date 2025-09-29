import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import re

# Cấu hình cơ bản
CSV_PATH = "Student Insomnia and Educational Outcomes Dataset.csv"
YEAR_COL = "1. What is your year of study?"
HOURS_COL = "4. On average, how many hours of sleep do you get on a typical day?"
ALPHA = 0.05
MU0 = 8.0

# Hàm chuyển đổi dữ liệu giờ ngủ sang số
def map_hours_to_numeric(s):
    if pd.isna(s):
        return np.nan
    text = str(s).strip().lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\bhours?\b", "", text).strip()

    m = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", text)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return (a + b) / 2.0

    m = re.search(r"\b(less than|under|<)\s*(\d+(?:\.\d+)?)\b", text)
    if m:
        x = float(m.group(2))
        return max(0.0, x - 0.5)

    m = re.search(r"\b(more than|over|above|>)\s*(\d+(?:\.\d+)?)\b", text)
    if m:
        x = float(m.group(2))
        return x + 0.5

    m = re.search(r"\b(\d+(?:\.\d+)?)\s*\+\b", text)
    if m:
        x = float(m.group(1))
        return x + 0.5

    m = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
    if m:
        return float(m.group(1))

    return np.nan

def line():
    return "-" * 72

# Đọc dữ liệu
df_full = pd.read_csv(CSV_PATH)

# Lọc sinh viên đại học
if YEAR_COL in df_full.columns:
    year_clean = df_full[YEAR_COL].astype(str).str.strip().str.lower()
    mask_undergrad = (year_clean != "graduate student") & (year_clean != "graduate") & (year_clean != "postgraduate")
    removed = (~mask_undergrad).sum()
    df = df_full.loc[mask_undergrad].reset_index(drop=True)
else:
    removed = 0
    df = df_full.copy()

work = df.copy()
work['hours_numeric'] = work[HOURS_COL].apply(map_hours_to_numeric)
hours = work['hours_numeric'].dropna().to_numpy()

# Báo cáo kết quả
report_lines = []
report_lines.append(line())
report_lines.append("BƯỚC 1: KHẢO SÁT & LÀM SẠCH DỮ LIỆU")
report_lines.append(line())
report_lines.append(f"Tệp dữ liệu: {CSV_PATH}")
report_lines.append(f"Số dòng loại bỏ (không phải ĐH): {removed}")
report_lines.append(f"Tổng số bản ghi (ĐH): {len(work)}")
report_lines.append(f"Số bản ghi hợp lệ về giờ ngủ: {len(hours)}")
report_lines.append("Tần suất các câu trả lời cột giờ ngủ:")
for k, v in work[HOURS_COL].value_counts(dropna=False).items():
    report_lines.append(f"  - {k}: {v}")

# Chuẩn hóa dữ liệu
report_lines.append("")
report_lines.append(line())
report_lines.append("BƯỚC 2: CHUẨN HÓA (Z-score)")
report_lines.append(line())

xbar = np.mean(hours) if len(hours) else np.nan
s = np.std(hours, ddof=1) if len(hours) > 1 else np.nan
if np.isnan(s) or s == 0:
    report_lines.append("Không đủ dữ liệu để tính z-score.")
    work['hours_z'] = np.nan
else:
    valid_mask = work['hours_numeric'].notna()
    work.loc[valid_mask, 'hours_z'] = (work.loc[valid_mask, 'hours_numeric'] - xbar) / s
    report_lines.append(f"Trung bình mẫu (x̄): {xbar:.3f} giờ")
    report_lines.append(f"Độ lệch chuẩn mẫu (s): {s:.3f} giờ")
    report_lines.append("Đã thêm cột: 'hours_z'")
    
    outliers_mask = work['hours_z'].abs() > 3
    n_outliers = outliers_mask.sum()
    work = work.loc[~outliers_mask].reset_index(drop=True)
    report_lines.append(f"Loại bỏ {n_outliers} giá trị ngoại lai với |z| > 3")
    hours = work['hours_numeric'].dropna().to_numpy()
    xbar = np.mean(hours) if len(hours) else np.nan
    s = np.std(hours, ddof=1) if len(hours) > 1 else np.nan

# Khoảng tin cậy
report_lines.append("")
report_lines.append(line())
report_lines.append("BƯỚC 3: ƯỚC LƯỢNG KHOẢNG TIN CẬY 95%")
report_lines.append(line())

n = len(hours)
if n >= 2 and not np.isnan(s):
    tcrit = stats.t.ppf(1 - ALPHA/2, df=n-1)
    se = s / np.sqrt(n)
    ci_low = xbar - tcrit * se
    ci_high = xbar + tcrit * se
    report_lines.append(f"n = {n}, α = {ALPHA:.2f}, t* = {tcrit:.3f}")
    report_lines.append(f"Khoảng tin cậy 95% cho trung bình: [{ci_low:.3f}, {ci_high:.3f}]")
else:
    report_lines.append("Không đủ dữ liệu để tính khoảng tin cậy.")

# Kiểm định giả thuyết
report_lines.append("")
report_lines.append(line())
report_lines.append("BƯỚC 4: KIỂM ĐỊNH GIẢ THUYẾT (t-test 1 mẫu)")
report_lines.append(line())
report_lines.append(f"H0: μ = {MU0} giờ   vs   H1: μ ≠ {MU0} giờ")

if n >= 2 and not np.isnan(s):
    tstat, pval = stats.ttest_1samp(hours, popmean=MU0, alternative="two-sided")
    decision = "BÁC BỎ H0" if pval < ALPHA else "CHƯA ĐỦ BẰNG CHỨNG ĐỂ BÁC BỎ H0"
    report_lines.append(f"t = {tstat:.4f}")
    report_lines.append(f"p-value = {pval:.4f}")
    report_lines.append(f"Kết luận ở mức α={ALPHA:.2f}: {decision}")
    if pval < ALPHA:
        report_lines.append("Kết luận: Số giờ ngủ trung bình KHÁC 8 giờ/ngày (có ý nghĩa thống kê).")
    else:
        report_lines.append("Kết luận: Không đủ bằng chứng để nói số giờ ngủ trung bình khác 8 giờ/ngày.")
else:
    report_lines.append("Không đủ dữ liệu để thực hiện kiểm định.")

# Xuất kết quả
report = "\n".join(report_lines)
print(report)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_csv = f"du_lieu_sach_{timestamp}.csv"
out_txt = f"bao_cao_{timestamp}.txt"
work.to_csv(out_csv, index=False)
with open(out_txt, "w", encoding="utf-8") as f:
    f.write(report)

print("\n" + line())
print(f"Đã lưu dữ liệu sạch: {out_csv}")
print(f"Đã lưu báo cáo:      {out_txt}")
print(line())
