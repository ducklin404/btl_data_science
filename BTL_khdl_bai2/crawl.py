import aiohttp
import asyncio
import pandas as pd
import datetime
import time

# Cấu hình
url_list = "https://gateway.chotot.com/v1/public/ad-listing"
url_detail = "https://gateway.chotot.com/v1/public/ad-listing/{}"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/121.0.0.0 Safari/537.36"
}

max_ads = 3000     # số tin cần lấy
limit = 20         # mỗi lần gọi list API lấy 20 tin
batch_size = 200   # mỗi batch lấy 200 tin
pause_time = 5     # nghỉ 5 giây sau mỗi batch
concurrency = 10   # số request chạy song song tối đa trong 1 batch


# Hàm tiện ích: lấy giá trị từ params/parameters
def get_value(ad_detail, params, parameters, key):
    for source in [params, parameters]:
        for p in source:
            if p.get("id") == key:
                return p.get("value", "")
    return ad_detail.get(key, "")


# Lấy chi tiết tin
async def fetch_detail(session, list_id, sem, retries=5):
    async with sem:  # semaphore để hạn chế số request song song
        for attempt in range(retries):
            try:
                async with session.get(url_detail.format(list_id), headers=headers, timeout=10) as r:
                    if r.status != 200:
                        raise ValueError(f"HTTP {r.status}")
                    detail = await r.json()

                    ad_detail = detail.get("ad", {})
                    if not ad_detail:
                        raise ValueError("Không có dữ liệu ad")

                    params_detail = detail.get("params", [])
                    parameters = detail.get("parameters", [])

                    # Ngày đăng
                    list_time = ad_detail.get("list_time")
                    ngay_dang = datetime.datetime.fromtimestamp(
                        int(list_time) / 1000
                    ).strftime("%Y-%m-%d") if list_time else ""

                    record = {
                        "Ngày đăng": ngay_dang,
                        "Năm SX": get_value(ad_detail, params_detail, parameters, "mfdate"),
                        "Xuất xứ": get_value(ad_detail, params_detail, parameters, "carorigin"),
                        "Địa điểm": get_value(ad_detail, params_detail, parameters, "address"),
                        "Kiểu dáng": get_value(ad_detail, params_detail, parameters, "cartype"),
                        "Số km đã đi": get_value(ad_detail, params_detail, parameters, "mileage_v2"),
                        "Hộp số": get_value(ad_detail, params_detail, parameters, "gearbox"),
                        "Tình trạng": get_value(ad_detail, params_detail, parameters, "condition_ad"),
                        "Nhiên liệu": get_value(ad_detail, params_detail, parameters, "fuel"),
                        "Giá": ad_detail.get("price", 0)
                    }

                    # Nếu dữ liệu quá thiếu thì thử lại
                    if not record["Năm SX"] and not record["Nhiên liệu"]:
                        raise ValueError("Dữ liệu thiếu")

                    return record
            except Exception as e:
                print(f"Lỗi khi lấy {list_id} (attempt {attempt+1}): {e}")
                await asyncio.sleep(1)
        return None


# Lấy danh sách ID tin
async def fetch_list(session, offset):
    params = {
        "limit": limit,
        "o": offset,
        "cg": 2010,          # danh mục ô tô
        "region_v2": 12000,  # Hà Nội
    }
    async with session.get(url_list, headers=headers, params=params, timeout=10) as r:
        if r.status != 200:
            print(f"Lỗi fetch_list offset={offset}, HTTP {r.status}")
            return []
        res = await r.json()
        ads = res.get("ads", [])
        return [ad["list_id"] for ad in ads]


# Hàm chia batch
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# Hàm chính
async def main():
    async with aiohttp.ClientSession() as session:
        list_ids = []

        # Lấy danh sách ID
        for offset in range(0, max_ads, limit):
            ids = await fetch_list(session, offset)
            if not ids:
                break
            list_ids.extend(ids)
            if len(list_ids) >= max_ads:
                break

        print(f"🔎 Thu được {len(list_ids)} ID tin")

        records = []
        sem = asyncio.Semaphore(concurrency)

        for batch_num, batch in enumerate(chunks(list_ids, batch_size), 1):
            print(f"Đang xử lý batch {batch_num} ({len(batch)} tin)...")

            tasks = [fetch_detail(session, lid, sem) for lid in batch]
            results = await asyncio.gather(*tasks)

            records.extend([r for r in results if r])

            print(f"Hoàn thành batch {batch_num}, tổng cộng {len(records)} tin")
            await asyncio.sleep(pause_time)  # nghỉ sau mỗi batch

        df = pd.DataFrame(records).drop_duplicates()
        df.to_csv("oto_chitiet.csv", index=False, encoding="utf-8-sig")
        print(f"Đã lưu {len(df)} tin vào oto_chitiet.csv")


if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    print("Thời gian chạy:", round(time.time() - start, 2), "giây")
