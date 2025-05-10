from collections import Counter
import re

with open("D:/Vietnamese-handwritten/data/char_to_idx.json", "r", encoding="utf-8") as f:
    raw = f.read()

# Regex để lấy toàn bộ key
keys = re.findall(r'"(.*?)"\s*:', raw)

# Đếm số lần xuất hiện từng key
counter = Counter(keys)
duplicates = {k: v for k, v in counter.items() if v > 1}

print(f"Tổng số key trong raw JSON: {len(keys)}")
print(f"Số key duy nhất sau json.load: {len(set(keys))}")
print("Các key bị trùng:")
for k, v in duplicates.items():
    print(f"{k} xuất hiện {v} lần")
