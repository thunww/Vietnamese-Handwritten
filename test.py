import json

# Danh sách ký tự giữ lại (ví dụ)
chars_to_keep = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Chữ số
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',  # Chữ thường a-z
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'à', 'á', 'â', 'ã', 'è', 'é', 'ê', 'ì', 'í', 'ò', 'ó', 'ô', 'õ',  # Chữ tiếng Việt thường
    'ù', 'ú', 'ý', 'ă', 'đ', 'ĩ', 'ũ', 'ơ', 'ư', 'ạ', 'ả', 'ấ', 'ầ',
    'ẩ', 'ẫ', 'ậ', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'ẹ', 'ẻ', 'ẽ', 'ế', 'ề',
    'ể', 'ễ', 'ệ', 'ỉ', 'ị', 'ọ', 'ỏ', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ớ',
    'ờ', 'ở', 'ỡ', 'ợ', 'ụ', 'ủ', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'ỳ', 'ỵ',
    'ỷ', 'ỹ',
    ' ', '.', ',', ':', '!', '?', '(', ')', '-',  # Dấu câu cơ bản
    '@', '#', '$', '%', '&', '*', '+', '=', '/', '<', '>', '_', '^', '|', '{', '}', '[', ']'  # Thêm ký tự đặc biệt
]

# Tạo char_to_idx mới
new_char_to_idx = {char: str(idx + 1) for idx, char in enumerate(chars_to_keep)}

# Lưu vào file mới
with open("D:/Vietnamese-handwritten/data/char_to_idx_simplified.json", 'w', encoding='utf-8') as f:
    json.dump(new_char_to_idx, f, ensure_ascii=False, indent=4)

print(f"Số ký tự mới: {len(new_char_to_idx)}")
print(f"Số lớp đầu ra (num_classes): {len(new_char_to_idx) + 1}")
