import cv2
import numpy as np
from PIL import Image, ImageOps
import os


def resize_and_pad(image, target_size=(32, 200)):
    """
    Resize ảnh giữ nguyên tỷ lệ, sau đó pad trắng cho đủ kích thước target.
    """
    h, w = image.shape[:2]
    scale = min(target_size[1] / w, target_size[0] / h)  # Giữ tỷ lệ của ảnh gốc
    new_w = int(w * scale)
    new_h = int(h * scale)

    image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Tạo ảnh trắng (RGB)
    canvas = np.ones((target_size[0], target_size[1], 3), dtype=np.uint8) * 255
    pad_top = (target_size[0] - new_h) // 2
    pad_left = (target_size[1] - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = image_resized

    return canvas


def preprocess_image(image_path, target_size=(32, 200)):
    """
    Tiền xử lý ảnh OCR để sử dụng cho mô hình VietOCR.
    """
    try:
        # Đọc ảnh màu
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Không thể đọc ảnh.")
        
        # Resize và pad ảnh
        resized = resize_and_pad(img, target_size)

        # Chuyển ảnh sang grayscale (1 kênh màu)
        img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Chuyển sang ảnh PIL grayscale
        pil_image = Image.fromarray(img_gray).convert("L")  # "L" mode is grayscale

        # Chuẩn hóa ánh sáng (optional, giữ lại để tăng độ tương phản nếu cần)
        pil_image = ImageOps.autocontrast(pil_image)

        return pil_image

    except Exception as e:
        raise ValueError(f"Lỗi xử lý ảnh: {str(e)}")



def load_data(data_dir, label_file, image_size=(32, 200), threshold_method='simple'):
    """
    Load và xử lý ảnh từ thư mục và nhãn từ file.
    """
    images = []
    labels = []

    # Load nhãn
    label_dict = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                label_dict[parts[0]] = parts[1]

    print("Image size:", image_size)

    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(data_dir, filename)
            try:
                pil_img = preprocess_image(img_path, image_size)
                np_img = np.array(pil_img).reshape((image_size[0], image_size[1], 3))
                label = label_dict.get(filename, filename.split('.')[0])
                images.append(np_img)
                labels.append(label)
            except Exception as e:
                print(f"Lỗi khi xử lý {filename}: {e}")

    images = np.array(images, dtype=np.uint8)
    return images, labels
