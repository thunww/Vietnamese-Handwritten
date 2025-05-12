# preprocess.py
import cv2
import numpy as np
from PIL import Image, ImageOps

def resize_and_pad(image, target_size=(118, 2167)):
    """
    Resize ảnh giữ nguyên tỷ lệ, sau đó pad trắng cho đủ kích thước target.
    """
    h, w = image.shape
    scale = min(target_size[1] / w, target_size[0] / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Tạo ảnh trắng
    canvas = np.ones(target_size, dtype=np.uint8) * 255
    pad_top = (target_size[0] - new_h) // 2
    pad_left = (target_size[1] - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = image_resized

    return canvas

def preprocess_image(image_path, target_size=(118, 2167)):
    """
    Tiền xử lý ảnh OCR để sử dụng cho mô hình CRNN và VietOCR.
    """
    try:
        # Đọc ảnh
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Không thể đọc ảnh.")
        
        # Chuyển sang grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Binarization - làm nét văn bản
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Loại bỏ nhiễu nhỏ (morphology opening)
        kernel = np.ones((2, 2), np.uint8)
        clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Resize về đúng kích thước đầu vào mô hình với padding trắng nếu cần
        resized = resize_and_pad(clean, target_size)

        # Chuyển sang dạng ảnh PIL để tương thích cả VietOCR
        pil_image = Image.fromarray(resized).convert("L")

        # Chuẩn hóa ánh sáng (optional, có thể thử nghiệm)
        pil_image = ImageOps.autocontrast(pil_image)

        return pil_image

    except Exception as e:
        raise ValueError(f"Lỗi xử lý ảnh: {str(e)}")


def load_data(data_dir, label_file, image_size=(118, 2167), threshold_method='simple'):
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
                pil_img = preprocess_image(img_path, image_size, method=threshold_method)
                np_img = np.array(pil_img).reshape((image_size[0], image_size[1], 1))
                label = label_dict.get(filename, filename.split('.')[0])
                images.append(np_img)
                labels.append(label)
            except Exception as e:
                print(f"Lỗi khi xử lý {filename}: {e}")

    images = np.array(images, dtype=np.uint8)
    return images, labels
