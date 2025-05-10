import os
import cv2
import numpy as np
from PIL import Image

def preprocess_image(image_path, target_size=(118, 2167), method='adaptive'):
    """
    Xử lý ảnh đầu vào cho mô hình CRNN.

    Args:
        image_path (str): Đường dẫn đến file ảnh.
        target_size (tuple): Kích thước mục tiêu (chiều cao, chiều rộng).
        method (str): Phương pháp thresholding ('simple', 'adaptive', 'otsu').

    Returns:
        numpy.ndarray: Ảnh đã xử lý với shape (chiều cao, chiều rộng, 1).
    """
    # Đọc ảnh dưới dạng grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ {image_path}")

    # Làm mờ ảnh để giảm nhiễu
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Chọn phương pháp thresholding
    if method == 'simple':
        _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    elif method == 'adaptive':
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    elif method == 'otsu':
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

    # Resize ảnh về kích thước mục tiêu (chiều rộng, chiều cao)
    thresh = cv2.resize(thresh, (target_size[1], target_size[0]))

    # Thêm chiều kênh (1) để có shape (chiều cao, chiều rộng, 1)
    thresh = thresh[..., np.newaxis]

    # Chuẩn hóa giá trị pixel về [0, 1] (tùy chọn, nếu mô hình yêu cầu)
    thresh = thresh / 255.0

    return thresh

def load_data(data_dir, label_file, image_size=(118, 2167)):
    images = []
    labels = []
    
    # Đọc nhãn từ file labels.txt
    label_dict = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                label_dict[parts[0]] = parts[1]
    
    print("Image size:", image_size)
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(data_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                img = cv2.resize(img, (image_size[1], image_size[0]))
                img = img.reshape((image_size[0], image_size[1], 1))
                label = label_dict.get(filename, filename.split('.')[0])
                images.append(img)
                labels.append(label)
    
    images = np.array(images)
    return images, labels