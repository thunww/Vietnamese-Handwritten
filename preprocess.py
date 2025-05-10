import os
import cv2
import numpy as np
from PIL import Image

# Hàm xử lý ảnh
def preprocess_image(image_path, method='adaptive'):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ {image_path}")
    
    # Chuyển ảnh sang grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Làm mờ ảnh để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
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
    
    # Chuyển đổi mảng NumPy sang Image (PIL) nếu cần
    return Image.fromarray(thresh)


def load_data(data_dir, label_file, image_size=(118, 2167)):
    images = []
    labels = []
    
    # Đọc nhãn từ file labels.txt
    label_dict = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')  # Giả sử labels.txt có định dạng: filename.jpg<tab>nhãn
            if len(parts) == 2:
                label_dict[parts[0]] = parts[1]
    
    # Kiểm tra và chắc chắn image_size là (height, width)
    print("Image size:", image_size)  # Debugging: In ra kích thước hình ảnh
    
    # Lặp qua các file ảnh trong thư mục
    for filename in os.listdir(data_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(data_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # Thay đổi kích thước ảnh về (height, width)
                img = cv2.resize(img, (image_size[1], image_size[0]))  # (width, height)
                img = img.reshape((image_size[0], image_size[1], 1))  # (height, width, 1)
                
                # Lấy nhãn từ label_dict
                label = label_dict.get(filename, filename.split('.')[0])  # Fallback nếu không tìm thấy nhãn
                images.append(img)
                labels.append(label)
    
    images = np.array(images)
    return images, labels