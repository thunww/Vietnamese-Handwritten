import os
import cv2

train_data_dir = 'D:/Vietnamese-handwritten/data/train'
image_sizes = []
for filename in os.listdir(train_data_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(train_data_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            image_sizes.append(img.shape)  # (height, width)
        else:
            print(f"Không đọc được ảnh: {img_path}")
print("Kích thước ảnh gốc:", set(image_sizes))