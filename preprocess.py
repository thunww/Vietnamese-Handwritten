import cv2
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_image(image_path, target_size=(118, 2167)):
    if not os.path.exists(image_path):
        logger.warning(f"Image {image_path} not found.")
        return None

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.warning(f"Cannot read image {image_path}")
        return None

    original_h, original_w = img.shape
    target_h, target_w = target_size

    # Tính tỷ lệ scale (giữ nguyên tỷ lệ gốc)
    scale = min(target_w / original_w, target_h / original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    # Resize theo tỷ lệ
    img = cv2.resize(img, (new_w, new_h))

    # Padding ảnh vào khung trắng (255) đúng size
    padded_img = np.ones((target_h, target_w), dtype=np.uint8) * 255
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    padded_img[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = img

    # Normalize và thêm chiều channel
    padded_img = padded_img.astype('float32') / 255.0
    padded_img = np.expand_dims(padded_img, axis=-1)
    return padded_img

def load_data(data_dir, label_file, target_size=(118, 2167)):
    images = []
    labels = []
    skipped = 0
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' not in line:
                logger.warning(f"Invalid line in {label_file}: {line.strip()}")
                skipped += 1
                continue
            img_name, text = line.strip().split(':', 1)
            img_path = os.path.join(data_dir, img_name)
            img = preprocess_image(img_path, target_size)
            if img is not None:
                images.append(img)
                labels.append(text)
            else:
                skipped += 1
    logger.info(f"Loaded {len(images)} images, skipped {skipped} entries from {label_file}")
    return np.array(images), labels

if __name__ == "__main__":
    train_images, train_labels = load_data('data/train', 'data/train/labels.txt')
    logger.info(f"Loaded {len(train_images)} images and {len(train_labels)} labels")