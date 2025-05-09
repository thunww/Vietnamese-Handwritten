import os
import cv2
import numpy as np
from PIL import Image
import logging
import tensorflow as tf
from module.crnn_model import build_model
from tensorflow.keras.backend import ctc_decode
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"TensorFlow version: {tf.__version__}")

try:
    from module.vietnamese_ocr import recognize_text
    from module.vietocr_module import load_vietocr_model, recognize_text_vietocr
except ImportError as e:
    logger.error(f"Lỗi import module: {str(e)}")
    raise ImportError("Vui lòng cài đặt VietOCR và cấu hình module đúng cách")

from preprocess import preprocess_image

class OCR:
    def __init__(self, char_to_idx_path, custom_model_path=None):
        with open(char_to_idx_path, 'r', encoding='utf-8') as f:
            self.char_to_idx = json.load(f)
        self.idx_to_char = {int(idx): char for char, idx in self.char_to_idx.items()}
        self.vietocr_predictor = None
        self.custom_model = None
        
        try:
            logger.info("Đang tải model VietOCR...")
            self.vietocr_predictor = load_vietocr_model(use_gpu=False)
            logger.info("Đã tải xong model VietOCR")
        except Exception as e:
            logger.warning(f"Không thể tải model VietOCR: {str(e)}")
        
        if custom_model_path:
            try:
                logger.info(f"Đang tải mô hình tự huấn luyện từ {custom_model_path}...")
                self.custom_model = load_custom_model(custom_model_path, len(self.char_to_idx) + 1)
                logger.info("Đã tải mô hình tự huấn luyện")
            except Exception as e:
                logger.warning(f"Không thể tải mô hình tự huấn luyện: {str(e)}")

    def preprocess_for_crnn(self, image_path, target_size=(118, 2167)):
        img = preprocess_image(image_path, target_size)
        if img is None:
            raise ValueError(f"Không thể xử lý ảnh từ {image_path}")
        return np.expand_dims(img, axis=0)

    def recognize(self, image_path, use_vietocr=True, preprocess=True):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy file ảnh tại {image_path}")
        
        processed_image_path = image_path
        if preprocess:
            try:
                logger.info("Đang tiền xử lý ảnh...")
                processed_image = preprocess_image(image_path)
                filename = os.path.basename(image_path)
                dirname = os.path.dirname(image_path) or '.'
                processed_image_path = os.path.join(dirname, "processed_" + filename)
                processed_image.save(processed_image_path)
                logger.info(f"Đã lưu ảnh đã xử lý tại {processed_image_path}")
            except Exception as e:
                logger.error(f"Lỗi khi xử lý ảnh: {str(e)}")
                logger.warning("Sử dụng ảnh gốc...")
                processed_image = Image.open(image_path)
                processed_image_path = image_path
        
        if use_vietocr:
            if self.vietocr_predictor is None:
                raise ValueError("VietOCR model không được tải.")
            logger.info("Sử dụng VietOCR để nhận dạng...")
            return recognize_text_vietocr(processed_image_path, self.vietocr_predictor)
        else:
            if self.custom_model is None or self.idx_to_char is None:
                raise ValueError("Không có mô hình tự huấn luyện hoặc từ điển ký tự.")
            logger.info("Sử dụng model tự train để nhận dạng...")
            
            img = self.preprocess_for_crnn(processed_image_path)
            y_pred = self.custom_model.predict(img)
            input_length = np.ones((1,)) * y_pred.shape[1]
            decoded, _ = ctc_decode(y_pred, input_length=input_length, greedy=True)
            
            prediction = []
            for seq in decoded[0].numpy():
                pred = ''.join(self.idx_to_char.get(idx, '') for idx in seq if idx != -1 and idx in self.idx_to_char)
                prediction.append(pred)
            
            return prediction[0]

def preprocess_image(image_path, method='adaptive'):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
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
    return Image.fromarray(thresh)

def load_custom_model(custom_model_path, num_classes):
    if not os.path.exists(custom_model_path):
        raise FileNotFoundError(f"Không tìm thấy mô hình tại {custom_model_path}")
    try:
        model = build_model(num_classes)
        model.load_weights(custom_model_path)
        return model
    except Exception as e:
        raise ValueError(f"Lỗi khi tải mô hình tự huấn luyện: {str(e)}")

def main():
    char_to_idx_path = "D:/Vietnamese-handwritten/data/char_to_idx.json"
    custom_model_path = "D:/Vietnamese-handwritten/data/final_model.keras"
    ocr = OCR(char_to_idx_path, custom_model_path=custom_model_path)
    image_path = input("Nhập đường dẫn đến file ảnh: ")
    use_vietocr = input("Sử dụng VietOCR? (y/n): ").lower() == 'y'
    try:
        text = ocr.recognize(image_path, use_vietocr)
        print(f"Văn bản được nhận dạng: {text}")
    except Exception as e:
        print(f"Lỗi: {str(e)}")

if __name__ == "__main__":
    main()